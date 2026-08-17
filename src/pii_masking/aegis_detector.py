"""AEGIS ONNX 모델을 실행하고 예측을 OCR token과 bbox에 연결한다.

SearchView 문자열을 모델 입력 token으로 변환하고, 연속된 BIO 라벨을 하나의
개인정보 범위로 합친다. 모델이 반환한 문자 위치는 SearchView의 위치 매핑을
통해 원본 OCR token ID와 bbox로 변환한다.

코드 검토 순서
1. ``FULL_ENTITY_TO_CATEGORY``에서 AEGIS 18개 라벨과 내부 카테고리의 대응을
   확인한다.
2. ``ENTITY_THRESHOLDS``에서 라벨별 후처리 cutoff를 확인한다. 이 값은 ONNX
   모델 파일에 자동 내장된 공통 threshold가 아니다.
3. ``AegisOnnxNer.__init__()``에서 tokenizer와 ONNX Runtime session을 로드한다.
4. ``predict_tokens()``에서 logits를 softmax 확률과 BIO token 예측으로 바꾼다.
5. ``detect()``에서 threshold를 통과한 연속 B/I token을 하나의 NER span으로
   합친다.
6. ``detect_aegis_spans()``에서 ``detector_engine``을 통해 문자 span을 OCR token과
   bbox로 역매핑한다.

AEGIS는 이미지나 OCR bbox를 직접 읽지 않는다. SearchView 문자열만 분석하며,
이미지 위치 연결은 공통 엔진이 담당한다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import BertWordPieceTokenizer

from .detector_engine import PiiSpan, ProviderMatch, detect_with_provider
from .models import PreprocessedDocument
from .paths import MODEL_DIR
from .search_views import (
    DEFAULT_SEARCH_VIEW_SETTINGS,
    SearchView,
    SearchViewSettings,
)
from .address_expander import extend_address_spans


MODEL_NAME = "YATAV-ENT/aegis-personal-pii-ner"
MODEL_REVISION = "v2"
DEFAULT_MODEL_DIR = MODEL_DIR

# 기본 운영 모드는 규칙만으로 어려운 이름·주소에 제한한다. 전체 라벨 모드는
# 관리자 정책 실험에서만 선택하며 구조화 번호는 이후 규칙 검증과 병합한다.
ENTITY_TO_CATEGORY = {
    "SURNAME": "person_name",
    "GIVENNAME": "person_name",
    "CITY": "address",
    "STREET": "address",
    "ZIPCODE": "address",
    "BUILDINGNUM": "address",
}

FULL_ENTITY_TO_CATEGORY = {
    **ENTITY_TO_CATEGORY,
    "USERNAME": "username",
    "EMAIL": "email_address",
    "TELEPHONENUM": "phone_number",
    "DATEOFBIRTH": "date",
    "CREDITCARDNUMBER": "account_card_number",
    "IDCARD": "id_card_number",
    "IP_ADDRESS": "ip_address",
    "PASSWORD": "credential_secret",
    "ACCOUNTNUM": "account_card_number",
    "DRIVERLICENSENUM": "driver_license_number",
    "TIME": "time",
    "COMPANY": "company_name",
}

ENTITY_THRESHOLDS = {
    "SURNAME": 0.85,
    "GIVENNAME": 0.85,
    "CITY": 0.80,
    "STREET": 0.80,
    "ZIPCODE": 0.90,
    "BUILDINGNUM": 0.90,
    "USERNAME": 0.90,
    "EMAIL": 0.85,
    "TELEPHONENUM": 0.85,
    "DATEOFBIRTH": 0.85,
    "CREDITCARDNUMBER": 0.90,
    "IDCARD": 0.90,
    "IP_ADDRESS": 0.90,
    "PASSWORD": 0.90,
    "ACCOUNTNUM": 0.90,
    "DRIVERLICENSENUM": 0.90,
    "TIME": 0.90,
    "COMPANY": 0.90,
}

FULL_AEGIS_CATEGORIES = frozenset(FULL_ENTITY_TO_CATEGORY.values())
ALL_AEGIS_ENTITY_TYPES = tuple(FULL_ENTITY_TO_CATEGORY)
SEMANTIC_AEGIS_ENTITY_TYPES = tuple(ENTITY_TO_CATEGORY)


# AEGIS는 주민등록번호와 외국인등록번호를 모두 IDCARD로 반환할 수 있다.
# 모델 결과의 원본 span은 건드리지 않고, 판정할 때만 아래 구분자를 제거한다.
_IDCARD_SEPARATORS = frozenset({
    " ", "\t", "-", ".", "/", "‐", "‑", "‒", "–", "—",
})

_IDCARD_CENTURY_BY_DIGIT = {
    "0": 1800,
    "9": 1800,
    "1": 1900,
    "2": 1900,
    "5": 1900,
    "6": 1900,
    "3": 2000,
    "4": 2000,
    "7": 2000,
    "8": 2000,
}


def _get_idcard_digits(text: str) -> str | None:
    """IDCARD 문자열에서 숫자만 꺼내되 허용하지 않은 문자는 거부한다."""
    digits = ""

    for char in text:
        if char.isdecimal():
            digits += str(int(char))
            continue
        if char in _IDCARD_SEPARATORS:
            continue
        return None

    if len(digits) != 13:
        return None
    return digits


def classify_aegis_idcard(text: str) -> str:
    """AEGIS IDCARD를 주민등록번호 또는 외국인등록번호로 세분화한다.

    하이픈·공백·점·슬래시는 판정용 문자열에서만 제거한다. 앞 6자리의
    생년월일과 7번째 구분 숫자가 모두 유효할 때만 세부 유형을 반환한다.
    판정하지 못한 값은 일반 신분증번호로 남겨 정책 단계에서 처리한다.
    """
    digits = _get_idcard_digits(text)
    if digits is None:
        return "id_card_number"

    century_gender = digits[6]
    century = _IDCARD_CENTURY_BY_DIGIT.get(century_gender)
    if century is None:
        return "id_card_number"

    try:
        birth_date = date(
            century + int(digits[0:2]),
            int(digits[2:4]),
            int(digits[4:6]),
        )
    except ValueError:
        return "id_card_number"

    if birth_date > date.today():
        return "id_card_number"
    if century_gender in "5678":
        return "foreigner_registration_number"
    return "resident_registration_number"


@dataclass(frozen=True)
class AegisTokenPrediction:
    """AEGIS가 한 token에 부여한 원시 entity 예측과 threshold 판정."""

    category: str
    bio: str
    entity_type: str
    start: int
    end: int
    text: str
    confidence: float
    threshold: float
    passed: bool


@dataclass(frozen=True)
class AegisPredictionTrace:
    """한 SearchView에서 나온 원시 AEGIS token 예측의 OCR 연결 정보."""

    line_ids: tuple[int, ...]
    input_text: str
    category: str
    bio: str
    entity_type: str
    text: str
    confidence: float
    threshold: float
    passed: bool
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class NerDetection:
    """한 SearchView에서 NER가 반환한 문자 span."""

    category: str
    entity_types: tuple[str, ...]
    start: int
    end: int
    text: str
    confidence: float

    @property
    def rule_id(self) -> str:
        entities = "+".join(self.entity_types)
        return f"ner:aegis-v2:{entities}"

    def to_dict(self) -> dict[str, object]:
        """문자열 전용 실행기가 그대로 JSON으로 저장할 수 있는 형식."""
        return {
            "category": self.category,
            "entity_types": list(self.entity_types),
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "confidence": self.confidence,
            "rule_id": self.rule_id,
        }


def _build_ner_detection(
    text: str,
    pieces: list[tuple[str, str, int, int, float]],
    entity_to_category: dict[str, str],
) -> NerDetection | None:
    """연속된 BIO token 조각을 한 개의 개인정보 문자 span으로 합친다."""
    if not pieces:
        return None
    entity_types = tuple(dict.fromkeys(piece[1] for piece in pieces))
    category = entity_to_category[pieces[0][1]]
    start = pieces[0][2]
    end = pieces[-1][3]
    span_text = text[start:end].strip()
    if not span_text:
        return None
    start += text[start:end].index(span_text)
    end = start + len(span_text)
    compact_length = len("".join(span_text.split()))
    if category == "person_name" and compact_length < 2:
        return None
    if category == "address" and (
        compact_length < 2
        or (
            set(entity_types) <= {"STREET", "BUILDINGNUM"}
            and compact_length < 3
        )
    ):
        return None
    confidence = float(sum(piece[4] for piece in pieces) / len(pieces))
    return NerDetection(
        category=category,
        entity_types=entity_types,
        start=start,
        end=end,
        text=span_text,
        confidence=confidence,
    )


def _selected_entity_categories(
    scope: str,
    enabled_entity_types: tuple[str, ...],
) -> dict[str, str]:
    """정책에서 활성화한 AEGIS entity와 내부 개인정보 유형을 연결한다."""
    if scope not in {"semantic", "all"}:
        raise ValueError("scope는 'semantic' 또는 'all'이어야 합니다.")
    entity_to_category = (
        ENTITY_TO_CATEGORY
        if scope == "semantic"
        else FULL_ENTITY_TO_CATEGORY
    )
    if not enabled_entity_types:
        return dict(entity_to_category)
    enabled = frozenset(enabled_entity_types)
    return {
        entity_type: category
        for entity_type, category in FULL_ENTITY_TO_CATEGORY.items()
        if entity_type in enabled
    }


_EXECUTION_PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
}


class AegisOnnxNer:
    """ONNX Runtime CPU 또는 CUDA로 실행하는 AEGIS v2 토큰 NER."""

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        *,
        execution_provider: str = "cpu",
        device_id: int = 0,
    ) -> None:
        self.model_dir = Path(model_dir)
        model_path = self.model_dir / "onnx" / "model_quantized.onnx"
        config_path = self.model_dir / "config.json"
        if not model_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(
                f"AEGIS NER 모델을 찾지 못했습니다: {self.model_dir}\n"
                f"Hugging Face {MODEL_NAME}의 {MODEL_REVISION} 파일이 필요합니다."
            )

        self.tokenizer = BertWordPieceTokenizer(
            str(self.model_dir / "vocab.txt"),
            lowercase=False,
        )
        self.tokenizer.enable_truncation(max_length=512)
        with config_path.open(encoding="utf-8") as file:
            config = json.load(file)
        self.id2label = {
            int(index): label for index, label in config["id2label"].items()
        }

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        provider_key = execution_provider.casefold()
        if provider_key not in _EXECUTION_PROVIDERS:
            supported = ", ".join(sorted(_EXECUTION_PROVIDERS))
            raise ValueError(
                f"execution_provider는 {supported} 중 하나여야 합니다."
            )
        requested_provider = _EXECUTION_PROVIDERS[provider_key]
        available_providers = tuple(ort.get_available_providers())
        if requested_provider not in available_providers:
            raise RuntimeError(
                f"{requested_provider}를 사용할 수 없습니다. "
                f"현재 ONNX Runtime provider: {available_providers}. "
                "CUDA 측정에는 NVIDIA GPU와 onnxruntime-gpu가 필요합니다."
            )
        providers: tuple[object, ...]
        if provider_key == "cuda":
            # ORT 1.21+ GPU 패키지는 PyTorch 또는 nvidia-* wheel에 포함된
            # CUDA·cuDNN DLL을 세션 생성 전에 미리 로드할 수 있다.
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            # CUDA에서 처리할 수 없는 INT8 연산은 CPU에서 이어서 실행하도록
            # CPU 실행 공급자를 두 번째 순서로 등록한다.
            providers = (
                ("CUDAExecutionProvider", {"device_id": device_id}),
                "CPUExecutionProvider",
            )
        else:
            providers = ("CPUExecutionProvider",)
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=providers,
        )
        self.execution_provider = provider_key
        self.requested_provider = requested_provider
        self.session_providers = tuple(self.session.get_providers())
        self.input_names = {item.name for item in self.session.get_inputs()}

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)

    @lru_cache(maxsize=2048)
    def predict_tokens(
        self,
        text: str,
        scope: str = "semantic",
        threshold_adjustment: float = 0.0,
        enabled_entity_types: tuple[str, ...] = (),
    ) -> tuple[AegisTokenPrediction, ...]:
        """threshold 적용 전 원시 token 예측과 통과 여부를 함께 반환한다."""
        if not text.strip():
            return ()
        entity_to_category = _selected_entity_categories(
            scope,
            enabled_entity_types,
        )

        encoded = self.tokenizer.encode(text)
        offsets = encoded.offsets
        encoded_arrays = {
            "input_ids": np.asarray([encoded.ids], dtype=np.int64),
            "attention_mask": np.asarray([encoded.attention_mask], dtype=np.int64),
            "token_type_ids": np.asarray([encoded.type_ids], dtype=np.int64),
        }
        inputs = {
            name: encoded_arrays[name]
            for name in self.input_names
            if name in encoded_arrays
        }
        logits = self.session.run(None, inputs)[0][0]
        probabilities = self._softmax(logits)
        predicted_ids = probabilities.argmax(axis=-1)
        confidences = probabilities.max(axis=-1)

        predictions: list[AegisTokenPrediction] = []
        for predicted_id, confidence, offset in zip(
            predicted_ids, confidences, offsets
        ):
            start, end = int(offset[0]), int(offset[1])
            if start == end:
                continue
            label = self.id2label[int(predicted_id)]
            if label == "O" or "-" not in label:
                continue
            bio, entity_type = label.split("-", 1)
            category = entity_to_category.get(entity_type)
            if category is None:
                continue
            threshold = min(
                1.0,
                max(
                    0.0,
                    ENTITY_THRESHOLDS[entity_type] + threshold_adjustment,
                ),
            )
            confidence_value = float(confidence)
            predictions.append(AegisTokenPrediction(
                category=category,
                bio=bio,
                entity_type=entity_type,
                start=start,
                end=end,
                text=text[start:end],
                confidence=confidence_value,
                threshold=threshold,
                passed=confidence_value >= threshold,
            ))
        return tuple(predictions)

    @lru_cache(maxsize=2048)
    def detect(
        self,
        text: str,
        scope: str = "semantic",
        threshold_adjustment: float = 0.0,
        enabled_entity_types: tuple[str, ...] = (),
    ) -> tuple[NerDetection, ...]:
        """threshold를 통과한 token 예측을 개인정보 문자 span으로 묶는다."""
        entity_to_category = _selected_entity_categories(
            scope,
            enabled_entity_types,
        )
        predictions = self.predict_tokens(
            text,
            scope,
            threshold_adjustment,
            enabled_entity_types,
        )
        pieces = [
            (
                prediction.bio,
                prediction.entity_type,
                prediction.start,
                prediction.end,
                prediction.confidence,
            )
            for prediction in predictions
            if prediction.passed
        ]

        detections: list[NerDetection] = []
        current: list[tuple[str, str, int, int, float]] = []

        for piece in pieces:
            bio, entity_type, start, _, _ = piece
            category = entity_to_category[entity_type]
            if not current:
                current.append(piece)
                continue
            previous_category = entity_to_category[current[-1][1]]
            # SURNAME+GIVENNAME, CITY+STREET처럼 모델의 세부 라벨이 달라도 우리
            # 최종 카테고리가 같고 사이가 공백 한 칸 이내면 하나의 span으로 묶는다.
            is_continuation = (
                category == previous_category
                and start <= current[-1][3] + 1
                and (bio == "I" or category in {"person_name", "address"})
            )
            if is_continuation:
                current.append(piece)
            else:
                detection = _build_ner_detection(text, current, entity_to_category)
                if detection is not None:
                    detections.append(detection)
                current = [piece]
        detection = _build_ner_detection(text, current, entity_to_category)
        if detection is not None:
            detections.append(detection)
        return tuple(detections)


@dataclass(frozen=True)
class AegisDetectionProvider:
    """AEGIS 실행 설정을 보관하며 SearchView 결과를 공통 형식으로 변환한다."""

    scope: str
    threshold_adjustment: float
    enabled_entity_types: tuple[str, ...]
    review_required: bool
    ner: AegisOnnxNer | None

    def __call__(self, view: SearchView) -> tuple[ProviderMatch, ...]:
        detections = iter_aegis_ner_detections(
            view,
            scope=self.scope,
            threshold_adjustment=self.threshold_adjustment,
            enabled_entity_types=self.enabled_entity_types,
            ner=self.ner,
        )
        return tuple(
            ProviderMatch(
                category=detection.category,
                start=detection.start,
                end=detection.end,
                text=detection.text,
                confidence=detection.confidence,
                rule_id=detection.rule_id,
                review_required=self.review_required,
            )
            for detection in detections
        )


@lru_cache(maxsize=1)
def get_aegis_ner() -> AegisOnnxNer:
    """Streamlit 재실행에도 ONNX session을 한 번만 로드한다."""
    return AegisOnnxNer()


def iter_aegis_ner_detections(
    view: SearchView,
    *,
    scope: str = "semantic",
    threshold_adjustment: float = 0.0,
    enabled_entity_types: tuple[str, ...] = (),
    ner: AegisOnnxNer | None = None,
):
    """일반 줄과 bbox로 검증한 표 필드 SearchView에 NER를 실행한다."""
    if view.mode not in {"token_spaced", "table_field"}:
        return
    runtime = ner or get_aegis_ner()
    for detection in runtime.detect(
        view.text,
        scope,
        threshold_adjustment,
        enabled_entity_types,
    ):
        # AEGIS의 IDCARD는 주민등록번호와 외국인등록번호를 구분하지 않는다.
        # ko-pii와 같은 내부 유형으로 병합할 수 있도록 정책 필터 전에 세분화한다.
        if "IDCARD" in detection.entity_types:
            category = classify_aegis_idcard(detection.text)
            if category != detection.category:
                detection = replace(detection, category=category)

        # 일반 token_spaced에는 별도 제한이 없다. 표 필드 보기에는 bbox로 확인한
        # 헤더 유형 하나만 허용해 옆 열의 값을 다른 유형으로 잘못 사용하지 않는다.
        if (
            view.allowed_categories is not None
            and detection.category not in view.allowed_categories
        ):
            continue
        yield detection


def trace_aegis_predictions(
    search_views: tuple[SearchView, ...],
    *,
    scope: str = "semantic",
    threshold_adjustment: float = 0.0,
    enabled_entity_types: tuple[str, ...] = (),
    ner: AegisOnnxNer | None = None,
) -> tuple[AegisPredictionTrace, ...]:
    """실제 AEGIS 입력별 원시 token 예측과 threshold 제외 결과를 반환한다."""
    runtime = ner or get_aegis_ner()
    traces: list[AegisPredictionTrace] = []
    for view in search_views:
        if view.mode not in {"token_spaced", "table_field"}:
            continue
        predictions = runtime.predict_tokens(
            view.text,
            scope,
            threshold_adjustment,
            enabled_entity_types,
        )
        for prediction in predictions:
            traces.append(AegisPredictionTrace(
                line_ids=tuple(line.line_id for line in view.lines),
                input_text=view.text,
                category=prediction.category,
                bio=prediction.bio,
                entity_type=prediction.entity_type,
                text=prediction.text,
                confidence=prediction.confidence,
                threshold=prediction.threshold,
                passed=prediction.passed,
                token_ids=view.token_ids_for_span(
                    prediction.start,
                    prediction.end,
                ),
            ))
    return tuple(traces)


def detect_aegis_spans(
    document: PreprocessedDocument,
    *,
    scope: str,
    threshold_adjustment: float = 0.0,
    enabled_entity_types: tuple[str, ...] = (),
    review_required: bool = True,
    search_settings: SearchViewSettings = DEFAULT_SEARCH_VIEW_SETTINGS,
    ner: AegisOnnxNer | None = None,
) -> tuple[PiiSpan, ...]:
    """선택 범위의 NER 결과를 공통 span→token→bbox 경로로 변환한다."""
    provider = AegisDetectionProvider(
        scope=scope,
        threshold_adjustment=threshold_adjustment,
        enabled_entity_types=enabled_entity_types,
        review_required=review_required,
        ner=ner,
    )
    return extend_address_spans(
        document,
        detect_with_provider(document, provider, search_settings=search_settings),
    )
