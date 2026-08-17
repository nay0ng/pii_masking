"""PoC 시연에 필요한 기능만 모은 Streamlit 화면.

이 화면은 관리자 정책을 실험하기 위한 기존 ``streamlit_app.py``와 목적이
다르다. 민감도·Custom 필드·탐지기 결합 설정은 숨기고 다음 흐름만 보여준다.

1. 저장된 OCR JSON을 열거나 이미지 한 장을 OCR 서버에 전송한다.
2. OCR 원문과 token을 확인하고 JSON/TXT를 내려받는다.
3. 선택한 개인정보 유형의 검출 위치와 검출 내역을 확인한다.
4. 원본 이미지와 자동 마스킹 결과를 비교하고 결과를 내려받는다.
"""

from __future__ import annotations

import csv
import hashlib
from io import BytesIO, StringIO
import json
import mimetypes
import os
from pathlib import Path
from time import perf_counter
import sys
import warnings

import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
KO_PII_SRC_ROOT = PROJECT_ROOT / "ko-pii" / "src"
for source_root in (PROJECT_ROOT, SRC_ROOT, KO_PII_SRC_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from pii_masking import (
    DetectorExecutionMode,
    DetectorMergeMode,
    GateAction,
    MaskingPolicyConfig,
    PatternStrictness,
    ReviewHandling,
    SearchViewSettings,
    combine_policy_spans,
    detect_all_pii_spans,
    load_ocr_results,
    preprocess_document,
)
from pii_masking.aegis_detector import (
    AegisOnnxNer,
    ALL_AEGIS_ENTITY_TYPES,
)
from pii_masking.detection_pipeline import detect_document_pii
from pii_masking.image_masker import render_masked_image
from apps.dashboard_support import (
    CATEGORY_NAMES,
    render_span_overlay,
    token_rows,
)


OCR_RESULTS_ROOT = PROJECT_ROOT / "sample_data"
IMAGE_ROOT = PROJECT_ROOT / "sample_data"
MODEL_ROOT = PROJECT_ROOT / "models"
POC_RESULT_ROOT = PROJECT_ROOT / "results" / "poc"
POC_IMAGE_ROOT = POC_RESULT_ROOT / "images"
POC_OCR_ROOT = POC_RESULT_ROOT / "ocr"
DEFAULT_OCR_URL = os.environ.get(
    "PII_OCR_URL",
    "http://127.0.0.1:19815/sample/fullpage",
)

POC_IDENTIFIER_CATEGORIES = (
    "resident_registration_number",
    "foreigner_registration_number",
    "passport_number",
    "driver_license_number",
)

COMMON_OPTIONAL_CATEGORIES = (
    "person_name",
    "address",
    "phone_number",
    "email_address",
    "account_card_number",
    "date",
)

CATEGORY_ORDER = tuple(dict.fromkeys(
    (*POC_IDENTIFIER_CATEGORIES, *COMMON_OPTIONAL_CATEGORIES, *CATEGORY_NAMES.keys())
))

DETECTOR_MODE_NAMES = {
    DetectorExecutionMode.ALWAYS_BOTH.value: "ko-pii·문서 규칙 + AEGIS",
    DetectorExecutionMode.RULE_FIRST_FALLBACK.value: (
        "ko-pii·문서 규칙 우선, 부족할 때 AEGIS"
    ),
    DetectorExecutionMode.RULE_ONLY.value: "ko-pii·문서 규칙만",
    DetectorExecutionMode.NER_ONLY.value: "AEGIS만",
}
DETECTOR_MODE_OPTIONS = tuple(DETECTOR_MODE_NAMES)


def _category_name(category: str) -> str:
    return CATEGORY_NAMES.get(category, category)


def _detector_mode_name(mode: str) -> str:
    return DETECTOR_MODE_NAMES[mode]


class _DocumentIndexFormatter:
    """문서 선택값은 번호로 유지하고 화면에는 파일명을 표시한다."""

    def __init__(self, labels: tuple[str, ...]) -> None:
        self.labels = labels

    def __call__(self, index: int) -> str:
        return self.labels[index]


def _image_to_png_bytes(image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _path_sort_key(path: Path) -> str:
    return str(path).casefold()


def _discover_ocr_result_files() -> tuple[Path, ...]:
    candidates: set[Path] = set()
    for root in (OCR_RESULTS_ROOT, POC_OCR_ROOT):
        if not root.exists():
            continue
        candidates.update(root.glob("*_results.json"))
        candidates.update(root.glob("*_result.json"))
    return tuple(sorted(candidates, key=_path_sort_key))


def _display_result_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _save_uploaded_json(uploaded_file) -> Path:
    POC_OCR_ROOT.mkdir(parents=True, exist_ok=True)
    data = uploaded_file.getvalue()
    digest = hashlib.sha256(data).hexdigest()[:8]
    name = Path(uploaded_file.name).name
    stem = Path(name).stem
    output_path = POC_OCR_ROOT / f"{stem}_{digest}_results.json"
    output_path.write_bytes(data)
    return output_path


def _request_ocr(uploaded_file, ocr_url: str) -> tuple[Path, float]:
    """업로드 이미지 한 장을 기존 전문인식 OCR API에 전송하고 결과를 저장한다."""
    image_bytes = uploaded_file.getvalue()
    if not image_bytes:
        raise ValueError("업로드한 이미지가 비어 있습니다.")

    safe_name = Path(uploaded_file.name).name
    suffix = Path(safe_name).suffix.lower() or ".png"
    stem = Path(safe_name).stem
    digest = hashlib.sha256(image_bytes).hexdigest()[:8]

    POC_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    POC_OCR_ROOT.mkdir(parents=True, exist_ok=True)
    image_path = POC_IMAGE_ROOT / f"{stem}_{digest}{suffix}"
    image_path.write_bytes(image_bytes)

    mime_type, _ = mimetypes.guess_type(safe_name)
    if mime_type is None:
        mime_type = "application/octet-stream"

    files = {"srcFile": (safe_name, image_bytes, mime_type)}
    form_data = {
        "base64Image": "false",
        "fullText": "true",
        "saveOption": "false",
    }
    started_at = perf_counter()
    response = requests.post(
        ocr_url.strip(),
        data=form_data,
        files=files,
        timeout=120,
    )
    elapsed = perf_counter() - started_at
    response.raise_for_status()

    result = response.json()
    if not isinstance(result, dict):
        raise ValueError("OCR 서버 응답이 JSON 객체가 아닙니다.")
    if result.get("resultCode") != "0000":
        raise ValueError(
            "OCR 처리에 실패했습니다. "
            f"resultCode={result.get('resultCode')!r}"
        )

    result_path = POC_OCR_ROOT / f"{stem}_{digest}_ocr_results.json"
    batch_result = {str(image_path.resolve()): result}
    result_path.write_text(
        json.dumps(batch_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result_path, elapsed


@st.cache_data(show_spinner="OCR JSON을 읽고 문서 좌표를 구성하는 중입니다...")
def _load_documents(results_path: str, modified_ns: int):
    del modified_ns
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        documents = load_ocr_results(
            results_path,
            skip_failed_responses=True,
            image_root=IMAGE_ROOT,
        )
    items = tuple(
        (document, preprocess_document(document))
        for document in documents
    )
    warning_messages = tuple(str(item.message) for item in caught)
    return items, warning_messages


def _fixed_poc_policy(
    selected_categories: tuple[str, ...],
    checksum_validation_enabled: bool,
    execution_mode: str,
) -> MaskingPolicyConfig:
    """화면에서 숨긴 PoC 고정 정책을 한 곳에서 정의한다."""
    return MaskingPolicyConfig(
        execution_mode=DetectorExecutionMode(execution_mode),
        merge_mode=DetectorMergeMode.ANY,
        selected_categories=frozenset(selected_categories),
        pattern_strictness=PatternStrictness.NORMALIZED,
        aegis_scope="all",
        aegis_entity_types=ALL_AEGIS_ENTITY_TYPES,
        aegis_threshold_adjustment=0.0,
        minimum_confidence=0.0,
        minimum_risk_level=1,
        # OFF이면 번들 안의 ko-pii 원본을 수정하지 않고 프로젝트 어댑터의
        # 완화 탐지를 사용한다. 합성 번호나 체크섬 오류 번호도 형식이 맞으면
        # 후보로 남으므로, ON/OFF 결과를 같은 화면에서 비교할 수 있다.
        checksum_validation_enabled=checksum_validation_enabled,
        checksum_invalid_action=(
            GateAction.EXCLUDE
            if checksum_validation_enabled
            else GateAction.MASK
        ),
        minimum_context_evidence_count=0,
        missing_context_action=GateAction.REVIEW,
        review_handling=ReviewHandling.MASK_ALL,
        max_adjacent_lines=3,
        custom_person_field_labels=(),
        custom_address_field_labels=(),
        mask_padding=2,
    )


@st.cache_resource(show_spinner="AEGIS 모델을 불러오는 중입니다...")
def _load_aegis_runtime(model_dir: str) -> AegisOnnxNer:
    """화면이 다시 실행되어도 같은 ONNX Runtime session을 재사용한다."""
    return AegisOnnxNer(model_dir)


@st.cache_data(show_spinner="개인정보를 탐지하는 중입니다...")
def _detect_document(
    results_path: str,
    modified_ns: int,
    document_index: int,
    selected_categories: tuple[str, ...],
    checksum_validation_enabled: bool,
    execution_mode: str,
):
    items, _ = _load_documents(results_path, modified_ns)
    _, processed = items[document_index]
    policy_config = _fixed_poc_policy(
        selected_categories,
        checksum_validation_enabled,
        execution_mode,
    )
    search_settings = SearchViewSettings(
        max_adjacent_lines=policy_config.max_adjacent_lines,
        pattern_strictness=policy_config.pattern_strictness,
        checksum_validation_enabled=policy_config.checksum_validation_enabled,
        custom_person_field_labels=(),
        custom_address_field_labels=(),
    )

    started_at = perf_counter()
    ner_error = ""
    try:
        ner_runtime = None
        if policy_config.execution_mode != DetectorExecutionMode.RULE_ONLY:
            ner_runtime = _load_aegis_runtime(str(MODEL_ROOT.resolve()))
        policy_result = detect_document_pii(
            processed,
            policy_config,
            search_settings,
            ner=ner_runtime,
        )
        rule_spans = policy_result.rule_spans
        ner_spans = policy_result.ner_spans
        final_spans = policy_result.final_spans
        rule_seconds = policy_result.rule_seconds
        ner_seconds = policy_result.ner_seconds
    except Exception as error:
        # 모델 파일이나 ONNX Runtime 문제로 AEGIS가 실행되지 않아도 ko-pii와
        # 프로젝트 규칙의 PoC 결과는 계속 확인할 수 있게 한다. 단, AEGIS만
        # 선택한 경우에는 비교 의미가 달라지지 않도록 규칙 결과를 대신 넣지 않는다.
        rule_spans = ()
        rule_seconds = 0.0
        if policy_config.execution_mode != DetectorExecutionMode.NER_ONLY:
            rule_started_at = perf_counter()
            rule_spans = detect_all_pii_spans(
                processed,
                search_settings=search_settings,
            )
            rule_seconds = perf_counter() - rule_started_at
        ner_spans = ()
        ner_seconds = 0.0
        ner_error = str(error)
        final_spans = combine_policy_spans(
            rule_spans,
            ner_spans,
            policy_config.to_detection_policy(),
        )
    total_seconds = perf_counter() - started_at
    selected = set(selected_categories)
    selected_rule_spans = tuple(
        span for span in rule_spans if span.entity_type in selected
    )
    selected_ner_spans = tuple(
        span for span in ner_spans if span.entity_type in selected
    )
    return (
        selected_rule_spans,
        selected_ner_spans,
        final_spans,
        rule_seconds,
        ner_seconds,
        total_seconds,
        ner_error,
    )


def _span_source(rule_id: str) -> str:
    if rule_id.startswith("ko_pii:"):
        return "ko-pii"
    if "aegis" in rule_id.casefold():
        return "AEGIS"
    if rule_id.startswith("context:") or rule_id.startswith("table:"):
        return "문서 구조 규칙"
    return rule_id.split(":", 1)[0]


def _span_rows(spans) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, span in enumerate(spans, start=1):
        boxes = "; ".join(
            f"({box.x1},{box.y1})-({box.x2},{box.y2})"
            for box in span.boxes
        )
        rows.append({
            "번호": index,
            "개인정보 유형": _category_name(span.entity_type),
            "검출 문자열": span.text.replace("\n", " ↵ "),
            "탐지기": _span_source(span.rule_id),
            "OCR 신뢰도": f"{span.ocr_confidence:.3f}",
            "탐지 신뢰도": f"{span.detector_confidence:.3f}",
            "검수 필요": "예" if span.review_required else "아니오",
            "token ID": ", ".join(str(value) for value in span.token_ids),
            "bbox": boxes,
            "규칙": span.rule_id,
        })
    return rows


def _span_json_bytes(image_path: Path, spans) -> bytes:
    detections = []
    for span in spans:
        detections.append({
            "category": span.entity_type,
            "category_name": _category_name(span.entity_type),
            "text": span.text,
            "source": _span_source(span.rule_id),
            "rule": span.rule_id,
            "ocr_confidence": span.ocr_confidence,
            "detector_confidence": span.detector_confidence,
            "review_required": span.review_required,
            "line_ids": list(span.line_ids),
            "token_ids": list(span.token_ids),
            "boxes": [
                {"x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2}
                for box in span.boxes
            ],
        })
    payload = {
        "image": str(image_path),
        "detection_count": len(detections),
        "detections": detections,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _span_csv_bytes(rows: list[dict[str, object]]) -> bytes:
    if not rows:
        return b""
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def _reset_document_index() -> None:
    st.session_state["poc_document_index"] = 0


def _move_document(delta: int, document_count: int) -> None:
    current = int(st.session_state.get("poc_document_index", 0))
    next_index = max(0, min(document_count - 1, current + delta))
    st.session_state["poc_document_index"] = next_index


def _render_sidebar() -> tuple[Path | None, tuple[str, ...]]:
    with st.sidebar:
        st.header("PoC 입력")
        input_mode = st.radio(
            "입력 방식",
            ("저장된 OCR 결과", "이미지 OCR 요청"),
        )

        results_path: Path | None = None
        if input_mode == "저장된 OCR 결과":
            result_files = _discover_ocr_result_files()
            if result_files:
                active_path = st.session_state.get("poc_active_result_path")
                default_index = 0
                if active_path:
                    for index, path in enumerate(result_files):
                        if str(path) == active_path:
                            default_index = index
                            break
                results_path = st.selectbox(
                    "OCR 결과 JSON",
                    result_files,
                    index=default_index,
                    format_func=_display_result_path,
                    on_change=_reset_document_index,
                )
                st.session_state["poc_active_result_path"] = str(results_path)
            else:
                st.warning("사용할 수 있는 OCR 결과 JSON이 없습니다.")

            uploaded_json = st.file_uploader(
                "OCR 결과 JSON 업로드",
                type=("json",),
            )
            if uploaded_json is not None and st.button("업로드 JSON 사용"):
                try:
                    results_path = _save_uploaded_json(uploaded_json)
                    st.session_state["poc_active_result_path"] = str(results_path)
                    _reset_document_index()
                    st.rerun()
                except Exception as error:
                    st.error(f"JSON 저장에 실패했습니다: {error}")
        else:
            uploaded_image = st.file_uploader(
                "전문인식할 이미지",
                type=("jpg", "jpeg", "png", "bmp", "tif", "tiff"),
            )
            ocr_url = st.text_input("OCR 서버 URL", value=DEFAULT_OCR_URL)
            if st.button("전문인식 실행", type="primary", width="stretch"):
                if uploaded_image is None:
                    st.error("먼저 이미지를 선택해 주세요.")
                else:
                    try:
                        with st.spinner("OCR 서버에서 전문인식 중입니다..."):
                            results_path, elapsed = _request_ocr(uploaded_image, ocr_url)
                        st.session_state["poc_active_result_path"] = str(results_path)
                        st.session_state["poc_last_ocr_seconds"] = elapsed
                        _reset_document_index()
                        st.rerun()
                    except Exception as error:
                        st.error(f"전문인식에 실패했습니다: {error}")
            active_path = st.session_state.get("poc_active_result_path")
            if active_path and Path(active_path).exists():
                results_path = Path(active_path)

        st.divider()
        st.header("탐지기")
        execution_mode = st.radio(
            "실행할 탐지기",
            options=DETECTOR_MODE_OPTIONS,
            index=0,
            format_func=_detector_mode_name,
        )
        st.caption(
            "개인정보 유형 목록은 두 탐지기가 공유하는 내부 분류입니다. "
            "선택한 탐지기가 지원하지 않는 유형은 결과가 나오지 않습니다."
        )
        if execution_mode == DetectorExecutionMode.RULE_FIRST_FALLBACK.value:
            st.caption(
                "규칙 결과를 먼저 확인하고, 선택한 AEGIS 지원 유형이 문서에서 "
                "하나도 발견되지 않았을 때 AEGIS를 추가 실행합니다. 특정 값 하나의 "
                "누락 여부는 AEGIS를 실행하기 전에는 알 수 없으므로 문서·유형 "
                "단위로 판단합니다."
            )

        st.divider()
        st.header("마스킹할 개인정보")
        button_left, button_right = st.columns(2)
        if button_left.button("PoC 4종", width="stretch"):
            st.session_state["poc_categories"] = list(POC_IDENTIFIER_CATEGORIES)
        if button_right.button("전체 선택", width="stretch"):
            st.session_state["poc_categories"] = list(CATEGORY_ORDER)

        if "poc_categories" not in st.session_state:
            st.session_state["poc_categories"] = list(POC_IDENTIFIER_CATEGORIES)
        selected_categories = st.multiselect(
            "개인정보 유형",
            options=CATEGORY_ORDER,
            format_func=_category_name,
            key="poc_categories",
        )
        st.caption(
            "기본값은 주민등록번호·외국인등록번호·여권번호·운전면허번호입니다. "
            "전화번호와 이메일 같은 부가 개인정보도 여기서 선택할 수 있습니다."
        )

        st.divider()
        st.header("번호 검증")
        checksum_validation_enabled = st.toggle(
            "체크섬 검사",
            value=True,
            help=(
                "ON이면 번호 내부 검증을 사용합니다. OFF이면 주민등록번호처럼 "
                "정해진 형태가 확인된 번호를 체크섬 오류와 관계없이 마스킹 "
                "후보로 유지합니다."
            ),
        )
        if checksum_validation_enabled:
            st.caption("ON · 체크섬 불일치 번호는 마스킹 대상에서 제외합니다.")
        else:
            st.caption(
                "OFF · 합성 번호나 체크섬 오류 번호도 형식이 맞으면 마스킹합니다."
            )
        st.caption(
            "적용 유형: 주민등록번호·외국인등록번호·사업자등록번호·"
            "법인등록번호·카드번호. 여권번호와 운전면허번호 등 체크섬이 "
            "없는 유형에는 영향을 주지 않습니다."
        )

        st.divider()
        st.caption(
            "고정 탐지 정책: ko-pii와 AEGIS 결과 통합, 일반 OCR 공백·하이픈 "
            "정리, 검수 후보 우선 마스킹"
        )
    return (
        results_path,
        tuple(selected_categories),
        checksum_validation_enabled,
        execution_mode,
    )


def main() -> None:
    st.set_page_config(
        page_title="개인정보 마스킹 PoC",
        page_icon="🔒",
        layout="wide",
    )
    st.title("개인정보 탐지·마스킹 PoC")
    st.caption(
        "전문인식 OCR 결과에서 개인정보를 검출하고 원본 이미지 좌표에 마스킹합니다."
    )

    (
        results_path,
        selected_categories,
        checksum_validation_enabled,
        execution_mode,
    ) = _render_sidebar()
    if results_path is None:
        st.info("왼쪽에서 OCR 결과 JSON 또는 이미지를 선택해 주세요.")
        return
    if not selected_categories:
        st.warning("마스킹할 개인정보 유형을 한 개 이상 선택해 주세요.")
        return

    try:
        modified_ns = results_path.stat().st_mtime_ns
        items, load_warnings = _load_documents(str(results_path), modified_ns)
    except Exception as error:
        st.error(f"OCR 결과를 읽지 못했습니다: {error}")
        return

    previous_path = st.session_state.get("poc_loaded_result_path")
    if previous_path != str(results_path):
        st.session_state["poc_loaded_result_path"] = str(results_path)
        _reset_document_index()

    document_count = len(items)
    current_index = int(st.session_state.get("poc_document_index", 0))
    if current_index >= document_count:
        current_index = 0
        st.session_state["poc_document_index"] = 0

    with st.sidebar:
        document_labels = [
            document.image_path.name
            for document, _ in items
        ]
        selected_document_index = st.selectbox(
            "문서",
            options=range(document_count),
            index=current_index,
            format_func=_DocumentIndexFormatter(tuple(document_labels)),
            key="poc_document_index",
        )
        st.caption(f"{selected_document_index + 1} / {document_count}")

    ocr_document, processed = items[selected_document_index]
    try:
        (
            rule_spans,
            ner_spans,
            final_spans,
            rule_seconds,
            ner_seconds,
            total_seconds,
            ner_error,
        ) = _detect_document(
            str(results_path),
            modified_ns,
            selected_document_index,
            tuple(sorted(selected_categories)),
            checksum_validation_enabled,
            execution_mode,
        )
    except Exception as error:
        st.error(f"개인정보 탐지에 실패했습니다: {error}")
        return

    try:
        masked_image, span_count, box_count = render_masked_image(
            processed,
            final_spans,
            padding=2,
        )
    except Exception as error:
        st.error(f"마스킹 이미지를 만들지 못했습니다: {error}")
        return

    source_path = Path(ocr_document.image_path)
    st.subheader(source_path.name)
    if load_warnings:
        with st.expander(f"OCR 결과 경고 {len(load_warnings)}건"):
            for message in load_warnings:
                st.warning(message)
    if ner_error:
        st.warning(
            "AEGIS 실행에 실패하여 ko-pii·문서 구조 규칙 결과만 사용했습니다. "
            f"오류: {ner_error}"
        )

    metrics = st.columns(6)
    metrics[0].metric("OCR token", len(processed.index.tokens))
    metrics[1].metric("검출 개인정보", span_count)
    metrics[2].metric("마스킹 영역", box_count)
    metrics[3].metric("ko-pii·규칙", len(rule_spans))
    metrics[4].metric("AEGIS", len(ner_spans))
    metrics[5].metric("PII 처리시간", f"{total_seconds * 1000:.0f} ms")

    ocr_seconds = st.session_state.get("poc_last_ocr_seconds")
    if ocr_seconds is not None and str(results_path) == st.session_state.get("poc_active_result_path"):
        st.caption(
            f"최근 전문인식 OCR 요청 시간: {ocr_seconds:.3f}초 · "
            f"ko-pii·규칙 {rule_seconds * 1000:.0f}ms · "
            f"AEGIS {ner_seconds * 1000:.0f}ms"
        )

    ocr_tab, detection_tab, masking_tab = st.tabs(
        ("1. OCR 결과", "2. 개인정보 검출", "3. 마스킹 결과")
    )

    with ocr_tab:
        image_column, text_column = st.columns((1, 1))
        image_column.image(str(source_path), caption="원본 이미지", width="stretch")
        text_column.text_area(
            "OCR full_text",
            value=ocr_document.full_text,
            height=520,
            disabled=True,
        )

        download_left, download_right = st.columns(2)
        download_left.download_button(
            "OCR 결과 JSON 다운로드",
            data=results_path.read_bytes(),
            file_name=results_path.name,
            mime="application/json",
            width="stretch",
        )
        download_right.download_button(
            "현재 문서 TXT 다운로드",
            data=ocr_document.full_text.encode("utf-8"),
            file_name=f"{source_path.stem}_ocr.txt",
            mime="text/plain",
            width="stretch",
        )

        with st.expander("OCR token·좌표 확인"):
            st.dataframe(
                token_rows(processed),
                width="stretch",
                hide_index=True,
            )

    with detection_tab:
        rows = _span_rows(final_spans)
        if final_spans:
            overlay = render_span_overlay(processed, final_spans)
            st.image(
                overlay,
                caption="개인정보 검출 위치",
                width="stretch",
            )
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.info("선택한 유형에서 검출된 개인정보가 없습니다.")

        detection_left, detection_right = st.columns(2)
        detection_left.download_button(
            "개인정보 검출 내역 JSON 다운로드",
            data=_span_json_bytes(source_path, final_spans),
            file_name=f"{source_path.stem}_pii.json",
            mime="application/json",
            width="stretch",
        )
        detection_right.download_button(
            "개인정보 검출 내역 CSV 다운로드",
            data=_span_csv_bytes(rows),
            file_name=f"{source_path.stem}_pii.csv",
            mime="text/csv",
            width="stretch",
            disabled=not rows,
        )

    with masking_tab:
        original_column, masked_column = st.columns(2)
        original_column.image(
            str(source_path),
            caption="원본",
            width="stretch",
        )
        masked_column.image(
            masked_image,
            caption=f"자동 마스킹 · 개인정보 {span_count}건 / 영역 {box_count}개",
            width="stretch",
        )
        st.download_button(
            "마스킹 이미지 다운로드",
            data=_image_to_png_bytes(masked_image),
            file_name=f"{source_path.stem}_masked.png",
            mime="image/png",
            type="primary",
            width="stretch",
        )

    st.divider()
    navigation_left, navigation_center, navigation_right = st.columns((1, 2, 1))
    navigation_left.button(
        "← 이전 문서",
        disabled=selected_document_index == 0,
        width="stretch",
        on_click=_move_document,
        args=(-1, document_count),
    )
    navigation_center.markdown(
        f"<p style='text-align:center'>{selected_document_index + 1} / "
        f"{document_count}</p>",
        unsafe_allow_html=True,
    )
    navigation_right.button(
        "다음 문서 →",
        disabled=selected_document_index == document_count - 1,
        width="stretch",
        on_click=_move_document,
        args=(1, document_count),
    )


if __name__ == "__main__":
    main()
