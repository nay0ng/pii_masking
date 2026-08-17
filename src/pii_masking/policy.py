"""관리자 화면과 개인정보 탐지 코드가 공유하는 설정 파일.

Streamlit 화면의 위젯 값을 엔진 코드에 직접 연결하지 않는다. 대신 이 모듈의
JSON 형식을 중간 계약으로 사용한다. 현재는 Python 엔진이 읽고, 이후에는
Java 백엔드와 C++ 공유 라이브러리도 같은 형식을 사용할 수 있다.

코드 검토 순서
1. ``SensitivityPreset``에서 민감도 단계가 실제 설정값을 어떻게 바꾸는지 확인한다.
2. ``MaskingPolicyConfig``에서 policy.json이 저장할 전체 옵션을 확인한다.
3. ``from_dict()``와 ``to_dict()``에서 JSON과 Python 객체의 변환을 확인한다.
4. ``to_detection_policy()``에서 화면용 설정이 실제 Gate 판단 설정으로 바뀌는
   과정을 확인한다.
5. ``load_masking_policy()``가 파일을 읽고 위 객체를 반환한다.

이 파일은 탐지나 마스킹을 실행하지 않는다. 실행할 때 사용할 설정의 형식과
기본값을 정의하고 ``detection_policy.py``에 전달할 값으로 변환한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import json
from pathlib import Path
from typing import Any

from .aegis_detector import (
    ALL_AEGIS_ENTITY_TYPES,
    FULL_AEGIS_CATEGORIES,
    FULL_ENTITY_TO_CATEGORY,
    SEMANTIC_AEGIS_ENTITY_TYPES,
)
from .categories import PII_CATEGORIES
from .detection_policy import (
    DetectionPolicy,
    DetectorExecutionMode,
    DetectorMergeMode,
    GateAction,
    ReviewHandling,
)
from .paths import CONFIG_DIR
from .search_views import PatternStrictness


POLICY_SCHEMA_VERSION = 1
DEFAULT_POLICY_PATH = CONFIG_DIR / "policy.json"
DEFAULT_SELECTED_CATEGORIES = frozenset(
    category.label for category in PII_CATEGORIES
)
VALID_CATEGORY_LABELS = DEFAULT_SELECTED_CATEGORIES | FULL_AEGIS_CATEGORIES


class MaskingSensitivity(IntEnum):
    """관리자가 선택하는 전체 마스킹 민감도."""

    PRECISE = 1
    STANDARD = 2
    SENSITIVE = 3
    MAXIMUM = 4


@dataclass(frozen=True)
class SensitivityPreset:
    """민감도 한 단계가 실제 엔진 설정으로 해석된 결과."""

    pattern_strictness: PatternStrictness
    checksum_validation_enabled: bool
    minimum_risk_level: int
    checksum_invalid_action: GateAction
    minimum_context_evidence_count: int
    missing_context_action: GateAction
    aegis_threshold_adjustment: float
    review_handling: ReviewHandling


SENSITIVITY_PRESETS = {
    MaskingSensitivity.PRECISE: SensitivityPreset(
        pattern_strictness=PatternStrictness.EXACT,
        checksum_validation_enabled=True,
        minimum_risk_level=3,
        checksum_invalid_action=GateAction.EXCLUDE,
        minimum_context_evidence_count=1,
        missing_context_action=GateAction.EXCLUDE,
        aegis_threshold_adjustment=0.05,
        review_handling=ReviewHandling.EXCLUDE_REVIEW,
    ),
    MaskingSensitivity.STANDARD: SensitivityPreset(
        pattern_strictness=PatternStrictness.NORMALIZED,
        checksum_validation_enabled=True,
        minimum_risk_level=2,
        checksum_invalid_action=GateAction.REVIEW,
        minimum_context_evidence_count=1,
        missing_context_action=GateAction.REVIEW,
        aegis_threshold_adjustment=0.0,
        review_handling=ReviewHandling.MASK_ALL,
    ),
    MaskingSensitivity.SENSITIVE: SensitivityPreset(
        pattern_strictness=PatternStrictness.RECOVERED,
        checksum_validation_enabled=True,
        minimum_risk_level=1,
        checksum_invalid_action=GateAction.MASK,
        minimum_context_evidence_count=0,
        missing_context_action=GateAction.MASK,
        aegis_threshold_adjustment=-0.05,
        review_handling=ReviewHandling.MASK_ALL,
    ),
    MaskingSensitivity.MAXIMUM: SensitivityPreset(
        pattern_strictness=PatternStrictness.OCR_TOLERANT,
        checksum_validation_enabled=False,
        minimum_risk_level=1,
        checksum_invalid_action=GateAction.MASK,
        minimum_context_evidence_count=0,
        missing_context_action=GateAction.MASK,
        aegis_threshold_adjustment=-0.15,
        review_handling=ReviewHandling.MASK_ALL,
    ),
}


def get_sensitivity_preset(
    sensitivity: MaskingSensitivity | int,
) -> SensitivityPreset:
    return SENSITIVITY_PRESETS[MaskingSensitivity(sensitivity)]


@dataclass
class MaskingPolicyConfig:
    """OCR 이후 탐지·결합·마스킹 과정에서 사용하는 관리자 설정.

    설정은 다음 네 묶음으로 나뉜다.
    1. 탐지기 실행 및 결과 병합 방법
    2. 규칙 탐지와 AEGIS의 판정 기준
    3. 문서 구조 탐색 범위와 관리자 추가 필드명
    4. 이미지에 마스크를 그릴 때 사용할 여백
    """

    # 어떤 탐지기를 실행하고 두 탐지 결과를 어떻게 합칠지 결정한다.
    execution_mode: DetectorExecutionMode = DetectorExecutionMode.ALWAYS_BOTH
    merge_mode: DetectorMergeMode = DetectorMergeMode.ANY
    selected_categories: frozenset[str] = DEFAULT_SELECTED_CATEGORIES

    # 민감도 프리셋을 적용한 뒤 실제 탐지기가 사용하는 세부 설정이다.
    sensitivity_level: MaskingSensitivity = MaskingSensitivity.STANDARD
    pattern_strictness: PatternStrictness = PatternStrictness.NORMALIZED

    # AEGIS의 실행 범위, 사용할 모델 라벨, 라벨별 기준값 조정치다.
    aegis_scope: str = "semantic"
    aegis_entity_types: tuple[str, ...] = SEMANTIC_AEGIS_ENTITY_TYPES
    aegis_threshold_adjustment: float = 0.0

    # 모든 탐지 결과에 공통으로 적용하는 최소 신뢰도와 위험도다.
    minimum_confidence: float = 0.0
    minimum_risk_level: int = 1

    # 체크섬이 틀리거나 문맥 근거가 부족한 후보의 처리 방법이다.
    checksum_validation_enabled: bool = True
    checksum_invalid_action: GateAction = GateAction.MASK
    minimum_context_evidence_count: int = 0
    missing_context_action: GateAction = GateAction.REVIEW
    anchor_required_categories: frozenset[str] = frozenset()
    missing_anchor_action: GateAction = GateAction.EXCLUDE

    # 검수가 필요한 후보를 마스킹, 제외, 검수 전용 중 어떻게 처리할지 정한다.
    review_handling: ReviewHandling = ReviewHandling.MASK_ALL

    # bbox 바깥으로 추가할 픽셀 여백과 인접 OCR 줄 탐색 범위다.
    mask_padding: int = 2
    max_adjacent_lines: int = 3

    # 기본 필드명 외에 관리자가 실행 중 추가한 이름·주소 필드명이다.
    custom_person_field_labels: tuple[str, ...] = ()
    custom_address_field_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """생성된 설정의 자료형과 허용 범위를 한 곳에서 검증한다."""
        # JSON이나 UI에서 숫자와 문자열로 들어온 값을 enum으로 통일한다.
        if not isinstance(self.sensitivity_level, MaskingSensitivity):
            self.sensitivity_level = MaskingSensitivity(self.sensitivity_level)
        if not isinstance(self.pattern_strictness, PatternStrictness):
            self.pattern_strictness = PatternStrictness(self.pattern_strictness)
        self.selected_categories = frozenset(self.selected_categories)
        self.anchor_required_categories = frozenset(self.anchor_required_categories)

        # AEGIS 관련 설정을 먼저 검증한다.
        if self.aegis_scope not in {"semantic", "all"}:
            raise ValueError("aegis_scope은 'semantic' 또는 'all'이어야 합니다.")
        known_aegis_entities = set(ALL_AEGIS_ENTITY_TYPES)
        unknown_entities = set(self.aegis_entity_types) - known_aegis_entities
        if unknown_entities:
            raise ValueError(
                "지원하지 않는 AEGIS 라벨입니다: "
                + ", ".join(sorted(unknown_entities))
            )
        if not -0.5 <= self.aegis_threshold_adjustment <= 0.5:
            raise ValueError(
                "aegis_threshold_adjustment는 -0.5~0.5 사이여야 합니다."
            )

        # 후보 필터와 문서 구조 탐색에 사용하는 숫자 범위를 검증한다.
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence는 0~1 사이여야 합니다.")
        if self.minimum_risk_level not in {1, 2, 3, 4, 5}:
            raise ValueError("minimum_risk_level은 1~5 사이여야 합니다.")
        if self.minimum_context_evidence_count not in {0, 1, 2, 3}:
            raise ValueError(
                "minimum_context_evidence_count는 0~3 사이여야 합니다."
            )
        if not 0 <= self.mask_padding <= 50:
            raise ValueError("mask_padding은 0~50 사이여야 합니다.")
        if self.max_adjacent_lines not in {1, 2, 3}:
            raise ValueError("max_adjacent_lines는 1, 2, 3 중 하나여야 합니다.")

        # 선택한 개인정보 유형이 현재 엔진에 등록되어 있는지 확인한다.
        unknown = self.selected_categories - VALID_CATEGORY_LABELS
        if unknown:
            raise ValueError(
                "알 수 없는 개인정보 유형이 있습니다: "
                + ", ".join(sorted(unknown))
            )
        unknown_anchors = (
            self.anchor_required_categories - VALID_CATEGORY_LABELS
        )
        if unknown_anchors:
            raise ValueError(
                "알 수 없는 anchor 필수 유형이 있습니다: "
                + ", ".join(sorted(unknown_anchors))
            )

    @property
    def ner_categories(self) -> frozenset[str]:
        """현재 AEGIS 범위에 해당하는 내부 개인정보 유형."""
        categories: set[str] = set()
        for entity_type in self.aegis_entity_types:
            if entity_type == "IDCARD":
                # AEGIS IDCARD는 탐지 후 번호 형식에 따라 두 유형으로 재분류한다.
                # 이 연결이 있어야 주민·외국인번호만 선택한 정책에서도 NER가 실행된다.
                categories.add("resident_registration_number")
                categories.add("foreigner_registration_number")
                continue
            category = FULL_ENTITY_TO_CATEGORY[entity_type]
            categories.add(category)
        return frozenset(categories)

    def to_detection_policy(self) -> DetectionPolicy:
        """실제 탐지 파이프라인이 사용하는 정책 객체로 변환한다."""
        return DetectionPolicy(
            execution_mode=self.execution_mode,
            merge_mode=self.merge_mode,
            selected_categories=self.selected_categories,
            ner_categories=self.ner_categories,
            minimum_confidence=self.minimum_confidence,
            minimum_risk_level=self.minimum_risk_level,
            checksum_validation_enabled=self.checksum_validation_enabled,
            checksum_invalid_action=self.checksum_invalid_action,
            minimum_context_evidence_count=self.minimum_context_evidence_count,
            missing_context_action=self.missing_context_action,
            anchor_required_categories=self.anchor_required_categories,
            missing_anchor_action=self.missing_anchor_action,
            review_handling=self.review_handling,
        )

    def to_dict(self) -> dict[str, Any]:
        """Java·C++에서도 읽을 수 있는 JSON 호환 사전으로 변환한다."""
        detection = {
            "execution_mode": self.execution_mode.value,
            "merge_mode": self.merge_mode.value,
            "selected_categories": sorted(self.selected_categories),
            "sensitivity_level": int(self.sensitivity_level),
            "pattern_strictness": self.pattern_strictness.value,
            "aegis_scope": self.aegis_scope,
            "aegis_entity_types": list(self.aegis_entity_types),
            "aegis_threshold_adjustment": self.aegis_threshold_adjustment,
            "minimum_confidence": self.minimum_confidence,
            "minimum_risk_level": self.minimum_risk_level,
            "checksum_validation_enabled": self.checksum_validation_enabled,
            "checksum_invalid_action": self.checksum_invalid_action.value,
            "minimum_context_evidence_count": self.minimum_context_evidence_count,
            "missing_context_action": self.missing_context_action.value,
            "anchor_required_categories": sorted(self.anchor_required_categories),
            "missing_anchor_action": self.missing_anchor_action.value,
            "review_handling": self.review_handling.value,
            "max_adjacent_lines": self.max_adjacent_lines,
            "custom_person_field_labels": list(self.custom_person_field_labels),
            "custom_address_field_labels": list(self.custom_address_field_labels),
        }
        masking = {"padding": self.mask_padding}
        return {
            "schema_version": POLICY_SCHEMA_VERSION,
            "detection": detection,
            "masking": masking,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MaskingPolicyConfig":
        """JSON에서 읽은 사전을 검증하여 정책 객체로 만든다."""
        version = data.get("schema_version", POLICY_SCHEMA_VERSION)
        if version != POLICY_SCHEMA_VERSION:
            raise ValueError(
                f"지원하지 않는 정책 schema_version입니다: {version}"
            )

        detection = data.get("detection", {})
        if not isinstance(detection, dict):
            raise ValueError("정책의 detection 값은 객체여야 합니다.")

        masking = data.get("masking", {})
        if not isinstance(masking, dict):
            raise ValueError("정책의 masking 값은 객체여야 합니다.")

        defaults = cls()

        # 먼저 JSON 값을 기본 자료형과 enum으로 변환한다. 값을 한 줄씩 읽어 두면
        # 아래 객체 생성 부분에서 중첩된 get 호출을 따라갈 필요가 없다.
        execution_mode = DetectorExecutionMode(
            detection.get("execution_mode", defaults.execution_mode.value)
        )
        merge_mode = DetectorMergeMode(
            detection.get("merge_mode", defaults.merge_mode.value)
        )
        selected_categories = frozenset(
            detection.get("selected_categories", defaults.selected_categories)
        )
        sensitivity_level = MaskingSensitivity(
            detection.get("sensitivity_level", int(defaults.sensitivity_level))
        )
        pattern_strictness = PatternStrictness(
            detection.get("pattern_strictness", defaults.pattern_strictness.value)
        )
        aegis_scope = str(detection.get("aegis_scope", defaults.aegis_scope))
        aegis_entity_types = tuple(
            detection.get("aegis_entity_types", defaults.aegis_entity_types)
        )
        aegis_threshold_adjustment = float(
            detection.get(
                "aegis_threshold_adjustment",
                defaults.aegis_threshold_adjustment,
            )
        )
        minimum_confidence = float(
            detection.get("minimum_confidence", defaults.minimum_confidence)
        )
        minimum_risk_level = int(
            detection.get("minimum_risk_level", defaults.minimum_risk_level)
        )
        checksum_validation_enabled = bool(
            detection.get(
                "checksum_validation_enabled",
                defaults.checksum_validation_enabled,
            )
        )
        checksum_invalid_action = GateAction(
            detection.get(
                "checksum_invalid_action",
                defaults.checksum_invalid_action.value,
            )
        )
        minimum_context_evidence_count = int(
            detection.get(
                "minimum_context_evidence_count",
                defaults.minimum_context_evidence_count,
            )
        )
        missing_context_action = GateAction(
            detection.get(
                "missing_context_action",
                defaults.missing_context_action.value,
            )
        )
        anchor_required_categories = frozenset(
            detection.get(
                "anchor_required_categories",
                defaults.anchor_required_categories,
            )
        )
        missing_anchor_action = GateAction(
            detection.get(
                "missing_anchor_action",
                defaults.missing_anchor_action.value,
            )
        )
        review_handling = ReviewHandling(
            detection.get("review_handling", defaults.review_handling.value)
        )
        max_adjacent_lines = int(
            detection.get("max_adjacent_lines", defaults.max_adjacent_lines)
        )
        custom_person_field_labels = tuple(
            detection.get(
                "custom_person_field_labels",
                defaults.custom_person_field_labels,
            )
        )
        custom_address_field_labels = tuple(
            detection.get(
                "custom_address_field_labels",
                defaults.custom_address_field_labels,
            )
        )
        mask_padding = int(masking.get("padding", defaults.mask_padding))

        return cls(
            execution_mode=execution_mode,
            merge_mode=merge_mode,
            selected_categories=selected_categories,
            sensitivity_level=sensitivity_level,
            pattern_strictness=pattern_strictness,
            aegis_scope=aegis_scope,
            aegis_entity_types=aegis_entity_types,
            aegis_threshold_adjustment=aegis_threshold_adjustment,
            minimum_confidence=minimum_confidence,
            minimum_risk_level=minimum_risk_level,
            checksum_validation_enabled=checksum_validation_enabled,
            checksum_invalid_action=checksum_invalid_action,
            minimum_context_evidence_count=minimum_context_evidence_count,
            missing_context_action=missing_context_action,
            anchor_required_categories=anchor_required_categories,
            missing_anchor_action=missing_anchor_action,
            review_handling=review_handling,
            max_adjacent_lines=max_adjacent_lines,
            custom_person_field_labels=custom_person_field_labels,
            custom_address_field_labels=custom_address_field_labels,
            mask_padding=mask_padding,
        )


def load_masking_policy(path: str | Path = DEFAULT_POLICY_PATH) -> MaskingPolicyConfig:
    """정책 파일을 읽는다. 파일이 없으면 안전한 기본값을 반환한다."""
    policy_path = Path(path)
    if not policy_path.exists():
        return MaskingPolicyConfig()
    data = json.loads(policy_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("마스킹 정책 JSON의 최상위 값은 객체여야 합니다.")
    return MaskingPolicyConfig.from_dict(data)


def save_masking_policy(policy: MaskingPolicyConfig, path: str | Path = DEFAULT_POLICY_PATH) -> Path:
    """검증된 정책을 UTF-8 JSON으로 저장한다."""
    policy_path = Path(path)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(
        json.dumps(policy.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return policy_path
