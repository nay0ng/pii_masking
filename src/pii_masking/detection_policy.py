"""관리자 설정을 개인정보 후보에 적용해 최종 포함 여부를 결정한다.

선택 유형, 신뢰도, 위험도, 체크섬, 문맥 근거와 검수 정책을 순서대로 확인한다.
ko-pii와 AEGIS 결과를 하나만 채택할지, 겹치는 결과만 채택할지도 이 파일에서
결정한다. 이 단계에서는 이미지에 마스크를 그리지 않는다.

코드 검토 순서
1. ``DetectionPolicy``에서 실제 판단에 사용하는 설정값을 확인한다.
2. ``should_execute_rule()``과 ``decide_ner_execution()``에서 두 탐지기의 실행
   순서를 확인한다.
3. ``evaluate_span_policy()``에서 후보 한 건에 유형·신뢰도·체크섬·문맥 Gate를
   적용하는 순서를 확인한다.
4. ``combine_policy_spans()``에서 OR, 일치 결과만 사용, 카테고리 우선 병합 방식을
   확인한다.
5. ``create_policy_run_result()``가 최종 span과 중간 판단 기록을 묶어 반환한다.

이 파일은 후보를 새로 찾지 않는다. 이미 탐지된 PiiSpan을 관리자 정책에 따라
마스킹, 검수 또는 제외로 분류한다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from .categories import PII_CATEGORIES
from .detector_engine import PiiSpan, merge_pii_spans


class DetectorExecutionMode(StrEnum):
    """규칙 탐지기와 NER를 어떤 순서로 실행할지 결정한다."""

    RULE_ONLY = "rule_only"
    NER_ONLY = "ner_only"
    ALWAYS_BOTH = "always_both"
    RULE_FIRST_FALLBACK = "rule_first_fallback"
    CATEGORY_ROUTING = "category_routing"


class DetectorMergeMode(StrEnum):
    """두 탐지기가 반환한 span을 최종 결과로 채택하는 방식."""

    ANY = "any"
    AGREEMENT = "agreement"
    CATEGORY_PRIORITY = "category_priority"


class ReviewHandling(StrEnum):
    """검수 필요 표시가 붙은 후보를 최종 결과에서 다루는 방식."""

    MASK_ALL = "mask_all"
    EXCLUDE_REVIEW = "exclude_review"
    REVIEW_ONLY = "review_only"


class GateAction(StrEnum):
    """검증 조건을 충족하지 못한 후보의 처리 방법."""

    MASK = "mask"
    REVIEW = "review"
    EXCLUDE = "exclude"


DEFAULT_CATEGORIES = frozenset(category.label for category in PII_CATEGORIES)
DEFAULT_NER_CATEGORIES = frozenset({"person_name", "address"})


@dataclass(frozen=True)
class DetectionPolicy:
    """관리자 화면과 API가 공통으로 사용할 탐지·마스킹 설정."""

    execution_mode: DetectorExecutionMode = DetectorExecutionMode.ALWAYS_BOTH
    merge_mode: DetectorMergeMode = DetectorMergeMode.ANY
    selected_categories: frozenset[str] = DEFAULT_CATEGORIES
    ner_categories: frozenset[str] = DEFAULT_NER_CATEGORIES
    minimum_confidence: float = 0.0
    minimum_risk_level: int = 1
    checksum_validation_enabled: bool = True
    checksum_invalid_action: GateAction = GateAction.MASK
    minimum_context_evidence_count: int = 0
    missing_context_action: GateAction = GateAction.REVIEW
    anchor_required_categories: frozenset[str] = frozenset()
    missing_anchor_action: GateAction = GateAction.EXCLUDE
    review_handling: ReviewHandling = ReviewHandling.MASK_ALL

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected_categories",
            frozenset(self.selected_categories),
        )
        object.__setattr__(
            self,
            "ner_categories",
            frozenset(self.ner_categories),
        )
        object.__setattr__(
            self,
            "anchor_required_categories",
            frozenset(self.anchor_required_categories),
        )
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence는 0~1 사이여야 합니다.")
        if self.minimum_risk_level not in {1, 2, 3, 4, 5}:
            raise ValueError("minimum_risk_level은 1~5 사이여야 합니다.")
        if self.minimum_context_evidence_count not in {0, 1, 2, 3}:
            raise ValueError(
                "minimum_context_evidence_count는 0~3 사이여야 합니다."
            )


@dataclass(frozen=True)
class PolicyRunResult:
    """정책 실행 결과와 탐지기별 실제 실행 여부·처리시간."""

    rule_spans: tuple[PiiSpan, ...]
    ner_spans: tuple[PiiSpan, ...]
    final_spans: tuple[PiiSpan, ...]
    rule_executed: bool
    ner_executed: bool
    rule_seconds: float
    ner_seconds: float
    ner_decision: str
    gate_decisions: tuple["SpanPolicyDecision", ...]


@dataclass(frozen=True)
class SpanPolicyDecision:
    """후보 한 건에 Gate 정책을 적용한 결과와 설명."""

    original_span: PiiSpan
    effective_span: PiiSpan
    action: GateAction
    included: bool
    reasons: tuple[str, ...]


_ACTION_PRIORITY = {
    GateAction.MASK: 0,
    GateAction.REVIEW: 1,
    GateAction.EXCLUDE: 2,
}


def _stronger_action(first: GateAction, second: GateAction) -> GateAction:
    return max((first, second), key=_ACTION_PRIORITY.__getitem__)


def _has_invalid_checksum(span: PiiSpan) -> bool:
    return any(
        evidence.startswith(("checksum:invalid", "checksum:fail"))
        for evidence in span.evidence
    )


def _has_valid_checksum(span: PiiSpan) -> bool:
    return any(
        evidence == "checksum:valid"
        for evidence in span.evidence
    )


def _has_anchor_evidence(span: PiiSpan) -> bool:
    if span.rule_id.startswith("context:"):
        return True
    return any(
        evidence.startswith(("anchor:", "keyword:", "context:", "field:"))
        for evidence in span.evidence
    )


def _context_evidence_gates(span: PiiSpan) -> frozenset[str]:
    """ko-pii evidence를 필드명·키워드·문맥 Gate로 분류한다."""
    gates: set[str] = set()
    if span.rule_id.startswith("context:"):
        gates.add("field")
    for evidence in span.evidence:
        if (
            evidence == "anchor:prefix"
            or evidence.startswith("pos:field_label")
            or evidence == "structure:address-field"
        ):
            gates.add("field")
        if (
            evidence.startswith("keyword:")
            and evidence != "keyword:None"
        ) or evidence.startswith(
            ("anchor:context_keyword", "anchor:주소", "pos:title", "marker:")
        ):
            gates.add("keyword")
        if evidence.startswith(
            (
                "dict:",
                "layout:",
                "purpose:",
                "type:",
                "position:",
                "pos:name_",
                "pos:surname",
                "pos:particle",
                "pos:co_occurrence",
                "pos:deterministic",
            )
        ):
            gates.add("context")
    return frozenset(gates)


def _has_format_evidence(span: PiiSpan) -> bool:
    return any(
        evidence.startswith(
            ("pattern:", "format:", "checksum:", "date_valid:", "date:")
        )
        for evidence in span.evidence
    )


def evaluate_span_policy(
    span: PiiSpan,
    policy: DetectionPolicy,
) -> SpanPolicyDecision:
    """한 span에 유형·신뢰도·위험도·체크섬·anchor 정책을 적용한다."""
    reasons: list[str] = []
    action = GateAction.MASK
    explicit_gate_action = False

    if span.entity_type not in policy.selected_categories:
        return SpanPolicyDecision(
            span,
            span,
            GateAction.EXCLUDE,
            False,
            ("선택하지 않은 개인정보 유형",),
        )
    if span.detector_confidence < policy.minimum_confidence:
        return SpanPolicyDecision(
            span,
            span,
            GateAction.EXCLUDE,
            False,
            (
                "신뢰도 미달: "
                f"{span.detector_confidence:.2f} < {policy.minimum_confidence:.2f}",
            ),
        )
    if (
        span.risk_level is not None
        and span.risk_level < policy.minimum_risk_level
    ):
        return SpanPolicyDecision(
            span,
            span,
            GateAction.EXCLUDE,
            False,
            (
                "위험도 미달: "
                f"{span.risk_level} < {policy.minimum_risk_level}",
            ),
        )

    context_gates = _context_evidence_gates(span)
    if context_gates:
        reasons.append(
            "문맥 Gate 통과: " + ", ".join(sorted(context_gates))
        )
    if _has_format_evidence(span):
        reasons.append("형식 Gate 근거 확인")

    if not policy.checksum_validation_enabled:
        if _has_valid_checksum(span) or _has_invalid_checksum(span) or any(
            evidence == "checksum:skipped" for evidence in span.evidence
        ):
            reasons.append("체크섬 Gate 비활성화")
    elif _has_valid_checksum(span):
        reasons.append("체크섬 정상")
    elif _has_invalid_checksum(span):
        explicit_gate_action = True
        action = _stronger_action(
            action,
            policy.checksum_invalid_action,
        )
        reasons.append(
            "체크섬 불일치 → "
            f"{policy.checksum_invalid_action.value}"
        )

    if (
        span.risk_level is not None
        and policy.minimum_context_evidence_count > 0
        and len(context_gates) < policy.minimum_context_evidence_count
        and not (
            policy.checksum_validation_enabled
            and _has_valid_checksum(span)
        )
    ):
        explicit_gate_action = True
        action = _stronger_action(action, policy.missing_context_action)
        reasons.append(
            "필드명·키워드·문맥 Gate 부족: "
            f"{len(context_gates)}/"
            f"{policy.minimum_context_evidence_count} → "
            f"{policy.missing_context_action.value}"
        )

    # anchor 필수 설정은 ko-pii/자체 규칙 후보에만 적용한다.
    # AEGIS는 ko-pii RiskLevel을 갖지 않으며 문맥 모델 자체의 결과이므로,
    # 같은 개인정보 유형을 선택해도 이 Gate 때문에 제거하지 않는다.
    if (
        span.risk_level is not None
        and span.entity_type in policy.anchor_required_categories
    ):
        if _has_anchor_evidence(span):
            reasons.append("anchor 근거 확인")
        else:
            explicit_gate_action = True
            action = _stronger_action(
                action,
                policy.missing_anchor_action,
            )
            reasons.append(
                "anchor 근거 없음 → "
                f"{policy.missing_anchor_action.value}"
            )

    if span.review_required and not (
        explicit_gate_action and action == GateAction.MASK
    ):
        action = _stronger_action(action, GateAction.REVIEW)
        reasons.append("탐지기가 검수 필요로 표시")
    elif (
        span.review_required
        and explicit_gate_action
        and action == GateAction.MASK
    ):
        reasons.append("관리자 Gate 설정으로 자동 마스킹")

    if action == GateAction.REVIEW and not span.review_required:
        effective_span = replace(span, review_required=True)
    elif action == GateAction.MASK and span.review_required and explicit_gate_action:
        effective_span = replace(span, review_required=False)
    else:
        effective_span = span
    included = action != GateAction.EXCLUDE
    if action == GateAction.REVIEW:
        if policy.review_handling == ReviewHandling.EXCLUDE_REVIEW:
            included = False
            reasons.append("검수 후보 제외 정책")
        elif policy.review_handling == ReviewHandling.REVIEW_ONLY:
            reasons.append("검수 후보 전용 결과에 포함")
        else:
            reasons.append("검수 표시 후 마스킹")
    elif policy.review_handling == ReviewHandling.REVIEW_ONLY:
        included = False
        reasons.append("검수 후보가 아니므로 제외")

    if not reasons:
        reasons.append("Gate 정책 통과")
    return SpanPolicyDecision(
        original_span=span,
        effective_span=effective_span,
        action=action,
        included=included,
        reasons=tuple(reasons),
    )


def _filter_spans(
    spans: tuple[PiiSpan, ...],
    policy: DetectionPolicy,
) -> tuple[PiiSpan, ...]:
    return tuple(
        decision.effective_span
        for span in spans
        if (decision := evaluate_span_policy(span, policy)).included
    )


def should_execute_rule(policy: DetectionPolicy) -> bool:
    """현재 정책에서 규칙 탐지기를 실행해야 하는지 반환한다."""
    if not policy.selected_categories:
        return False
    if policy.execution_mode == DetectorExecutionMode.NER_ONLY:
        return False
    if policy.execution_mode == DetectorExecutionMode.CATEGORY_ROUTING:
        return bool(policy.selected_categories - policy.ner_categories)
    return True


def decide_ner_execution(
    policy: DetectionPolicy,
    rule_spans: tuple[PiiSpan, ...],
) -> tuple[bool, str]:
    """규칙 결과와 선택 유형을 기준으로 AEGIS 실행 여부를 결정한다."""
    target_categories = policy.selected_categories & policy.ner_categories
    if not target_categories:
        return False, "선택한 유형 중 NER 담당 유형이 없음"

    if policy.execution_mode == DetectorExecutionMode.RULE_ONLY:
        return False, "규칙 탐지만 선택"
    if policy.execution_mode == DetectorExecutionMode.NER_ONLY:
        return True, "NER 단독 실행"
    if policy.execution_mode == DetectorExecutionMode.ALWAYS_BOTH:
        return True, "항상 두 탐지기 실행"
    if policy.execution_mode == DetectorExecutionMode.CATEGORY_ROUTING:
        return True, "이름·주소를 NER에 라우팅"

    covered_categories = {
        span.entity_type
        for span in _filter_spans(rule_spans, policy)
        if span.entity_type in target_categories
    }
    missing_categories = target_categories - covered_categories
    if missing_categories:
        return True, "규칙에서 누락된 NER 유형: " + ", ".join(
            sorted(missing_categories)
        )
    return False, "규칙이 선택된 NER 담당 유형을 모두 탐지"


def _spans_overlap(first: PiiSpan, second: PiiSpan) -> bool:
    if first.entity_type != second.entity_type:
        return False
    if set(first.token_ids) & set(second.token_ids):
        return True
    return first.raw_start < second.raw_end and second.raw_start < first.raw_end


def combine_policy_spans(
    rule_spans: tuple[PiiSpan, ...],
    ner_spans: tuple[PiiSpan, ...],
    policy: DetectionPolicy,
) -> tuple[PiiSpan, ...]:
    """이미 실행된 탐지 결과에 관리자 필터와 결합 정책을 적용한다."""
    filtered_rules = _filter_spans(rule_spans, policy)
    filtered_ner = _filter_spans(ner_spans, policy)

    if policy.execution_mode == DetectorExecutionMode.CATEGORY_ROUTING:
        filtered_rules = tuple(
            span
            for span in filtered_rules
            if span.entity_type not in policy.ner_categories
        )
        filtered_ner = tuple(
            span
            for span in filtered_ner
            if span.entity_type in policy.ner_categories
        )

    if policy.merge_mode == DetectorMergeMode.ANY:
        return merge_pii_spans(filtered_rules, filtered_ner)

    if policy.merge_mode == DetectorMergeMode.AGREEMENT:
        agreed_rules = tuple(
            span
            for span in filtered_rules
            if any(_spans_overlap(span, other) for other in filtered_ner)
        )
        agreed_ner = tuple(
            span
            for span in filtered_ner
            if any(_spans_overlap(span, other) for other in filtered_rules)
        )
        return merge_pii_spans(agreed_rules, agreed_ner)

    # 이름·주소는 같은 위치에서 NER가 탐지했다면 NER 범위를 우선하고,
    # NER가 찾지 못한 규칙 결과와 구조화 번호 규칙 결과는 그대로 보존한다.
    preferred_rules = tuple(
        span
        for span in filtered_rules
        if span.entity_type not in policy.ner_categories
        or not any(_spans_overlap(span, other) for other in filtered_ner)
    )
    return merge_pii_spans(preferred_rules, filtered_ner)


def create_policy_run_result(
    policy: DetectionPolicy,
    *,
    rule_spans: tuple[PiiSpan, ...],
    ner_spans: tuple[PiiSpan, ...],
    rule_executed: bool,
    ner_executed: bool,
    rule_seconds: float,
    ner_seconds: float,
    ner_decision: str,
) -> PolicyRunResult:
    """이미 실행한 두 탐지기의 결과를 결합하고 Gate 판단 기록을 만듭니다."""
    final_spans = combine_policy_spans(rule_spans, ner_spans, policy)
    final_keys = {
        (
            span.entity_type,
            span.token_ids,
            span.rule_id,
        )
        for span in final_spans
    }
    gate_decisions: list[SpanPolicyDecision] = []
    for span in (*rule_spans, *ner_spans):
        decision = evaluate_span_policy(span, policy)
        decision_key = (
            decision.effective_span.entity_type,
            decision.effective_span.token_ids,
            decision.effective_span.rule_id,
        )
        if decision.included and decision_key not in final_keys:
            decision = replace(
                decision,
                action=GateAction.EXCLUDE,
                included=False,
                reasons=(
                    *decision.reasons,
                    "탐지기 결합·중복 정책에서 제외",
                ),
            )
        gate_decisions.append(decision)

    return PolicyRunResult(
        rule_spans=rule_spans,
        ner_spans=ner_spans,
        final_spans=final_spans,
        rule_executed=rule_executed,
        ner_executed=ner_executed,
        rule_seconds=rule_seconds,
        ner_seconds=ner_seconds,
        ner_decision=ner_decision,
        gate_decisions=tuple(gate_decisions),
    )
