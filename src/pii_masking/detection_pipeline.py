"""전처리된 문서 한 건에 개인정보 탐지 정책을 적용하는 전체 흐름이다.

규칙 탐지 실행, AEGIS 실행 여부 결정, 두 결과의 병합과 정책 필터를 정해진
순서로 호출한다. 반환값에는 중간 결과와 단계별 실행 시간이 함께 들어 있다.
이미지 마스킹은 이 함수의 반환값을 받은 호출부에서 수행한다.

코드 검토는 ``detect_document_pii()`` 한 함수만 위에서 아래로 읽으면 된다.
``detectors.detect_all_pii_spans()``에서 ko-pii 계열 결과를 만들고,
``aegis_ner_adapter.detect_aegis_spans()``에서 NER 결과를 만든 뒤,
``detection_policy.create_policy_run_result()``에서 두 결과를 병합·필터한다.
"""

from __future__ import annotations

from time import perf_counter

from .aegis_detector import AegisOnnxNer, detect_aegis_spans
from .detection_policy import (
    PolicyRunResult,
    create_policy_run_result,
    decide_ner_execution,
    should_execute_rule,
)
from .detection_postprocessing import postprocess_detection_spans
from .pii_detectors import detect_all_pii_spans
from .policy import MaskingPolicyConfig
from .models import PreprocessedDocument
from .search_views import SearchViewSettings


def detect_document_pii(
    document: PreprocessedDocument,
    policy_config: MaskingPolicyConfig,
    search_settings: SearchViewSettings,
    ner: AegisOnnxNer | None = None,
) -> PolicyRunResult:
    """관리자 정책에 따라 두 탐지기를 실행하고 최종 PII span을 반환한다."""
    # 화면과 JSON에서 사용하는 설정을 실제 탐지 판단용 정책으로 변환한다.
    detection_policy = policy_config.to_detection_policy()

    # 규칙 전용, NER 전용 같은 실행 방식에 따라 ko-pii 실행 여부가 달라진다.
    rule_executed = should_execute_rule(detection_policy)
    rule_started = perf_counter()
    rule_spans = ()
    if rule_executed:
        rule_spans = detect_all_pii_spans(document, search_settings=search_settings)
        rule_spans = postprocess_detection_spans(document, rule_spans)
    rule_seconds = perf_counter() - rule_started if rule_executed else 0.0

    # 규칙 우선 모드에서는 규칙 결과를 확인한 다음 AEGIS 실행 여부를 결정한다.
    ner_executed, ner_decision = decide_ner_execution(detection_policy, rule_spans)
    ner_started = perf_counter()
    ner_spans = ()
    if ner_executed:
        ner_spans = detect_aegis_spans(
            document,
            scope=policy_config.aegis_scope,
            threshold_adjustment=policy_config.aegis_threshold_adjustment,
            enabled_entity_types=policy_config.aegis_entity_types,
            review_required=True,
            search_settings=search_settings,
            ner=ner,
        )
        ner_spans = postprocess_detection_spans(document, ner_spans)
    ner_seconds = perf_counter() - ner_started if ner_executed else 0.0

    # 두 탐지 결과에 유형·신뢰도·체크섬·검수 정책을 적용해 최종 span을 만든다.
    return create_policy_run_result(
        detection_policy,
        rule_spans=rule_spans,
        ner_spans=ner_spans,
        rule_executed=rule_executed,
        ner_executed=ner_executed,
        rule_seconds=rule_seconds,
        ner_seconds=ner_seconds,
        ner_decision=ner_decision,
    )
