"""핵심 단계만 남긴 PII 마스킹 파이프라인 검수 화면.

``streamlit_app.py``보다 화면 내용을 덜어내고 현재 운영 후보 정책의 결과를
문서별로 확인하는 용도다. 개인정보 탐지 규칙은 이 파일에 구현하지 않고
``src/pii_masking``의 동일한 엔진을 호출한다.

코드 검토 순서
1. ``load_and_process()``에서 OCR JSON 읽기와 전처리 cache를 확인한다.
2. ``load_aegis_runtime()``에서 AEGIS 모델 session cache를 확인한다.
3. ``main()``의 sidebar에서 입력 경로, 문서와 적용 정책을 선택한다.
4. 화면 설정을 ``MaskingPolicyConfig``와 ``SearchViewSettings``로 조립한다.
5. ``detect_document_pii()``로 ko-pii·AEGIS·정책 병합을 실행한다.
6. 단계별 탭에서 OCR, SearchView, 탐지 span, bbox와 마스킹 결과를 확인한다.

화면용 표와 bbox 그림은 ``dashboard_support.py``에서 만들며, 배치 실행은
프로젝트 루트의 ``run_full_pipeline.py``가 담당한다.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
import warnings

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
KO_PII_SRC = PROJECT_ROOT / "ko-pii" / "src"
# 이 파일을 Streamlit로 직접 실행해도 프로젝트 모듈을 찾을 수 있도록
# 프로젝트 루트와 엔진 src를 Python 검색 경로에 추가한다.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(KO_PII_SRC) not in sys.path:
    sys.path.insert(0, str(KO_PII_SRC))

from pii_masking import (
    DEFAULT_POLICY_PATH,
    MaskingPolicyConfig,
    MaskingSensitivity,
    PiiSpan,
    DetectorExecutionMode,
    DetectorMergeMode,
    GateAction,
    PatternStrictness,
    ReviewHandling,
    SearchViewSettings,
    combine_policy_spans,
    detect_all_pii_spans,
    get_sensitivity_preset,
    load_masking_policy,
    load_ocr_results,
    preprocess_document,
)
from pii_masking.aegis_detector import (
    AegisOnnxNer,
    ALL_AEGIS_ENTITY_TYPES,
    SEMANTIC_AEGIS_ENTITY_TYPES,
    trace_aegis_predictions,
)
from pii_masking.image_masker import render_masked_image
from pii_masking.detection_pipeline import detect_document_pii
from apps.dashboard_support import (
    CATEGORY_NAMES,
    build_document_search_views,
    discover_result_files,
    render_span_overlay,
    render_token_overlay,
    span_rows,
    token_rows,
    trace_ko_pii,
)


DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "sample_data"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "sample_data"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
POC_POLICY_PATH = PROJECT_ROOT / "config" / "poc_policy.json"
DEFAULT_VIEWER_POLICY_PATH = (
    POC_POLICY_PATH if POC_POLICY_PATH.is_file() else DEFAULT_POLICY_PATH
)
PIPELINE_CACHE_VERSION = "admin-policy-v3-sample-data"
MASK_PREVIEW_WIDTH = 520

EXECUTION_MODE_NAMES = {
    "rule_only": "ko-pii·자체 규칙만",
    "ner_only": "AEGIS만",
    "always_both": "ko-pii·자체 규칙 + AEGIS",
    "rule_first_fallback": "규칙 우선, 부족할 때 AEGIS",
    "category_routing": "개인정보 유형별 탐지기 분담", # test용으로 어떤 유형으로 탐지할 지 미정.
}
MERGE_MODE_NAMES = {
    "any": "한 탐지기만 찾아도 유지",
    "agreement": "두 탐지기가 함께 찾은 결과만 유지",
    "category_priority": "유형별 우선 탐지기 적용",
}
REVIEW_HANDLING_NAMES = {
    "mask_all": "검수 표시와 관계없이 모두 마스킹",
    "exclude_review": "검수 필요 후보는 자동 마스킹에서 제외",
    "review_only": "검수 필요 후보만 결과에 표시",
}


def format_path_name(path: Path) -> str:
    return path.name


class DocumentIndexFormatter:
    """문서 선택값은 정수로 유지하고 화면에는 파일명만 표시한다."""

    def __init__(self, paths: tuple[str, ...]) -> None:
        self.paths = paths

    def __call__(self, index: int) -> str:
        return Path(self.paths[index]).name


def format_sensitivity(value: MaskingSensitivity) -> str:
    return SENSITIVITY_NAMES[value]


def format_execution_mode(mode: DetectorExecutionMode) -> str:
    return EXECUTION_MODE_NAMES[mode.value]


def format_merge_mode(mode: DetectorMergeMode) -> str:
    return MERGE_MODE_NAMES[mode.value]


def format_category(label: str) -> str:
    return f"{CATEGORY_NAMES.get(label, label)} ({label})"


def format_aegis_entity(entity: str) -> str:
    return f"{AEGIS_ENTITY_NAMES.get(entity, entity)} ({entity})"


def format_pattern_strictness(value: PatternStrictness) -> str:
    return PATTERN_STRICTNESS_NAMES[value]


def format_risk_level(value: int) -> str:
    return RISK_LEVEL_NAMES[value]


def format_gate_action(value: GateAction) -> str:
    return GATE_ACTION_NAMES[value.value]


def format_review_handling(value: ReviewHandling) -> str:
    return REVIEW_HANDLING_NAMES[value.value]


def format_span(span: PiiSpan) -> str:
    category = CATEGORY_NAMES.get(span.entity_type, span.entity_type)
    text = span.text.replace("\n", " ↵ ")
    return f"{category} — {text}"


GATE_ACTION_NAMES = {
    "mask": "자동 마스킹",
    "review": "검수 필요로 전환",
    "exclude": "자동 마스킹에서 제외",
}
RISK_LEVEL_NAMES = {
    1: "INFO 이상 — 모두 허용",
    2: "LOW 이상",
    3: "MEDIUM 이상",
    4: "HIGH 이상",
    5: "CRITICAL만",
}
SENSITIVITY_NAMES = {
    MaskingSensitivity.PRECISE: "1단계 정확 우선",
    MaskingSensitivity.STANDARD: "2단계 기본",
    MaskingSensitivity.SENSITIVE: "3단계 민감",
    MaskingSensitivity.MAXIMUM: "4단계 최대",
}
SENSITIVITY_DESCRIPTIONS = {
    MaskingSensitivity.PRECISE: (
        "정확한 형식과 강한 근거를 우선하며, 검수 후보는 자동 마스킹에서 제외합니다."
    ),
    MaskingSensitivity.STANDARD: (
        "토큰 공백과 하이픈 분리를 복원하고, 애매한 후보도 검수 표시 후 마스킹합니다."
    ),
    MaskingSensitivity.SENSITIVE: (
        "중복 하이픈까지 복구하고 체크섬 불일치 후보도 누락 방지를 위해 마스킹합니다."
    ),
    MaskingSensitivity.MAXIMUM: (
        "숫자 구간의 O/0·I/1 OCR 혼동까지 제한적으로 복구하고 체크섬을 요구하지 않습니다."
    ),
}
PATTERN_STRICTNESS_NAMES = {
    PatternStrictness.EXACT: "정확 형식만",
    PatternStrictness.NORMALIZED: "공백·토큰 분리 복원",
    PatternStrictness.RECOVERED: "중복 하이픈·구분자 복구",
    PatternStrictness.OCR_TOLERANT: "숫자 OCR O/0·I/1까지 제한 복구",
}
AEGIS_ENTITY_NAMES = {
    "SURNAME": "성",
    "GIVENNAME": "이름",
    "USERNAME": "사용자 ID",
    "EMAIL": "이메일",
    "TELEPHONENUM": "전화번호",
    "DATEOFBIRTH": "생년월일",
    "CREDITCARDNUMBER": "카드번호",
    "IDCARD": "신분증 번호",
    "STREET": "도로·거리 주소",
    "CITY": "도시·행정구역",
    "ZIPCODE": "우편번호",
    "BUILDINGNUM": "건물번호",
    "IP_ADDRESS": "IP 주소",
    "PASSWORD": "비밀번호",
    "ACCOUNTNUM": "계좌번호",
    "DRIVERLICENSENUM": "운전면허번호",
    "TIME": "시간",
    "COMPANY": "회사명",
}


def _parse_field_labels(value: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            label.strip()
            for line in value.splitlines()
            for label in line.split(",")
            if label.strip()
        )
    )


def _image_to_png_bytes(image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@st.cache_data(show_spinner="OCR JSON을 읽고 전처리하는 중입니다...")
def load_and_process(
    results_path: str,
    image_root: str,
    modified_ns: int,
    pipeline_version: str,
):
    """OCR JSON과 전처리 결과를 cache하고 정책에 따른 탐지는 포함하지 않는다."""
    del modified_ns
    del pipeline_version
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        documents = load_ocr_results(results_path, skip_failed_responses=True, image_root=image_root)
    items = tuple(
        (document, preprocess_document(document))
        for document in documents
    )
    return items, tuple(str(item.message) for item in caught)


@st.cache_resource(show_spinner="AEGIS 모델을 불러오는 중입니다...")
def load_aegis_runtime(model_dir: str) -> AegisOnnxNer:
    """화면 재실행 시 ONNX 모델을 다시 로드하지 않도록 session을 보관한다."""
    return AegisOnnxNer(model_dir)


def _resolve_configured_path(value: str) -> Path:
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _reset_document_selection() -> None:
    st.session_state["document_selector"] = 0


def _reset_sensitivity_overrides() -> None:
    """민감도 단계가 바뀌면 이전 단계의 고급 위젯 값을 제거한다."""
    for key in (
        "advanced_pattern_strictness",
        "advanced_checksum_enabled",
        "advanced_minimum_risk",
        "advanced_checksum_action",
        "advanced_context_count",
        "advanced_context_action",
        "advanced_anchor_categories",
        "advanced_missing_anchor_action",
        "advanced_aegis_threshold",
        "advanced_review_handling",
    ):
        st.session_state.pop(key, None)


def _reset_policy_widgets() -> None:
    """정책 파일이 바뀌면 이전 정책에서 남은 위젯 값을 제거한다."""
    for key in (
        "policy_sensitivity_level_v2",
        "policy_customize_sensitivity",
        "policy_execution_mode",
        "policy_merge_mode",
        "policy_selected_categories_v2",
        "policy_aegis_entities",
        "policy_minimum_confidence",
    ):
        st.session_state.pop(key, None)
    _reset_sensitivity_overrides()


def _move_document(delta: int, document_count: int) -> None:
    current = int(st.session_state.get("document_selector", 0))
    st.session_state["document_selector"] = min(
        document_count - 1,
        max(0, current + delta),
    )


def _category_summary(spans) -> str:
    counts: dict[str, int] = {}
    for span in spans:
        counts[span.entity_type] = counts.get(span.entity_type, 0) + 1
    return ", ".join(
        f"{CATEGORY_NAMES.get(label, label)} {count}"
        for label, count in sorted(counts.items())
    ) or "탐지 결과 없음"


def _policy_summary(policy_config) -> str:
    execution_name = EXECUTION_MODE_NAMES[
        policy_config.execution_mode.value
    ]
    merge_name = MERGE_MODE_NAMES[policy_config.merge_mode.value]
    aegis_count = len(policy_config.aegis_entity_types)
    return (
        f"**민감도:** "
        f"{SENSITIVITY_NAMES[policy_config.sensitivity_level]}  \n"
        f"**패턴 Gate:** "
        f"{PATTERN_STRICTNESS_NAMES[policy_config.pattern_strictness]}  \n"
        f"**체크섬 Gate:** "
        f"{'사용' if policy_config.checksum_validation_enabled else '사용 안 함'}  \n"
        f"**필드·키워드·문맥 Gate:** 최소 "
        f"{policy_config.minimum_context_evidence_count}개 근거  \n"
        f"**실행:** {execution_name}  \n"
        f"**결합:** {merge_name}  \n"
        f"**인접 줄 탐색:** 최대 3줄 자동  \n"
        f"**AEGIS 라벨:** {aegis_count}개  \n"
        f"**AEGIS threshold 보정:** "
        f"{policy_config.aegis_threshold_adjustment:+.2f}  \n"
        f"**마스킹 유형:** {len(policy_config.selected_categories)}개  \n"
        f"**공통 최소 신뢰도:** {policy_config.minimum_confidence:.2f}  \n"
        f"**ko-pii 최소 위험도:** "
        f"{RISK_LEVEL_NAMES[policy_config.minimum_risk_level]}  \n"
        f"**체크섬 불일치:** "
        f"{GATE_ACTION_NAMES[policy_config.checksum_invalid_action.value] if policy_config.checksum_validation_enabled else '검증하지 않음'}  \n"
        f"**anchor 필수 유형:** "
        f"{len(policy_config.anchor_required_categories)}개  \n"
        f"**검수 후보 처리:** "
        f"{REVIEW_HANDLING_NAMES[policy_config.review_handling.value]}"
    )


def _gate_decision_rows(decisions) -> list[dict[str, object]]:
    rows = []
    for index, decision in enumerate(decisions):
        span = decision.original_span
        if decision.included and decision.action == GateAction.REVIEW:
            outcome = "검수 표시 후 마스킹"
        elif decision.included:
            outcome = "자동 마스킹"
        else:
            outcome = "제외"
        rows.append({
            "#": index + 1,
            "유형": CATEGORY_NAMES.get(span.entity_type, span.entity_type),
            "탐지 문자열": span.text.replace("\n", " ↵ "),
            # Streamlit 표는 한 열에 숫자와 문자열이 섞이면 Arrow 변환 경고가
            # 발생하므로 표시용 위험도는 모두 문자열로 통일한다.
            "위험도": (
                RISK_LEVEL_NAMES.get(
                    span.risk_level,
                    f"레벨 {span.risk_level}",
                )
                if span.risk_level is not None
                else "해당 없음"
            ),
            "신뢰도": round(span.detector_confidence, 3),
            "Gate 동작": GATE_ACTION_NAMES[decision.action.value],
            "최종 처리": outcome,
            "판정 이유": " / ".join(decision.reasons),
            "탐지 근거": ", ".join(span.evidence),
            "규칙": span.rule_id,
        })
    return rows


def main() -> None:
    """현재 문서에 정책을 적용하고 핵심 중간 결과와 마스킹 결과를 표시한다."""
    # 1. 페이지와 입력 경로를 준비한다.
    st.set_page_config(
        page_title="PII 마스킹 검수",
        page_icon="🔎",
        layout="wide",
    )
    st.title("PII 마스킹 검수")
    st.caption(
        "사이드바에서 정책을 바꾸면 현재 문서의 탐지·마스킹 결과에 즉시 반영됩니다."
    )

    with st.sidebar:
        with st.expander("경로 설정", expanded=False):
            results_root_text = st.text_input("OCR 결과 폴더", value=str(DEFAULT_RESULTS_ROOT))
            image_root_text = st.text_input("원본 이미지 폴더", value=str(DEFAULT_IMAGE_ROOT))
            policy_path_text = st.text_input(
                "정책 JSON",
                value=str(DEFAULT_VIEWER_POLICY_PATH),
                key="policy_path_input",
                on_change=_reset_policy_widgets,
            )
            model_dir_text = st.text_input("AEGIS 모델 폴더", value=str(DEFAULT_MODEL_DIR))

    results_root = _resolve_configured_path(results_root_text)
    image_root = _resolve_configured_path(image_root_text)
    policy_path = _resolve_configured_path(policy_path_text)
    model_dir = _resolve_configured_path(model_dir_text)

    if not results_root.is_dir():
        st.error(f"OCR 결과 폴더를 찾지 못했습니다: {results_root}")
        st.stop()
    if not image_root.is_dir():
        st.error(f"원본 이미지 폴더를 찾지 못했습니다: {image_root}")
        st.stop()
    if not policy_path.is_file():
        st.error(f"정책 JSON을 찾지 못했습니다: {policy_path}")
        st.stop()
    if not model_dir.is_dir():
        st.error(f"AEGIS 모델 폴더를 찾지 못했습니다: {model_dir}")
        st.stop()

    # 2. 저장된 policy.json은 화면 위젯의 초기값으로 사용한다.
    try:
        saved_policy_config = load_masking_policy(policy_path)
    except Exception as exc:
        st.error(f"마스킹 정책을 읽지 못했습니다: {exc}")
        st.stop()

    policy_signature = (str(policy_path), policy_path.stat().st_mtime_ns)
    if st.session_state.get("loaded_policy_signature") != policy_signature:
        _reset_policy_widgets()
        st.session_state["loaded_policy_signature"] = policy_signature

    result_files = discover_result_files(results_root)
    if not result_files:
        st.error("sample_data에서 OCR 결과 JSON을 찾지 못했습니다.")
        st.stop()

    with st.sidebar:
        st.header("검수 대상")
        default_result_index = next(
            (
                index
                for index, path in enumerate(result_files)
                if path.name == "test_docs_ocr_results.json"
            ),
            0,
        )
        
        selected_file = st.selectbox(
            "OCR 결과 JSON",
            result_files,
            index=default_result_index,
            format_func=format_path_name,
            key="result_file_selector",
            on_change=_reset_document_selection,
        )

    # 선택한 OCR JSON을 읽고 OCR 파싱과 전처리를 수행한다.
    # 개인정보 탐지는 sidebar 설정을 모두 받은 뒤 별도로 실행한다.
    try:
        items, load_warnings = load_and_process(
            str(selected_file),
            str(image_root),
            selected_file.stat().st_mtime_ns,
            PIPELINE_CACHE_VERSION,
        )
    except Exception as exc:
        st.exception(exc)
        st.stop()

    for message in load_warnings:
        st.warning(message)
    if not items:
        st.warning("선택한 JSON에 정상 처리된 문서가 없습니다.")
        st.stop()

    # 3. 현재 문서에 적용할 정책을 sidebar에서 입력받는다.
    with st.sidebar:
        document_paths = tuple(
            item[0].image_path.as_posix()
            for item in items
        )
        selected_document_index = st.selectbox(
            "문서",
            range(len(items)),
            format_func=DocumentIndexFormatter(document_paths),
            key="document_selector",
        )
        st.divider()
        st.header("마스킹 정책")
        st.caption("값을 변경하면 현재 문서에 바로 적용됩니다.")
        sensitivity_level = st.selectbox(
            "마스킹 민감도",
            tuple(MaskingSensitivity),
            index=tuple(MaskingSensitivity).index(
                saved_policy_config.sensitivity_level
            ),
            format_func=format_sensitivity,
            key="policy_sensitivity_level_v2",
            on_change=_reset_sensitivity_overrides,
        )
        sensitivity_preset = get_sensitivity_preset(sensitivity_level)
        st.info(SENSITIVITY_DESCRIPTIONS[sensitivity_level])
        customize_sensitivity = st.checkbox(
            "정책 JSON 세부값 그대로 적용",
            value=True,
            key="policy_customize_sensitivity",
            help=(
                "켜면 민감도 프리셋 대신 선택한 정책 JSON의 패턴·체크섬·"
                "위험도·문맥·AEGIS·검수 설정을 그대로 사용합니다."
            ),
        )
        execution_mode = st.selectbox(
            "탐지기 실행 방식",
            tuple(DetectorExecutionMode),
            index=tuple(DetectorExecutionMode).index(
                saved_policy_config.execution_mode
            ),
            format_func=format_execution_mode,
            key="policy_execution_mode",
            help=(
                "규칙 우선 방식은 선택한 NER 담당 유형이 규칙에서 발견되면 "
                "AEGIS 실행을 생략합니다."
            ),
        )
        st.caption(
            "개인정보 유형별 탐지기 분담은 테스트용으로 "
            "어떤 유형을 어떤 탐지기가 담당할지 아직 미정입니다."
        )
        merge_mode = st.selectbox(
            "탐지 결과 결합 방식",
            tuple(DetectorMergeMode),
            index=tuple(DetectorMergeMode).index(
                saved_policy_config.merge_mode
            ),
            format_func=format_merge_mode,
            key="policy_merge_mode",
        )
        category_options = tuple(sorted(CATEGORY_NAMES))
        selected_categories = st.multiselect(
            "마스킹할 개인정보 유형",
            category_options,
            default=[
                label
                for label in category_options
                if label in saved_policy_config.selected_categories
            ],
            format_func=format_category,
            key="policy_selected_categories_v2",
        )
        st.subheader("문서 구조 탐색")
        st.caption(
            "주소 연속 줄과 금융기관·계좌·예금주 구조를 포함해 "
            "인접한 최대 3줄을 자동으로 검사합니다."
        )
    
        st.subheader("AEGIS")
        selected_aegis_entities = st.multiselect(
            "사용할 AEGIS 라벨",
            ALL_AEGIS_ENTITY_TYPES,
            default=list(saved_policy_config.aegis_entity_types),
            format_func=format_aegis_entity,
            disabled=execution_mode == DetectorExecutionMode.RULE_ONLY,
            key="policy_aegis_entities",
            help=(
                "AEGIS가 예측할 세부 라벨입니다. 성·이름·주소 외 라벨도 "
                "실험할 수 있으며, 최종 마스킹 유형 선택을 한 번 더 적용합니다."
            ),
        )
        if customize_sensitivity:
            aegis_threshold_adjustment = st.slider(
                "AEGIS threshold 보정",
                min_value=-0.20,
                max_value=0.10,
                value=float(
                    saved_policy_config.aegis_threshold_adjustment
                ),
                step=0.05,
                disabled=execution_mode == DetectorExecutionMode.RULE_ONLY,
                key="advanced_aegis_threshold",
                help=(
                    "AEGIS 라벨별 기본 threshold에 더하는 값입니다. 음수는 더 많이 "
                    "찾고 오탐도 늘 수 있으며, 양수는 더 엄격하게 찾습니다."
                ),
            )
        else:
            aegis_threshold_adjustment = (
                sensitivity_preset.aegis_threshold_adjustment
            )
            st.caption(
                "현재 민감도에서 AEGIS threshold 보정: "
                f"{aegis_threshold_adjustment:+.2f}"
            )

        st.subheader("최종 후보 처리")
        minimum_confidence = st.slider(
            "공통 최소 신뢰도(실험용)",
            min_value=0.0,
            max_value=1.0,
            value=float(saved_policy_config.minimum_confidence),
            step=0.05,
            key="policy_minimum_confidence",
            help=(
                "0.0이면 사용하지 않습니다. ko-pii 점수는 규칙 충족 정도이고 "
                "AEGIS 점수는 모델 확률이라 같은 척도로 직접 비교하기 어렵습니다."
            ),
        )
        with st.expander(
            "네 Gate 적용값",
            expanded=customize_sensitivity,
        ):
            if customize_sensitivity:
                pattern_strictness = st.selectbox(
                    "형식 Gate — 패턴 엄격도",
                    tuple(PatternStrictness),
                    index=tuple(PatternStrictness).index(
                        saved_policy_config.pattern_strictness
                    ),
                    format_func=format_pattern_strictness,
                    key="advanced_pattern_strictness",
                )
                checksum_validation_enabled = st.checkbox(
                    "체크섬 검증 사용",
                    value=(
                        saved_policy_config.checksum_validation_enabled
                    ),
                    key="advanced_checksum_enabled",
                    help=(
                        "끄면 형식은 맞지만 체크섬이 틀린 사업자·법인·카드번호도 "
                        "저신뢰 후보로 만듭니다. 체크섬을 끄는 것은 탐지를 더 "
                        "민감하게 만드는 설정입니다."
                    ),
                )
                minimum_risk_level = st.selectbox(
                    "ko-pii 최소 위험도",
                    tuple(RISK_LEVEL_NAMES),
                    index=tuple(RISK_LEVEL_NAMES).index(
                        saved_policy_config.minimum_risk_level
                    ),
                    format_func=format_risk_level,
                    key="advanced_minimum_risk",
                )
                checksum_invalid_action = st.selectbox(
                    "체크섬 불일치 후보",
                    tuple(GateAction),
                    index=tuple(GateAction).index(
                        saved_policy_config.checksum_invalid_action
                    ),
                    format_func=format_gate_action,
                    disabled=not checksum_validation_enabled,
                    key="advanced_checksum_action",
                )
                minimum_context_evidence_count = st.slider(
                    "필드명·키워드·문맥 Gate 최소 통과 수",
                    min_value=0,
                    max_value=3,
                    value=(
                        saved_policy_config.minimum_context_evidence_count
                    ),
                    key="advanced_context_count",
                    help=(
                        "0이면 주변 근거를 요구하지 않습니다. 유효한 체크섬이 있는 "
                        "정형 번호는 이 조건을 통과한 것으로 처리합니다."
                    ),
                )
                missing_context_action = st.selectbox(
                    "문맥 Gate 근거가 부족할 때",
                    tuple(GateAction),
                    index=tuple(GateAction).index(
                        saved_policy_config.missing_context_action
                    ),
                    format_func=format_gate_action,
                    disabled=minimum_context_evidence_count == 0,
                    key="advanced_context_action",
                )
                anchor_required_categories = st.multiselect(
                    "추가로 anchor를 필수 요구할 유형",
                    category_options,
                    default=list(
                        saved_policy_config.anchor_required_categories
                    ),
                    format_func=format_category,
                    key="advanced_anchor_categories",
                )
                missing_anchor_action = st.selectbox(
                    "필수 anchor가 없을 때",
                    tuple(GateAction),
                    index=tuple(GateAction).index(
                        saved_policy_config.missing_anchor_action
                    ),
                    format_func=format_gate_action,
                    disabled=not anchor_required_categories,
                    key="advanced_missing_anchor_action",
                )
                review_handling = st.selectbox(
                    "검수 필요 후보 처리",
                    tuple(ReviewHandling),
                    index=tuple(ReviewHandling).index(
                        saved_policy_config.review_handling
                    ),
                    format_func=format_review_handling,
                    key="advanced_review_handling",
                )
            else:
                pattern_strictness = (
                    sensitivity_preset.pattern_strictness
                )
                checksum_validation_enabled = (
                    sensitivity_preset.checksum_validation_enabled
                )
                minimum_risk_level = (
                    sensitivity_preset.minimum_risk_level
                )
                checksum_invalid_action = (
                    sensitivity_preset.checksum_invalid_action
                )
                minimum_context_evidence_count = (
                    sensitivity_preset.minimum_context_evidence_count
                )
                missing_context_action = (
                    sensitivity_preset.missing_context_action
                )
                anchor_required_categories = ()
                missing_anchor_action = missing_context_action
                review_handling = sensitivity_preset.review_handling
                st.markdown(
                    f"- **형식 Gate:** "
                    f"{PATTERN_STRICTNESS_NAMES[pattern_strictness]}\n"
                    f"- **체크섬 Gate:** "
                    f"{'사용' if checksum_validation_enabled else '사용 안 함'}\n"
                    f"- **필드명·키워드·문맥 Gate:** 최소 "
                    f"{minimum_context_evidence_count}개 근거\n"
                    f"- **근거 부족:** "
                    f"{GATE_ACTION_NAMES[missing_context_action.value]}\n"
                    f"- **검수 후보:** "
                    f"{REVIEW_HANDLING_NAMES[review_handling.value]}"
                )

    # 4. 위젯 값을 엔진에서 사용하는 정책 객체로 조립한다.
    aegis_scope = (
        "semantic"
        if set(selected_aegis_entities) <= set(SEMANTIC_AEGIS_ENTITY_TYPES)
        else "all"
    )
    policy_config = MaskingPolicyConfig(
        execution_mode=execution_mode,
        merge_mode=merge_mode,
        selected_categories=frozenset(selected_categories),
        sensitivity_level=sensitivity_level,
        pattern_strictness=pattern_strictness,
        aegis_scope=aegis_scope,
        aegis_entity_types=tuple(selected_aegis_entities),
        aegis_threshold_adjustment=aegis_threshold_adjustment,
        minimum_confidence=minimum_confidence,
        minimum_risk_level=minimum_risk_level,
        checksum_validation_enabled=checksum_validation_enabled,
        checksum_invalid_action=checksum_invalid_action,
        minimum_context_evidence_count=(
            minimum_context_evidence_count
        ),
        missing_context_action=missing_context_action,
        anchor_required_categories=frozenset(
            anchor_required_categories
        ),
        missing_anchor_action=missing_anchor_action,
        review_handling=review_handling,
        mask_padding=saved_policy_config.mask_padding,
        max_adjacent_lines=saved_policy_config.max_adjacent_lines,
        custom_person_field_labels=(
            saved_policy_config.custom_person_field_labels
        ),
        custom_address_field_labels=(
            saved_policy_config.custom_address_field_labels
        ),
    )

    with st.sidebar:
        with st.expander("현재 적용값 요약"):
            st.markdown(_policy_summary(policy_config))

    ocr_document, processed = items[selected_document_index]
    # SearchView 생성에 필요한 설정만 별도 객체로 전달한다.
    search_settings = SearchViewSettings(
        max_adjacent_lines=policy_config.max_adjacent_lines,
        pattern_strictness=policy_config.pattern_strictness,
        checksum_validation_enabled=(
            policy_config.checksum_validation_enabled
        ),
    )
    detector_policy = policy_config.to_detection_policy()
    policy_error: str | None = None
    ner_runtime: AegisOnnxNer | None = None
    # 5. 실제 엔진을 실행해 규칙 결과, AEGIS 결과와 최종 결과를 한 번에 받는다.
    # Ctrl+클릭: pipeline.detect_document_pii
    try:
        with st.spinner("현재 화면 설정으로 개인정보를 탐지하는 중입니다..."):
            if policy_config.execution_mode != DetectorExecutionMode.RULE_ONLY:
                ner_runtime = load_aegis_runtime(str(model_dir))
            policy_result = detect_document_pii(
                processed,
                policy_config,
                search_settings,
                ner_runtime,
            )
        rule_spans = policy_result.rule_spans
        ner_spans = policy_result.ner_spans
        final_spans = policy_result.final_spans
    except Exception as exc:
        policy_error = str(exc)
        rule_spans = detect_all_pii_spans(
            processed,
            search_settings=search_settings,
        )
        ner_spans = ()
        final_spans = combine_policy_spans(
            rule_spans,
            (),
            detector_policy,
        )
        policy_result = None

    # 6. 아래 trace와 SearchView는 엔진 결과를 이해하기 위한 화면 표시 자료다.
    low_confidence_tokens = tuple(
        token
        for token in processed.index.tokens
        if token.confidence < 0.8
    )
    search_views = build_document_search_views(
        processed,
        search_settings=search_settings,
    )
    aegis_prediction_traces = ()
    if (
        policy_result is not None
        and policy_result.ner_executed
        and ner_runtime is not None
    ):
        aegis_prediction_traces = trace_aegis_predictions(
            search_views,
            scope=policy_config.aegis_scope,
            threshold_adjustment=policy_config.aegis_threshold_adjustment,
            enabled_entity_types=policy_config.aegis_entity_types,
            ner=ner_runtime,
        )
    source_path = Path(ocr_document.image_path)

    st.subheader(source_path.name)
    if policy_error:
        st.warning(
            "AEGIS 실행에 실패하여 ko-pii·자체 규칙 결과로 표시합니다. "
            f"오류: {policy_error}"
        )
    elif policy_result is not None:
        st.info(
            # f"{_policy_summary(policy_config)}  \n"
            f"**규칙 실행 시간:** {policy_result.rule_seconds * 1000:.1f}ms  \n"
            f"**AEGIS 실행:** {'예' if policy_result.ner_executed else '아니오'} "
            f"({policy_result.ner_seconds * 1000:.1f}ms)  \n"
            f"**AEGIS 판단:** {policy_result.ner_decision}"
        )

    metrics = st.columns(6)
    metrics[0].metric("OCR token", len(processed.index.tokens))
    metrics[1].metric("재구성 line", len(processed.lines))
    metrics[2].metric("낮은 OCR 신뢰도", len(low_confidence_tokens))
    metrics[3].metric("규칙 span", len(rule_spans))
    metrics[4].metric("AEGIS span", len(ner_spans))
    metrics[5].metric("최종 span", len(final_spans))

    rule_traces = trace_ko_pii(
        processed,
        rule_spans,
        search_settings=search_settings,
    )

    # 7. 한 번 계산한 결과를 단계별 탭에 나누어 표시한다.
    (
        ocr_tab,
        preprocessing_tab,
        detection_tab,
        mapping_tab,
        masking_tab,
    ) = st.tabs(
        [
            "1. OCR",
            "2. 전처리·SearchView",
            "3. 개인정보 탐지",
            "4. span·bbox",
            "5. 최종 마스킹",
        ]
    )

    with ocr_tab:
        left, right = st.columns([1.2, 1])
        left.image(
            source_path,
            caption="원본 이미지",
            width="stretch",
        )
        with right:
            st.markdown("**OCR 서버 full_text — 참고용**")
            st.text_area(
                "full_text",
                ocr_document.full_text,
                height=260,
                label_visibility="collapsed",
            )
            if low_confidence_tokens:
                st.warning(
                    "신뢰도 0.8 미만 OCR token이 "
                    f"{len(low_confidence_tokens)}개입니다."
                )
            st.dataframe(
                token_rows(processed),
                width="stretch",
                height=340,
                hide_index=True,
            )

    with preprocessing_tab:
        st.image(
            render_token_overlay(processed),
            caption="파란색: OCR token bbox / L숫자: 재구성 line",
            width="stretch",
        )
        st.dataframe(
            [
                {
                    "line": line.line_id,
                    "token IDs": ", ".join(map(str, line.token_ids)),
                    "줄 문자열": line.text,
                    "공백 제거 참고값": line.compact_text,
                }
                for line in processed.lines
            ],
            width="stretch",
            hide_index=True,
        )
        st.markdown("**실제 탐지기에 전달되는 SearchView**")
        st.dataframe(
            [
                {   
                    "개수": idx+1,
                    "lines": ", ".join(
                        str(line.line_id) for line in view.lines
                    ),
                    "검색 방식": view.mode,
                    "탐지 입력": view.text.replace("\n", " ↵ "),
                    "허용 유형": (
                        "전체"
                        if view.allowed_categories is None
                        else ", ".join(sorted(view.allowed_categories))
                    ),
                }
                for idx, view in enumerate(search_views)
            ],
            width="stretch",
            height=440,
            hide_index=True,
        )

    with detection_tab:
        st.caption(
            "ko-pii·자체 규칙과 AEGIS를 현재 정책에 따라 실행한 결과입니다. "
            "Presidio와 추가 실험 모델은 기본 화면에서 제외했습니다."
        )
        rule_tab, aegis_tab, final_tab = st.tabs(
            ["ko-pii·자체 규칙", "AEGIS", "최종 결과"]
        )
        with rule_tab:
            with st.expander("ko-pii 신뢰도 기준", expanded=False):
                st.markdown(
                    """
`ko-pii`에는 모든 개인정보 유형이 공유하는 신뢰도 계산식이 없습니다.
표시되는 값은 각 탐지기가 조건별로 정해 둔 **규칙 점수**이며, AI 모델의
확률이나 실제 정확도를 의미하지 않습니다.

| 탐지 유형 | 규칙이 확인한 조건 | 반환 신뢰도 |
|---|---|---:|
| 주민·외국인등록번호 | 날짜·성별 코드·체크섬 통과 | `1.0` |
| 주민·외국인등록번호 | 날짜·성별 코드는 유효하지만 체크섬 불일치 | `0.7` |
| 사업자·법인등록번호, 카드번호 | 정해진 형식과 체크섬 통과 | `1.0` |
| 이메일·전화번호·IP 주소 | 각 탐지기의 전체 형식 통과 | `1.0` |
| 여권·계좌·건강보험번호·URL | 각 유형의 형식 또는 필수 키워드 통과 | `0.9` |
| 운전면허·팩스·차량번호 | 각 유형의 형식 통과 | `0.85` |
| 생년월일 | 사용된 날짜 표현에 따라 | `0.95`, `0.9`, `0.85` |
| 도로명 주소 | 주소 사전·행정구역·도로명 조건 통과 | `0.8` |
| 지번 주소 | 주소 사전·행정구역·지번 조건 통과 | `0.75` |
| 문맥형 주소 / 단독 행정구역 | 주소 키워드와 사전 조건 통과 | `0.6` / `0.7` |
| 이름 | 필드명·성씨·직책·음절 등의 점수를 합산 | 후보마다 다름 |

체크섬이 필수인 사업자·법인등록번호와 카드번호는 체크섬에 실패하면 원래
`ko-pii` 결과로 나오지 않습니다. 체크섬 검사를 끈 정책에서 보이는 `0.65`
후보와 관리자 이름·주소 필드의 `0.85`는 이 프로젝트의 자체 보완 점수입니다.
`table_field`는 점수를 만들지 않고 같은 열의 값을 ko-pii 또는 AEGIS에 전달합니다.

기본 `minimum_confidence`는 `0.0`이므로 공통 점수 경계로 결과를 제외하지
않습니다. 값을 올리면 ko-pii 규칙 점수와 AEGIS 모델 confidence에 모두 적용되지만
두 값의 의미가 다르므로 주의해야 합니다. 최종 마스킹·검수·제외 여부는 유형,
체크섬, 문맥 근거와 관리자 정책을 함께 확인해 결정합니다. OCR 신뢰도는 OCR
서버가 반환한 별개의 값입니다.
                    """
                )
            if rule_traces:
                st.dataframe(
                    [
                        {
                            "lines": ", ".join(map(str, trace.line_ids)),
                            "검색 방식": trace.view_mode,
                            "검색 입력": trace.search_text.replace(
                                "\n",
                                " ↵ ",
                            ),
                            "유형": CATEGORY_NAMES.get(
                                trace.category,
                                trace.category,
                            ),
                            "탐지 문자열": trace.detected_text,
                            "신뢰도": round(trace.confidence, 3),
                            "token IDs": ", ".join(
                                map(str, trace.token_ids)
                            ),
                            "규칙": trace.rule_id,
                            "처리 결과": trace.outcome,
                        }
                        for trace in rule_traces
                    ],
                    width="stretch",
                    height=500,
                    hide_index=True,
                )
            else:
                st.warning("규칙 탐지 결과가 없습니다.")
        with aegis_tab:
            raw_prediction_tab, mapped_span_tab = st.tabs(
                ["모델 원시 예측·threshold", "span·bbox 매핑 결과"]
            )
            with raw_prediction_tab:
                if aegis_prediction_traces:
                    passed_count = sum(
                        1 for trace in aegis_prediction_traces
                        if trace.passed
                    )
                    excluded_count = (
                        len(aegis_prediction_traces) - passed_count
                    )
                    show_excluded_only = st.checkbox(
                        "threshold 제외 예측만 보기",
                        value=True,
                        help=(
                            "해제하면 threshold를 통과한 원시 token 예측도 "
                            "함께 표시합니다."
                        ),
                    )
                    visible_prediction_traces = tuple(
                        trace
                        for trace in aegis_prediction_traces
                        if not show_excluded_only or not trace.passed
                    )
                    st.caption(
                        "AEGIS가 token마다 예측한 원시 entity입니다. "
                        f"threshold 통과 {passed_count}개 / 제외 {excluded_count}개"
                    )
                    if visible_prediction_traces:
                        st.dataframe(
                            [
                                {
                                    "lines": ", ".join(
                                        map(str, trace.line_ids)
                                    ),
                                    "실제 입력": trace.input_text.replace(
                                        "\n",
                                        " ↵ ",
                                    ),
                                    "예측 문자열": trace.text,
                                    "BIO": trace.bio,
                                    "AEGIS 라벨": trace.entity_type,
                                    "내부 유형": CATEGORY_NAMES.get(
                                        trace.category,
                                        trace.category,
                                    ),
                                    "confidence": round(
                                        trace.confidence,
                                        4,
                                    ),
                                    "threshold": round(
                                        trace.threshold,
                                        4,
                                    ),
                                    "판정": (
                                        "통과"
                                        if trace.passed
                                        else "threshold 제외"
                                    ),
                                    "token IDs": ", ".join(
                                        map(str, trace.token_ids)
                                    ),
                                }
                                for trace in visible_prediction_traces
                            ],
                            width="stretch",
                            height=500,
                            hide_index=True,
                        )
                    else:
                        st.info("현재 설정에서 threshold로 제외된 예측이 없습니다.")
                elif policy_result and not policy_result.ner_executed:
                    st.info(
                        "현재 실행 정책에 따라 AEGIS를 실행하지 않았습니다."
                    )
                else:
                    st.warning("선택한 AEGIS 라벨의 원시 예측이 없습니다.")
            with mapped_span_tab:
                if ner_spans:
                    st.dataframe(
                        span_rows(ner_spans),
                        width="stretch",
                        height=500,
                        hide_index=True,
                    )
                elif policy_result and not policy_result.ner_executed:
                    st.info(
                        "현재 실행 정책에 따라 AEGIS를 실행하지 않았습니다."
                    )
                else:
                    st.warning("AEGIS span·bbox 매핑 결과가 없습니다.")
        with final_tab:
            st.write(_category_summary(final_spans))
            if policy_result and policy_result.gate_decisions:
                with st.expander("Gate 판정 근거", expanded=True):
                    st.dataframe(
                        _gate_decision_rows(
                            policy_result.gate_decisions
                        ),
                        width="stretch",
                        height=420,
                        hide_index=True,
                    )
            st.dataframe(
                span_rows(final_spans),
                width="stretch",
                height=500,
                hide_index=True,
            )

    with mapping_tab:
        if final_spans:
            span_options = {}
            for index, span in enumerate(final_spans):
                label = f"{index + 1}. {format_span(span)}"
                span_options[label] = index

            selected_span_label = st.selectbox(
                "강조할 개인정보",
                tuple(span_options),
            )
            # 문서나 정책을 바꾼 직후에는 Streamlit이 이전 선택값을 한 번 더
            # 반환할 수 있다. 현재 목록에 없는 값이면 첫 번째 span을 선택한다.
            selected_span_index = span_options.get(selected_span_label, 0)
            st.image(
                render_span_overlay(
                    processed,
                    final_spans,
                    selected_index=selected_span_index,
                    padding=policy_config.mask_padding,
                ),
                caption="번호는 아래 span 표의 순서와 같습니다.",
                width="stretch",
            )
        else:
            st.warning("표시할 최종 span이 없습니다.")
        st.dataframe(
            span_rows(final_spans),
            width="stretch",
            hide_index=True,
        )

    with masking_tab:
        masked_image, span_count, box_count = render_masked_image(
            processed,
            final_spans,
            padding=policy_config.mask_padding,
        )
        left, right = st.columns(2)
        left.image(
            source_path,
            caption="원본",
            width=MASK_PREVIEW_WIDTH,
        )
        right.image(
            masked_image,
            caption=f"현재 정책 결과 — span {span_count} / box {box_count}",
            width=MASK_PREVIEW_WIDTH,
        )
        left.download_button(
            "원본 다운로드",
            data=source_path.read_bytes(),
            file_name=source_path.name,
            mime=(
                "image/png"
                if source_path.suffix.lower() == ".png"
                else "image/jpeg"
            ),
            width="stretch",
        )
        right.download_button(
            "마스킹 결과 PNG 다운로드",
            data=_image_to_png_bytes(masked_image),
            file_name=f"{source_path.stem}_masked.png",
            mime="image/png",
            width="stretch",
        )

    st.divider()
    previous_col, position_col, next_col = st.columns([1, 2, 1])
    previous_col.button(
        "← 이전 문서",
        disabled=selected_document_index == 0,
        on_click=_move_document,
        args=(-1, len(items)),
        width="stretch",
    )
    position_col.markdown(
        "<div style='text-align:center; padding-top:0.45rem'>"
        f"<strong>{selected_document_index + 1} / {len(items)}</strong><br>"
        f"<small>{source_path.name}</small></div>",
        unsafe_allow_html=True,
    )
    next_col.button(
        "다음 문서 →",
        disabled=selected_document_index == len(items) - 1,
        on_click=_move_document,
        args=(1, len(items)),
        width="stretch",
    )


if __name__ == "__main__":
    main()
