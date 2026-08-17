"""SearchView로 ko-pii를 실행하고 결과를 공통 탐지 형식으로 변환한다.

이 파일에는 성격이 다른 세 경로가 함께 있다.

1. **ko-pii 공식 탐지 경로**
   SearchView 문자열을 ko-pii ``detect_all()``에 전달한다. 정규식, 사전, 체크섬,
   문맥 검증과 ko-pii 내부 겹침 해소는 ko-pii 소스가 수행한다.
2. **OCR 입력 보정 경로**
   OCR이 주소의 쉼표나 이름 앞의 점을 잘못 붙인 경우 해당 문자만 공백으로 바꾼
   복사본을 ko-pii에 전달한다. 원본 위치로 돌아가야 하므로 문자열 길이는 유지한다.
3. **프로젝트 보완 경로**
   체크섬 검사를 끈 정책의 제한적 번호 복구, 주소 필드·계좌 예금주 구조와 관리자
   추가 이름 필드를 처리한다. 구조·필드 탐지 결과는 ``context:`` rule_id로
   구분한다. 체크섬 생략 번호는 원래 라벨을 유지하고 ``checksum:skipped``
   evidence로 정상 ko-pii 결과와 구분한다.

코드 검토 순서
1. ``KoPiiCategorySpec``에서 내부 카테고리와 실행할 ko-pii 라벨을 확인한다.
2. ``KoPiiDetectionProvider``가 SearchView 문자열을 ko-pii ``detect_all()``에
   전달하고 반환된 ``DetectionResult``를 공통 ``ProviderMatch``로 바꾼다.
3. ``detect_ko_pii_spans()``가 ``detector_engine.detect_with_provider()``를 통해
   ko-pii 문자 위치를 OCR token ID와 bbox로 역매핑한다.
4. 체크섬 검사를 끈 경우의 제한적 형식 복구는 ``_relaxed_checksum_results()``와
   ``_recover_field_anchored_native_results()``에서 확인한다.

일반 탐지는 ko-pii 정규식을 그대로 사용한다. 다만 체크섬을 끈 특별한 정책에서만
ko-pii가 반환하지 않은 후보를 보존하기 위한 제한된 형식 패턴을 이 파일이 가진다.
최종 목적은 ko-pii의 라벨, 문자 범위, confidence, risk와 evidence를 현재 OCR
파이프라인의 ``ProviderMatch``와 ``PiiSpan``으로 번역하는 것이다.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
import re

from ko_pii import detect_all
from ko_pii.core.types import DetectionResult, RiskLevel
from ko_pii.patterns import address, person

from .structure_rules import (
    detect_account_block,
    detect_account_owner,
    detect_address_field,
    detect_custom_person_field,
)
from .detector_engine import PiiSpan, ProviderMatch, detect_with_provider
from .models import PreprocessedDocument
from .search_views import (
    DEFAULT_SEARCH_VIEW_SETTINGS,
    SearchView,
    SearchViewSettings,
)


# ko-pii의 개별 탐지 함수는 문자열을 받고 DetectionResult를 여러 개 반환한다.
# address.detect와 person.detect를 같은 타입으로 다루기 위한 타입 별칭이다.
KoPiiDetector = Callable[[str], Iterable[DetectionResult]]


# ---------------------------------------------------------------------------
# 1. ko-pii에 전달하기 전 적용하는 OCR 문자 보정
# ---------------------------------------------------------------------------
# ko-pii의 주소 상세 확장은 건물번호 직후에 공백과 동/호/층이 올 때 동작한다.
# OCR이 ``356, 13층``처럼 쉼표를 붙이면 상세 확장이 끊기므로, 주소 탐지 입력에서만
# 쉼표를 같은 길이의 공백으로 바꾼다. 글자 수를 유지해야 기존 OCR 좌표 매핑이
# 그대로 유효하다.
_ADDRESS_ROAD_NUMBER_SEPARATOR = re.compile(
    r"[가-힣A-Za-z0-9]{1,16}(?:대로|로|길)"
    r"(?P<separator>\s*[.,·ㆍ:;]\s*)(?=\d)"
)
_ADDRESS_NUMBER_DETAIL_SEPARATOR = re.compile(
    r"\d+(?:-\d+)?"
    r"(?P<separator>\s*[.,·ㆍ:;]\s*)"
    r"(?=\d+(?:동|호|층)(?:\s|\(|$))"
)
_ADDRESS_OCR_PAREN_TAIL = re.compile(
    # 건물번호 뒤에는 2층, B동 709호, 101동 1203호처럼 건물 단위가
    # 하나 이상 이어질 수 있다. \s에는 SearchView의 줄바꿈도 포함되므로
    # 괄호 내용이 바로 다음 OCR 줄로 이어진 경우도 같은 규칙으로 처리한다.
    r"\s*[,.;·ㆍ]?\s*"
    r"(?:[가-힣A-Za-z0-9]{1,8}(?:동|층|호)\s*){0,3}"
    r"\((?P<detail>[가-힣A-Za-z0-9,·ㆍ\s]+)\)"
)
_ADDRESS_DETAIL_MARKERS = (
    "동", "읍", "면", "리", "가", "빌딩", "타워", "센터", "아파트",
    "오피스텔", "상가", "플라자", "스퀘어",
)

# 도장이나 표 선을 OCR이 점으로 읽어 ``대표이사 .형형근``처럼 만드는 사례를
# 이름 탐지 입력에서만 보정한다. 강한 직책/필드 문맥 뒤의 점 하나만 바꾸므로
# 일반 문장의 마침표에는 영향을 주지 않는다.
_PERSON_PUNCTUATION_SEPARATOR = re.compile(
    r"(?:대표이사|대표자|성명|이름|신청인|신청자|민원인|청구인|보호자|대리인)"
    r"\s*(?P<separator>[.·ㆍ])\s*(?=[가-힣]{2,4}(?:\s|$))"
)


def prepare_ko_pii_input(detector: KoPiiDetector, text: str) -> str:
    """탐지 종류에 필요한 OCR 문자 보정만 적용한 같은 길이 문자열을 반환한다.

    ``detector``가 ``address.detect``이면 주소 내부 구두점을 공백으로 바꾼다.
    ``person.detect``이면 강한 이름 필드 뒤의 점 하나를 공백으로 바꾼다. 다른
    탐지기에는 입력을 그대로 반환한다.

    문자를 삭제하거나 추가하지 않는 이유는 ko-pii가 반환하는 start/end가 원래
    SearchView의 문자 위치와 동일해야 OCR token과 bbox로 역매핑할 수 있기 때문이다.
    """
    if detector is address.detect:
        chars = list(text)
        for pattern in (
            _ADDRESS_ROAD_NUMBER_SEPARATOR,
            _ADDRESS_NUMBER_DETAIL_SEPARATOR,
        ):
            snapshot = "".join(chars)
            for match in pattern.finditer(snapshot):
                start, end = match.span("separator")
                for index in range(start, end):
                    if not chars[index].isspace():
                        chars[index] = " "
        return "".join(chars)

    if detector is person.detect:
        chars = list(text)
        for match in _PERSON_PUNCTUATION_SEPARATOR.finditer(text):
            chars[match.start("separator")] = " "
        return "".join(chars)

    return text


def _cross_line_tail_is_aligned(
    view: SearchView,
    result: DetectionResult,
    match: re.Match[str],
) -> bool:
    """다음 줄의 괄호 내용이 앞 주소와 같은 열에서 시작하는지 확인한다.

    괄호가 같은 줄에 있으면 좌표 검증 없이 True다. 줄바꿈 뒤에 있다면 주소의 첫
    token과 후속 줄의 첫 token을 찾아 x좌표 차이가 글자 높이 3배 이내인지 확인한다.
    고정 pixel이 아니라 글자 높이를 사용하므로 이미지 해상도가 달라도 적용된다.
    """
    line_break = view.text.find("\n", result.end, match.end())
    if line_break < 0:
        return True

    open_parenthesis = view.text.find("(", result.end, match.end())
    if open_parenthesis < 0 or open_parenthesis > line_break:
        return False

    address_token_ids = view.token_ids_for_span(result.start, result.end)
    continuation_token_ids = view.token_ids_for_span(line_break + 1, match.end())
    if not address_token_ids or not continuation_token_ids:
        return False

    tokens = {
        token.token_id: token
        for line in view.lines
        for token in line.tokens
    }
    address_start = tokens[address_token_ids[0]]
    continuation_start = tokens[continuation_token_ids[0]]
    token_height = max(
        1,
        address_start.box.y2 - address_start.box.y1,
        continuation_start.box.y2 - continuation_start.box.y1,
    )
    # 해상도별 고정 pixel 값 대신 글자 높이를 기준으로 들여쓰기 허용 범위를 정한다.
    return abs(continuation_start.box.x1 - address_start.box.x1) <= token_height * 3


def _extend_address_ocr_tail(
    result: DetectionResult,
    text: str,
    view: SearchView,
) -> DetectionResult:
    """ko-pii 주소 바로 뒤의 괄호가 주소 상세정보일 때 DetectionResult를 확장한다.

    다음 세 조건을 모두 만족해야 한다.
    1. ko-pii가 찾은 주소 직후가 ``(상현동, 아파트명)`` 형태다.
    2. 괄호 안에 동·읍·면·리·빌딩·타워·센터·아파트 등의 주소 표지가 있다.
    3. 다음 OCR 줄까지 이어졌다면 앞 주소와 x좌표가 정렬되어 있다.

    조건을 만족하지 않으면 원래 ko-pii 결과를 그대로 반환한다.
    """
    match = _ADDRESS_OCR_PAREN_TAIL.match(text, result.end)
    if match is None:
        return result
    compact_detail = re.sub(r"\s", "", match.group("detail"))
    if not any(marker in compact_detail for marker in _ADDRESS_DETAIL_MARKERS):
        return result
    if not _cross_line_tail_is_aligned(view, result, match):
        return result
    return replace(
        result,
        text=text[result.start:match.end()].strip(),
        end=match.end(),
    )


# ---------------------------------------------------------------------------
# 2. ko-pii와 프로젝트 사이에서 사용하는 자료구조
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KoPiiCategorySpec:
    """내부 카테고리 하나와 실행할 ko-pii 공식 라벨 목록의 대응표.

    예를 들어 내부 ``phone_number``는 ko-pii의 ``PHONE``과 ``FAX`` 두 라벨을
    함께 실행한다. labels가 비어 있으면 현재 ko-pii가 지원하지 않는 카테고리다.
    """

    category: str
    labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdapterDetection:
    """어댑터가 반환하는 ko-pii 또는 프로젝트 문맥 탐지 결과 한 건.

    ``result``는 문자 start/end와 ko-pii 형식의 근거를 가진다. ``category``는
    프로젝트 내부 유형이고 ``rule_id``는 결과 출처가 ``ko_pii:``인지
    ``context:``인지 구분한다. ``input_text``는 실제 탐지기에 전달된 문자열이다.
    """

    category: str
    result: DetectionResult
    rule_id: str
    input_text: str


@dataclass(frozen=True)
class KoPiiDetectionProvider:
    """SearchView 하나에서 ko-pii를 실행하고 공통 ProviderMatch를 반환한다.

    detector_engine은 ko-pii의 DetectionResult 형식을 알지 못한다. 따라서 이
    provider가 category, start/end, confidence, risk_level과 evidence를
    ProviderMatch에 옮긴다. 이후 bbox 역매핑은 detector_engine이 담당한다.
    """

    specs: tuple[KoPiiCategorySpec, ...]
    search_settings: SearchViewSettings

    def __call__(self, view: SearchView) -> tuple[ProviderMatch, ...]:
        """SearchView 한 개의 모든 허용 결과를 ProviderMatch로 변환한다."""
        matches: list[ProviderMatch] = []
        for detection in iter_adapter_detections(
            view,
            self.specs,
            search_settings=self.search_settings,
        ):
            result = detection.result
            matches.append(ProviderMatch(
                category=detection.category,
                start=result.start,
                end=result.end,
                text=result.text,
                confidence=float(result.confidence),
                rule_id=detection.rule_id,
                risk_level=int(result.risk_level),
                evidence=tuple(result.evidence),
                legal_basis=result.legal_basis,
            ))
        return tuple(matches)


# ---------------------------------------------------------------------------
# 3. 내부 개인정보 카테고리와 ko-pii 라벨 대응
# ---------------------------------------------------------------------------
# 왼쪽 key는 관리자 정책과 PiiSpan에서 사용하는 내부 카테고리다.
# tuple 안의 값은 ko-pii detect_all(include=...)에 전달하는 공식 라벨이다.
KO_PII_CATEGORY_SPECS: dict[str, KoPiiCategorySpec] = {
    "person_name": KoPiiCategorySpec("person_name", ("PERSON",)),
    "address": KoPiiCategorySpec("address", ("ADDRESS",)),
    "email_address": KoPiiCategorySpec("email_address", ("EMAIL",)),
    "phone_number": KoPiiCategorySpec("phone_number", ("PHONE", "FAX")),
    "url": KoPiiCategorySpec("url", ("URL",)),
    "date": KoPiiCategorySpec("date", ("DT_BIRTH",)),
    "account_card_number": KoPiiCategorySpec(
        "account_card_number",
        ("ACCOUNT", "CARD"),
    ),
    # ko-pii에는 API 키·토큰·비밀번호 통합 라벨이 없으므로 현재는 빈 설정이다.
    "credential_secret": KoPiiCategorySpec("credential_secret"),
    "resident_registration_number": KoPiiCategorySpec(
        "resident_registration_number",
        ("RRN",),
    ),
    "foreigner_registration_number": KoPiiCategorySpec(
        "foreigner_registration_number",
        ("FRN",),
    ),
    "passport_number": KoPiiCategorySpec("passport_number", ("PASSPORT",)),
    "driver_license_number": KoPiiCategorySpec(
        "driver_license_number",
        ("DRIVER_LICENSE",),
    ),
    "business_registration_number": KoPiiCategorySpec(
        "business_registration_number",
        ("BUSINESS_REG",),
    ),
    "corporate_registration_number": KoPiiCategorySpec(
        "corporate_registration_number",
        ("CORP_REG",),
    ),
    "health_insurance_number": KoPiiCategorySpec(
        "health_insurance_number",
        ("MEDICAL_INSURANCE",),
    ),
    "vehicle_plate_number": KoPiiCategorySpec(
        "vehicle_plate_number",
        ("VEHICLE",),
    ),
    "ip_address": KoPiiCategorySpec("ip_address", ("IP",)),
    "land_lot_number": KoPiiCategorySpec("land_lot_number", ("PNU",)),
    "prescription_number": KoPiiCategorySpec(
        "prescription_number",
        ("PRESCRIPTION_ID",),
    ),
    "drug_code": KoPiiCategorySpec("drug_code", ("EDI_DRUG",)),
    "employee_number": KoPiiCategorySpec("employee_number", ("EMPLOYEE_ID",)),
    "petition_number": KoPiiCategorySpec("petition_number", ("PETITION_ID",)),
    "court_case_number": KoPiiCategorySpec(
        "court_case_number",
        ("COURT_CASE",),
    ),
    "postal_code": KoPiiCategorySpec("postal_code", ("POSTAL_CODE",)),
    "document_number": KoPiiCategorySpec("document_number", ("DOC_ID",)),
    "nationality": KoPiiCategorySpec("nationality", ("NATIONALITY",)),
    "education_history": KoPiiCategorySpec(
        "education_history",
        ("EDUCATION",),
    ),
    "academic_major": KoPiiCategorySpec("academic_major", ("MAJOR",)),
    "job_position": KoPiiCategorySpec("job_position", ("POSITION",)),
    "age": KoPiiCategorySpec("age", ("AGE",)),
    "height": KoPiiCategorySpec("height", ("HEIGHT",)),
    "weight": KoPiiCategorySpec("weight", ("WEIGHT",)),
}

# ko-pii 결과를 받을 때 반대 방향으로 조회하기 위한 표다.
# 예: ``BUSINESS_REG`` → ``business_registration_number``
KO_PII_LABEL_TO_CATEGORY = {
    label: spec.category
    for spec in KO_PII_CATEGORY_SPECS.values()
    for label in spec.labels
}

# ---------------------------------------------------------------------------
# 4. 체크섬 검사를 끈 정책에서만 사용하는 제한적 번호 형식
# ---------------------------------------------------------------------------
# 이 패턴들은 평상시 ko-pii 탐지를 대체하지 않는다. 관리자가 체크섬 검사를 끈
# 경우에만 _relaxed_checksum_results()에서 실행된다.
_RELAXED_BUSINESS_PATTERN = re.compile(
    r"(?<!\d)\d{3}-?\d{2}-?\d{5}(?!\d)"
)
_RELAXED_CORPORATE_PATTERN = re.compile(
    r"(?<!\d)\d{6}-?\d{7}(?!\d)"
)
_RELAXED_RRN_PATTERN = re.compile(
    r"(?<!\d)\d{6}-(?P<century_gender>[0-49])\d{6}(?!\d)"
)
_RELAXED_FRN_PATTERN = re.compile(
    r"(?<!\d)\d{6}-(?P<century_gender>[5-8])\d{6}(?!\d)"
)
_RELAXED_CARD_PATTERN = re.compile(
    r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"
)


def _nearby_prefix(text: str, start: int, labels: tuple[str, ...]) -> bool:
    """후보 앞 32자 안의 공백을 제거하고 지정한 필드명으로 끝나는지 확인한다.

    문서 전체에서 느슨한 숫자 패턴을 찾으면 오탐이 커지므로 사업자번호, 법인번호와
    카드번호 복구에는 바로 앞의 강한 필드명을 anchor로 요구한다.
    """
    prefix = re.sub(r"\s", "", text[max(0, start - 32):start])
    prefix = prefix.rstrip(":：-")
    return any(prefix.endswith(label) for label in labels)


def _append_relaxed_candidate(
    recovered: list[DetectionResult],
    existing: set[tuple[str, int, int]],
    label: str,
    match: re.Match[str],
    risk_level: RiskLevel,
    pattern_evidence: str,
    anchored: bool,
) -> None:
    """체크섬 생략 후보를 ko-pii DetectionResult 형식으로 한 번만 추가한다.

    ``existing``은 같은 라벨과 문자 범위를 ko-pii가 이미 반환했는지 확인한다.
    복구 결과에는 ``checksum:skipped`` 근거와 고정 confidence 0.65를 부여하여
    정상 체크섬을 통과한 고신뢰 결과와 구분한다.
    """
    key = (label, match.start(), match.end())
    if key in existing:
        return
    evidence = [pattern_evidence, "checksum:skipped"]
    if anchored:
        evidence.append("anchor:prefix")
    recovered.append(DetectionResult(
        label=label,
        text=match.group(0),
        start=match.start(),
        end=match.end(),
        risk_level=risk_level,
        confidence=0.65,
        evidence=evidence,
        legal_basis=None,
        extra={
            "checksum_validation_enabled": False,
            "digits": re.sub(r"\D", "", match.group(0)),
        },
    ))
    existing.add(key)


def _relaxed_checksum_results(
    text: str,
    include_labels: tuple[str, ...],
    native_results: list[DetectionResult],
) -> list[DetectionResult]:
    """체크섬 검증을 끈 경우 형식 후보를 저신뢰 결과로 보존한다.

    이 함수가 복구하는 유형과 조건은 다음과 같다.

    - BUSINESS_REG: ``3-2-5`` 자리 형식과 사업자번호 계열 필드명이 모두 필요하다.
    - RRN: ``6-7`` 자리 형식과 주민 구분 숫자 0~4 또는 9가 필요하다. 누락
      위험이 커서 필드명이 없어도 후보로 남기지만, 법인번호 필드 앞이면 제외한다.
    - FRN: ``6-7`` 자리 형식과 외국인 구분 숫자 5~8이 필요하다.
    - CORP_REG: ``6-7`` 자리 형식과 법인등록번호 필드명이 모두 필요하다.
    - CARD: 13~19자리 카드 형식과 카드번호 계열 필드명이 모두 필요하다.

    여권번호, 운전면허번호와 건강보험번호에는 이 프로젝트의 체크섬 생략 복구를
    추가하지 않았다. 결과 문자 길이를 변경하지 않으므로 OCR bbox 역매핑은 유지된다.
    """
    recovered = list(native_results)
    existing = {
        (result.label, result.start, result.end)
        for result in native_results
    }

    if "BUSINESS_REG" in include_labels:
        for match in _RELAXED_BUSINESS_PATTERN.finditer(text):
            anchored = _nearby_prefix(
                text,
                match.start(),
                ("사업자등록번호", "사업자번호", "등록번호"),
            )
            if not anchored:
                continue
            _append_relaxed_candidate(
                recovered,
                existing,
                label="BUSINESS_REG",
                match=match,
                risk_level=RiskLevel.HIGH,
                pattern_evidence="pattern:business_reg",
                anchored=True,
            )

    if "RRN" in include_labels:
        for match in _RELAXED_RRN_PATTERN.finditer(text):
            if _nearby_prefix(
                text,
                match.start(),
                ("법인등록번호", "법인번호"),
            ):
                continue
            # ko-pii는 존재하지 않거나 미래인 생년월일을 체크섬 판단 전에도
            # 제외할 수 있다. 가장 민감한 정책에서는 명확한 6-7 자리 모양을
            # 마스킹 후보로 보존한다. 실제 하이픈과 1~8의 구분 숫자를 요구해
            # 일반적인 13자리 숫자보다 범위를 좁힌다.
            _append_relaxed_candidate(
                recovered,
                existing,
                label="RRN",
                match=match,
                risk_level=RiskLevel.CRITICAL,
                pattern_evidence="pattern:rrn_shape",
                anchored=_nearby_prefix(
                    text,
                    match.start(),
                    (
                        "주민등록번호",
                        "주민번호",
                        "주민(법인)등록번호",
                    ),
                ),
            )

    if "FRN" in include_labels:
        for match in _RELAXED_FRN_PATTERN.finditer(text):
            if _nearby_prefix(text, match.start(), ("법인등록번호", "법인번호")):
                continue
            _append_relaxed_candidate(
                recovered,
                existing,
                label="FRN",
                match=match,
                risk_level=RiskLevel.CRITICAL,
                pattern_evidence="pattern:frn_shape",
                anchored=_nearby_prefix(
                    text,
                    match.start(),
                    ("외국인등록번호", "외국인번호"),
                ),
            )

    if "CORP_REG" in include_labels:
        for match in _RELAXED_CORPORATE_PATTERN.finditer(text):
            anchored = _nearby_prefix(
                text,
                match.start(),
                ("법인등록번호", "법인번호"),
            )
            if not anchored:
                continue
            _append_relaxed_candidate(
                recovered,
                existing,
                label="CORP_REG",
                match=match,
                risk_level=RiskLevel.MEDIUM,
                pattern_evidence="pattern:corp_reg",
                anchored=True,
            )

    if "CARD" in include_labels:
        for match in _RELAXED_CARD_PATTERN.finditer(text):
            anchored = _nearby_prefix(
                text,
                match.start(),
                ("카드번호", "신용카드번호", "체크카드번호"),
            )
            if not anchored:
                continue
            _append_relaxed_candidate(
                recovered,
                existing,
                label="CARD",
                match=match,
                risk_level=RiskLevel.HIGH,
                pattern_evidence="pattern:card",
                anchored=True,
            )
    return recovered


def _prepare_native_ko_pii_input(
    text: str,
    categories: frozenset[str],
) -> str:
    """활성 카테고리에 필요한 OCR 보정을 ko-pii 통합 입력에 적용한다.

    주소가 활성화되면 주소 구두점 보정, 이름이 활성화되면 이름 앞 점 보정을
    차례로 적용한다. 보정 문자열은 길이가 같기 때문에 detect_all의 start/end를
    원래 SearchView에 그대로 사용할 수 있다.
    """
    prepared = text
    if "address" in categories:
        prepared = prepare_ko_pii_input(address.detect, prepared)
    if "person_name" in categories:
        prepared = prepare_ko_pii_input(person.detect, prepared)
    return prepared


def _recover_field_anchored_native_results(
    text: str,
    include_labels: tuple[str, ...],
    results: list[DetectionResult],
) -> list[DetectionResult]:
    """강한 필드명으로 유형이 명확한 13자리 후보를 유형별로 다시 확인한다.

    주민·외국인·법인번호는 모두 6자리-7자리 모양이어서 ko-pii 통합 API의
    겹침 우선순위에 따라 다른 유형 결과가 먼저 남을 수 있다. 특정 유형 필드가
    바로 앞에 있으면 그 유형의 공식 탐지기를 다시 실행해 결과 병합 단계까지
    전달한다. 공식 검증을 통과한 결과만 되살리며 번호 검증을 완화하지 않는다.
    """
    recovered = list(results)
    existing = {(result.label, result.start, result.end) for result in results}
    anchored_labels = (
        ("RRN", ("주민등록번호", "주민번호")),
        ("FRN", ("외국인등록번호", "외국인번호")),
        ("CORP_REG", ("법인등록번호", "법인번호", "주민(법인)등록번호")),
    )
    for label, field_labels in anchored_labels:
        if label not in include_labels:
            continue
        for candidate in detect_all(text, include=(label,), normalize=True):
            key = (candidate.label, candidate.start, candidate.end)
            if key in existing:
                continue
            if not _nearby_prefix(text, candidate.start, field_labels):
                continue
            recovered.append(candidate)
            existing.add(key)
    return recovered


# ---------------------------------------------------------------------------
# 5. pii_detectors.py가 호출하는 공개 진입 함수
# ---------------------------------------------------------------------------
def detect_ko_pii_spans(
    document: PreprocessedDocument,
    spec: KoPiiCategorySpec,
    *,
    search_settings: SearchViewSettings = DEFAULT_SEARCH_VIEW_SETTINGS,
) -> tuple[PiiSpan, ...]:
    """내부 카테고리 하나를 탐지해 OCR 좌표가 포함된 PiiSpan으로 반환한다.

    카테고리별 ``detect_person_name()`` 같은 기존 호출부가 공통 다중 카테고리
    구현을 사용할 수 있도록 제공하는 단일 카테고리 진입 함수다.
    """
    return detect_ko_pii_categories(
        document,
        (spec,),
        search_settings=search_settings,
    )


def detect_ko_pii_categories(
    document: PreprocessedDocument,
    specs: tuple[KoPiiCategorySpec, ...],
    *,
    search_settings: SearchViewSettings = DEFAULT_SEARCH_VIEW_SETTINGS,
) -> tuple[PiiSpan, ...]:
    """여러 내부 카테고리를 한 번의 SearchView 순회로 탐지한다.

    labels가 비어 있는 미지원 카테고리를 제외하고 provider를 만든다.
    ``detect_with_provider()``가 문서의 한 줄·인접 줄 SearchView를 만들고 provider를
    반복 호출한 뒤, 반환된 문자 범위를 token ID와 bbox가 포함된 PiiSpan으로 바꾼다.
    """
    active_specs = tuple(spec for spec in specs if spec.labels)
    if not active_specs:
        return ()

    provider = KoPiiDetectionProvider(active_specs, search_settings)
    return detect_with_provider(document, provider, search_settings=search_settings)


def iter_adapter_detections(
    view: SearchView,
    specs: tuple[KoPiiCategorySpec, ...] | None = None,
    *,
    search_settings: SearchViewSettings = DEFAULT_SEARCH_VIEW_SETTINGS,
) -> Iterable[AdapterDetection]:
    """SearchView 하나에서 허용된 ko-pii·문맥 탐지를 실행한다.

    이 파일의 핵심 generator다. 실행 순서는 다음과 같다.

    1. 요청받은 카테고리와 SearchView의 ``allowed_categories`` 교집합을 구한다.
    2. 구조 전용 SearchView면 해당 프로젝트 문맥 탐지기만 실행한다.
    3. 일반 SearchView면 허용 카테고리의 ko-pii 라벨을 한 번의 ``detect_all()``에
       전달한다.
    4. 정책에서 체크섬 검사를 끈 경우에만 제한적 저신뢰 후보를 추가한다.
    5. ko-pii 라벨을 내부 카테고리로 바꾸고 AdapterDetection을 yield한다.
    6. 관리자가 추가한 이름 필드가 있으면 별도의 문맥 결과를 마지막에 추가한다.

    검수 화면의 raw trace와 실제 파이프라인이 모두 이 함수를 사용하므로 화면에서
    보이는 원시 후보와 실제 탐지 후보가 같은 로직을 거친다.
    """
    # 호출부가 특정 spec을 주지 않으면 ko-pii 연결표 전체를 검사 대상으로 삼는다.
    selected_specs = specs or tuple(KO_PII_CATEGORY_SPECS.values())

    # SearchView마다 사용할 수 있는 개인정보 유형이 다르다. 예를 들어 번호 조각을
    # 붙인 format_compact에는 이름과 주소를 허용하지 않는다. 여기서 호출 요청과
    # SearchView 허용 유형의 교집합만 남긴다.
    selected_categories = frozenset(
        spec.category
        for spec in selected_specs
        if (
            view.allowed_categories is None
            or spec.category in view.allowed_categories
        )
    )

    # 인접 1~3줄에서 은행·계좌번호를 순서와 관계없이 확인한다. 예금주는
    # 필드명이나 짧은 이름 후보가 있을 때만 추가하며, 은행과 계좌는 필수다.
    if view.mode == "structured_block":
        context_detectors = []
        if "account_card_number" in selected_categories:
            context_detectors.append(("account_card_number", detect_account_block))
        if "person_name" in selected_categories:
            context_detectors.append(("person_name", detect_account_owner))
        for category, detector in context_detectors:
            for result in detector(view.text):
                yield AdapterDetection(
                    category=category,
                    result=result,
                    rule_id=f"context:{result.label}",
                    input_text=view.text,
                )
    # address_field는 주소 필드명에서 시작해 같은 줄과 정렬된 인접 줄을 연결한
    # SearchView다. 이 보기에서는 프로젝트 주소 필드 문맥 탐지기만 실행한다.
    elif view.mode == "address_field":
        if "address" in selected_categories:
            for result in detect_address_field(view.text):
                yield AdapterDetection(
                    category="address",
                    result=result,
                    rule_id=f"context:{result.label}",
                    input_text=view.text,
                )
    # token_spaced, format_compact와 field_compact 같은 일반 보기는 ko-pii 공식
    # 통합 API에 전달한다.
    else:
        # 내부 카테고리 목록을 ko-pii include 라벨 목록으로 펼친다.
        include_labels = tuple(
            label
            for spec in selected_specs
            if spec.category in selected_categories
            for label in spec.labels
        )
        # 주소 쉼표와 이름 앞 점처럼 OCR 때문에 끊기는 일부 문자만 보정한다.
        # 문자열 길이는 유지되므로 반환 start/end는 원래 SearchView와 같다.
        detector_text = _prepare_native_ko_pii_input(
            view.text,
            selected_categories,
        )
        if include_labels:
            # 이 호출이 일반 탐지의 중심이다. ko-pii 공식 통합 API가 유니코드
            # 정규화, 요청 라벨의 정규식·체크섬·사전·문맥 검증과 내부 겹침 해소를
            # 수행한다. 어댑터는 결과를 새로 판정하지 않고 이후 형식만 변환한다.
            native_results = detect_all(
                detector_text,
                include=include_labels,
                normalize=True,
            )
            # 법인번호와 외국인번호가 겹쳐 법인번호가 사라진 특수 충돌을 강한
            # 법인등록번호 필드가 있을 때만 재확인한다.
            native_results = _recover_field_anchored_native_results(
                detector_text,
                include_labels,
                native_results,
            )
            # 관리자가 체크섬 검사를 끈 경우에만 형식과 anchor가 충분한 일부 번호를
            # confidence 0.65의 저신뢰 후보로 보존한다.
            if not search_settings.checksum_validation_enabled:
                native_results = _relaxed_checksum_results(
                    detector_text,
                    include_labels,
                    native_results,
                )
            # ko-pii 공식 라벨을 프로젝트 내부 카테고리로 번역한다.
            for result in native_results:
                category = KO_PII_LABEL_TO_CATEGORY.get(result.label)
                if category is None or category not in selected_categories:
                    continue
                # ko-pii가 행정구역·도로명까지만 반환한 주소에 바로 이어지는 주소성
                # 괄호가 있으면 검증 후 같은 DetectionResult 범위에 포함한다.
                if result.label == "ADDRESS":
                    result = _extend_address_ocr_tail(
                        result,
                        detector_text,
                        view,
                    )
                yield AdapterDetection(
                    category=category,
                    result=result,
                    rule_id=f"ko_pii:{result.label}",
                    input_text=detector_text,
                )

    # 기본 이름 필드는 ko-pii PERSON과 AEGIS가 담당한다. 프로젝트 이름 필드 규칙은
    # 관리자가 추가한 필드명에만 실행한다. 한 줄의 token_spaced 또는 field_compact로
    # 제한하여 서로 먼 줄의 필드명과 값을 잘못 연결하지 않는다.
    person_context_labels = search_settings.custom_person_field_labels
    if (
        person_context_labels
        and any(spec.category == "person_name" for spec in selected_specs)
        and view.mode in {"token_spaced", "field_compact"}
        and len(view.lines) == 1
        and (
            view.allowed_categories is None
            or "person_name" in view.allowed_categories
        )
    ):
        for result in detect_custom_person_field(
            view.text,
            person_context_labels,
        ):
            yield AdapterDetection(
                category="person_name",
                result=result,
                rule_id=f"context:{result.label}",
                input_text=view.text,
            )
