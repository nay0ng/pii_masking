"""탐지기 결과를 최종 정책과 병합에 넘기기 전에 검증한다.

ko-pii와 AEGIS는 서로 다른 방법으로 후보를 찾는다. 이 파일에서는 모델 점수나
규칙 점수를 다시 계산하지 않고, 유형별로 반드시 확인할 수 있는 형식과 주변
필드명만 보완한다.

현재 적용하는 규칙은 세 가지다.

1. AEGIS 이메일은 전체 이메일 형식을 만족해야 한다.
2. AEGIS 운전면허번호는 국내 번호 형식과 지방경찰청 코드를 확인한다.
3. 15xx~18xx 대표전화는 대표번호 관련 필드명이나 키워드가 가까이 있어야 한다.
4. KB 연금·금융 서류에서 개인정보 값 바로 앞에 쓰이는 필드명을 탐지 근거에
   기록한다. 안내·동의 문구에만 등장하는 단어는 근거로 사용하지 않는다.
"""

from __future__ import annotations

from dataclasses import replace
import re
import unicodedata

from .detector_engine import PiiSpan
from .models import PreprocessedDocument


EMAIL_PATTERN = re.compile(
    r"^[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+"
    r"[A-Z]{2,63}$",
    re.IGNORECASE,
)

REPRESENTATIVE_PHONE_PATTERN = re.compile(r"^(?:15|16|17|18)\d{2}\d{4}$")

DRIVER_LICENSE_HYPHEN_PATTERN = re.compile(
    r"^([0-9]{2})-([0-9]{2})-([0-9]{6})-([0-9]{2})$"
)
DRIVER_LICENSE_COMPACT_PATTERN = re.compile(
    r"^([0-9]{2})([0-9]{2})([0-9]{6})([0-9]{2})$"
)
VALID_DRIVER_LICENSE_REGION_CODES = {f"{number:02d}" for number in range(11, 29)}

# 일반 휴대전화·유선전화 후보의 필드 근거로 사용하는 목록이다.
PHONE_FIELD_KEYWORDS = (
    "전화",
    "전화번호",
    "연락처",
    "휴대전화",
    "휴대전화번호",
    "휴대폰",
    "휴대폰번호",
    "핸드폰",
    "핸드폰번호",
    "자택전화",
    "자택전화번호",
    "직장전화",
    "직장전화번호",
    "직장번호",
    "자택연락처",
    "우선연락처",
    "담당자연락처",
    "대표자연락처",
    "사업장연락처",
    "고객연락처",
    "전화/휴대폰번호",
    "법인휴대폰",
    "대표전화",
    "대표번호",
    "고객센터",
    "상담전화",
    "문의전화",
    "TEL",
    "PHONE",
    "MOBILE",
)

# 15xx~18xx 대표번호는 일반 전화 문맥보다 강한 근거가 있을 때만 인정한다.
REPRESENTATIVE_PHONE_CONTEXT_KEYWORDS = (
    "대표전화",
    "대표번호",
    "고객센터",
    "상담전화",
    "문의전화",
    "콜센터",
    "ARS",
    "전화번호",
    "TEL",
    "연락처"
)

# KB 연금·금융 서식의 입력칸이나 표 머리글에서 개인정보 값 바로 앞에 쓰이는
# 필드명만 유형별 근거로 사용한다. 동의서·안내문 본문에 개인정보 유형이 단순히
# 열거된 경우는 이 목록에 넣지 않는다.
DOCUMENT_FIELD_KEYWORDS = {
    "resident_registration_number": (
        "주민등록번호",
        "주민번호",
        "주민(법인)등록번호",
        "가입자주민번호",
        "가입자주민등록번호",
        "대표자주민번호",
        "가입자고객번호",
        "실명확인번호",
    ),
    "foreigner_registration_number": (
        "외국인등록번호",
        "외국인번호",
        "국내거소신고번호",
        "거소신고번호",
        "가입자고객번호",
        "실명확인번호",
    ),
    "passport_number": (
        "여권번호",
        "실명확인번호",
    ),
    "driver_license_number": (
        "운전면허번호",
        "면허번호",
        "실명확인번호",
    ),
    "phone_number": PHONE_FIELD_KEYWORDS,
    "email_address": (
        "이메일",
        "이메일주소",
        "전자우편",
        "전자우편주소",
        "결과수신이메일",
        "전자세금계산서수신이메일",
        "전자세금계산서수신E-MAIL",
        "EMAIL",
        "email",
        "E-MAIL",
        "E-Mail",
        "E-mail",
        "e-mail",
        "e-mail주소",
    ),
}


def _compact_text(text: str) -> str:
    """문맥 비교용으로 유니코드를 정리하고 공백을 제거한다."""
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(char for char in normalized if not char.isspace()).upper()


def _digits_and_common_separators(text: str) -> str:
    """전화번호 판정에 불필요한 공백과 일반 구분자만 제거한다."""
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        char
        for char in normalized
        if char not in {" ", "\t", "-", ".", "/"}
    )


def is_valid_email(text: str) -> bool:
    """공백으로 나뉜 OCR 조각을 붙인 뒤 전체 이메일 형식을 확인한다."""
    candidate = _compact_text(text)
    return EMAIL_PATTERN.fullmatch(candidate) is not None


def is_representative_phone(text: str) -> bool:
    """15xx~18xx로 시작하는 8자리 대표전화 형식인지 확인한다."""
    candidate = _digits_and_common_separators(text)
    return REPRESENTATIVE_PHONE_PATTERN.fullmatch(candidate) is not None


def is_valid_driver_license(text: str, nearby_text: str) -> bool:
    """AEGIS 운전면허 후보가 국내 번호 형식인지 확인한다.

    하이픈이 있는 ``XX-YY-NNNNNN-CC`` 형태는 형식과 지역 코드를 확인한다.
    하이픈 없는 12자리는 다른 일련번호와 충돌할 수 있으므로 운전면허 관련
    필드명이 후보 가까이에 있을 때만 인정한다.
    """
    normalized = unicodedata.normalize("NFKC", text)
    candidate = "".join(char for char in normalized if not char.isspace())
    has_hyphen = "-" in candidate
    pattern = (
        DRIVER_LICENSE_HYPHEN_PATTERN
        if has_hyphen
        else DRIVER_LICENSE_COMPACT_PATTERN
    )
    match = pattern.fullmatch(candidate)
    if match is None or match.group(1) not in VALID_DRIVER_LICENSE_REGION_CODES:
        return False

    if has_hyphen:
        return True

    driver_keywords = DOCUMENT_FIELD_KEYWORDS["driver_license_number"]
    return _find_keyword(nearby_text, driver_keywords) is not None


def _nearby_text(document: PreprocessedDocument, span: PiiSpan) -> str:
    """후보 앞쪽 필드명과 바로 뒤 설명을 확인할 수 있는 문맥을 반환한다."""
    raw_text = document.index.raw_text
    start = max(0, span.raw_start - 64)
    end = min(len(raw_text), span.raw_end + 24)
    return _compact_text(raw_text[start:end])


def _find_keyword(text: str, keywords: tuple[str, ...]) -> str | None:
    """공백과 대소문자를 정리한 문맥에서 가장 구체적인 키워드를 반환한다."""
    for keyword in sorted(keywords, key=len, reverse=True):
        if _compact_text(keyword) in text:
            return keyword
    return None


def _is_aegis_email(span: PiiSpan) -> bool:
    return (
        span.entity_type == "email_address"
        and span.rule_id.startswith("ner:aegis-v2:")
    )


def _is_aegis_driver_license(span: PiiSpan) -> bool:
    return (
        span.entity_type == "driver_license_number"
        and span.rule_id.startswith("ner:aegis-v2:")
    )


def _append_field_evidence(span: PiiSpan, nearby_text: str) -> PiiSpan:
    """개인정보 값 바로 앞에서 확인된 필드명을 evidence에 추가한다."""
    evidence = list(span.evidence)
    field_keywords = DOCUMENT_FIELD_KEYWORDS.get(span.entity_type, ())
    field_keyword = _find_keyword(nearby_text, field_keywords)
    if field_keyword is not None:
        item = f"field:{field_keyword}"
        if item not in evidence:
            evidence.append(item)

    if tuple(evidence) == span.evidence:
        return span
    return replace(span, evidence=tuple(evidence))


def postprocess_detection_spans(
    document: PreprocessedDocument,
    spans: tuple[PiiSpan, ...],
) -> tuple[PiiSpan, ...]:
    """형식과 문맥을 확인한 뒤 정책·병합 단계에 전달할 후보를 반환한다."""
    processed: list[PiiSpan] = []

    for span in spans:
        if _is_aegis_email(span) and not is_valid_email(span.text):
            continue

        nearby_text = _nearby_text(document, span)
        if (
            _is_aegis_driver_license(span)
            and not is_valid_driver_license(span.text, nearby_text)
        ):
            continue

        if (
            span.entity_type == "phone_number"
            and is_representative_phone(span.text)
            and _find_keyword(
                nearby_text,
                REPRESENTATIVE_PHONE_CONTEXT_KEYWORDS,
            ) is None
        ):
            continue

        processed.append(_append_field_evidence(span, nearby_text))

    return tuple(processed)
