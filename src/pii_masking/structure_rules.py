"""ko-pii의 단일 문자열 규칙을 보완하는 최소 문서 구조 규칙.

특정 파일이나 실제 개인정보 값을 사용하지 않고, 인접한 1~3줄 안의 금융기관,
계좌번호, 선택적 예금주처럼 문서 종류가 달라도 반복되는 구조만 다룬다. 결과
offset은 입력 문자열 기준이므로 SearchView의 token/bbox 역매핑을 그대로 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterator

from ko_pii.core.types import DetectionResult, RiskLevel
from ko_pii.patterns import person


# 정식 은행명과 문서에서 흔히 쓰는 짧은 표기를 함께 둔다. 짧은 표기는 단독으로
# 신뢰하지 않고, 바로 다음 줄의 10~16자리 계좌 형식까지 있을 때만 anchor가 된다.
BANK_ALIASES = frozenset(
    {
        "국민", "국민은행", "KB국민은행",
        "신한", "신한은행",
        "우리", "우리은행",
        "하나", "하나은행", "KEB하나은행",
        "기업", "기업은행", "IBK기업은행",
        "농협", "농협은행", "NH농협",
        "수협", "수협은행",
        "신협", "신용협동조합",
        "새마을금고", "MG새마을금고",
        "우체국",
        "산업은행", "KDB산업은행",
        "카카오뱅크", "카뱅",
        "케이뱅크", "K뱅크",
        "토스뱅크",
        "부산은행", "대구은행", "경남은행", "광주은행",
        "전북은행", "제주은행",
    }
)

_ACCOUNT_LINE_PATTERN = re.compile(
    r"(?<!\d)(?P<value>\d(?:[\d\s-]{8,30})\d)(?!\d)"
)
_OWNER_LINE_PATTERN = re.compile(
    r"^\s*(?P<value>[가-힣](?:\s*[가-힣]){1,3})(?=\s*(?:\d|$))"
)
_OWNER_FIELD_PATTERN = re.compile(
    r"(?:예\s*금\s*주|계\s*좌\s*주|명\s*의\s*인|성\s*명)"
    r"\s*[:：·ㆍ.\-]?\s*(?P<value>[가-힣](?:\s*[가-힣]){1,3})"
)
_ACCOUNT_FIELD_LABELS = ("계좌번호", "계좌", "입금계좌", "결제계좌")
_ADDRESS_REGION_PATTERN = re.compile(
    r"(?:특별시|광역시|특별자치시|특별자치도|[가-힣]{2,}도|[가-힣]{2,}시)"
)
_ADDRESS_NUMBER_PATTERN = re.compile(r"(?<!\d)\d{1,6}(?:-\d{1,6})?(?!\d)")
_DATE_COLUMN_PATTERN = re.compile(
    r"(?<!\d)\d{4}\s*[-./년]\s*\d{1,2}\s*[-./월]\s*\d{1,2}(?:\s*일)?(?!\d)"
)
_ADDRESS_NEXT_FIELD_PATTERN = re.compile(
    r"\s+(?=(?:"
    r"(?:사업장\s*)?소\s*재\s*지|본\s*점\s*소\s*재\s*지|주\s*소|"
    r"사업의\s*종류|업\s*태|종\s*목|발급\s*사유|개업\s*연월일|"
    r"대표이사|대표자|성\s*명|생년월일|전화|팩스|전자우편주소|이메일"
    r")\s*[:：]?)"
)


def _compact_label(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", text)


def _edit_distance_at_most_one(first: str, second: str) -> bool:
    """짧은 금융기관 표기의 OCR 한 글자 오인만 허용한다."""
    if abs(len(first) - len(second)) > 1:
        return False
    if first == second:
        return True
    if len(first) == len(second):
        return sum(a != b for a, b in zip(first, second)) == 1

    short, long = (first, second) if len(first) < len(second) else (second, first)
    for index in range(len(long)):
        if long[:index] + long[index + 1:] == short:
            return True
    return False


def _bank_anchor(line: str) -> tuple[str, float] | None:
    """한 줄 전체 또는 문장 안에서 금융기관 표기를 찾는다."""
    compact = _compact_label(line)
    if not compact:
        return None
    if compact in BANK_ALIASES:
        return compact, 0.85

    # 정식 은행명은 ``은행명: 국민은행``처럼 다른 글자와 같은 줄에 있어도 찾는다.
    long_matches = tuple(
        alias for alias in BANK_ALIASES
        if len(alias) >= 4 and alias.casefold() in compact.casefold()
    )
    if long_matches:
        return max(long_matches, key=len), 0.85

    # 짧은 약칭은 일반 단어와 겹칠 수 있어 앞뒤에 다른 문자가 없을 때만 찾는다.
    for alias in sorted(BANK_ALIASES, key=len, reverse=True):
        if len(alias) >= 4:
            continue
        pattern = re.compile(
            rf"(?<![0-9A-Za-z가-힣]){re.escape(alias)}(?![0-9A-Za-z가-힣])",
            re.IGNORECASE,
        )
        if pattern.search(line):
            return alias, 0.80

    # 긴 은행명은 우연히 한 글자 차이가 날 가능성이 커서 fuzzy 보정을 하지 않는다.
    if len(compact) > 4:
        return None
    fuzzy_targets = tuple(alias for alias in BANK_ALIASES if 2 <= len(alias) <= 4)
    matches = tuple(
        alias for alias in fuzzy_targets
        if _edit_distance_at_most_one(compact, alias)
    )
    if matches:
        # ``신형``은 신한/신협 모두와 한 글자 차이라 특정 은행으로 확정할 수
        # 없지만, 뒤의 계좌 형식과 예금주 줄까지 맞으면 금융기관 anchor 자체는
        # 유효하다. 후보 은행은 모두 남기고 낮은 신뢰도로 검토 대상으로 보낸다.
        return "/".join(sorted(matches)), 0.60
    return None


@dataclass(frozen=True)
class _TextLine:
    text: str
    offset: int
    index: int


@dataclass(frozen=True)
class _ValueCandidate:
    value: str
    start: int
    end: int
    line_index: int
    field_anchored: bool


@dataclass(frozen=True)
class _AccountContext:
    bank: str
    bank_confidence: float
    account: _ValueCandidate
    owner: _ValueCandidate | None


def _text_lines(text: str) -> tuple[_TextLine, ...] | None:
    """한 줄부터 세 줄까지의 제한 SearchView를 offset과 함께 나눈다."""
    lines = text.splitlines(keepends=True)
    if not 1 <= len(lines) <= 3:
        return None
    result: list[_TextLine] = []
    offset = 0
    for index, line in enumerate(lines):
        result.append(_TextLine(line.rstrip("\r\n"), offset, index))
        offset += len(line)
    return tuple(result)


def _account_field_anchor(line: str, start: int) -> bool:
    prefix = _compact_label(line[max(0, start - 24):start])
    return any(prefix.endswith(label) for label in _ACCOUNT_FIELD_LABELS)


def _find_account_candidate(
    lines: tuple[_TextLine, ...],
    bank_line_indexes: tuple[int, ...],
) -> _ValueCandidate | None:
    candidates: list[tuple[tuple[int, int, int, int], _ValueCandidate]] = []
    for line in lines:
        for match in _ACCOUNT_LINE_PATTERN.finditer(line.text):
            value = match.group("value")
            digits = re.sub(r"[\s-]", "", value)
            if not digits.isdigit() or not 10 <= len(digits) <= 16:
                continue
            field_anchored = _account_field_anchor(line.text, match.start("value"))
            nearest_bank_distance = min(
                abs(line.index - bank_index) for bank_index in bank_line_indexes
            )
            candidate = _ValueCandidate(
                value=value,
                start=line.offset + match.start("value"),
                end=line.offset + match.end("value"),
                line_index=line.index,
                field_anchored=field_anchored,
            )
            score = (
                int(field_anchored),
                int("-" in value),
                -nearest_bank_distance,
                len(digits),
            )
            candidates.append((score, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _person_value_is_valid(value: str) -> bool:
    compact = re.sub(r"\s", "", value)
    return any(result.text == compact for result in person.detect(f"성명: {compact}"))


def _find_owner_candidate(
    lines: tuple[_TextLine, ...],
    account: _ValueCandidate,
    bank_line_indexes: tuple[int, ...],
) -> _ValueCandidate | None:
    # 예금주·명의인 같은 강한 필드명이 있으면 줄 순서와 관계없이 우선한다.
    for line in lines:
        match = _OWNER_FIELD_PATTERN.search(line.text)
        if match is None or not _person_value_is_valid(match.group("value")):
            continue
        return _ValueCandidate(
            value=match.group("value"),
            start=line.offset + match.start("value"),
            end=line.offset + match.end("value"),
            line_index=line.index,
            field_anchored=True,
        )

    # ``국민은행 123-... 홍길동``처럼 계좌 뒤에 이름이 붙은 한 줄 구조도 허용한다.
    account_line = lines[account.line_index]
    local_account_end = account.end - account_line.offset
    tail = account_line.text[local_account_end:]
    match = _OWNER_LINE_PATTERN.search(tail)
    if match is not None and _person_value_is_valid(match.group("value")):
        return _ValueCandidate(
            value=match.group("value"),
            start=account.end + match.start("value"),
            end=account.end + match.end("value"),
            line_index=account.line_index,
            field_anchored=False,
        )

    # 은행과 계좌가 이미 확인되었을 때만 별도 짧은 이름 줄을 예금주로 사용한다.
    for line in lines:
        if line.index in bank_line_indexes or line.index == account.line_index:
            continue
        match = _OWNER_LINE_PATTERN.search(line.text)
        if match is None or not _person_value_is_valid(match.group("value")):
            continue
        return _ValueCandidate(
            value=match.group("value"),
            start=line.offset + match.start("value"),
            end=line.offset + match.end("value"),
            line_index=line.index,
            field_anchored=False,
        )
    return None


def _account_context(text: str) -> _AccountContext | None:
    """인접 1~3줄에서 순서와 관계없이 은행·계좌·예금주 근거를 연결한다."""
    lines = _text_lines(text)
    if lines is None:
        return None

    bank_candidates_list: list[tuple[int, tuple[str, float]]] = []
    for line in lines:
        anchor = _bank_anchor(line.text)
        if anchor is not None:
            bank_candidates_list.append((line.index, anchor))
    bank_candidates = tuple(bank_candidates_list)
    if not bank_candidates:
        return None

    bank_line_indexes = tuple(item[0] for item in bank_candidates)
    account = _find_account_candidate(lines, bank_line_indexes)
    if account is None:
        return None

    _, (bank, bank_confidence) = max(
        bank_candidates,
        key=lambda item: (item[1][1], -abs(item[0] - account.line_index)),
    )
    owner = _find_owner_candidate(lines, account, bank_line_indexes)
    return _AccountContext(bank, bank_confidence, account, owner)


def detect_account_block(text: str) -> Iterator[DetectionResult]:
    """인접 1~3줄의 금융기관과 계좌 형식을 확인해 계좌번호를 반환한다."""
    context = _account_context(text)
    if context is None:
        return
    confidence = min(
        0.90,
        context.bank_confidence + (0.05 if context.account.field_anchored else 0.0),
    )
    evidence = ["structure:nearby-bank-account", f"bank:{context.bank}"]
    if context.account.field_anchored:
        evidence.append("anchor:account-field")
    yield DetectionResult(
        label="ACCOUNT_CONTEXT",
        text=context.account.value,
        start=context.account.start,
        end=context.account.end,
        risk_level=RiskLevel.HIGH,
        confidence=confidence,
        evidence=evidence,
        legal_basis="개인정보보호법 제2조; 금융실명법",
        extra={"bank": context.bank, "category": "일반개인정보"},
    )


def detect_account_owner(text: str) -> Iterator[DetectionResult]:
    """은행·계좌가 확인된 인접 구간에서 필드 또는 짧은 이름 후보를 반환한다."""
    context = _account_context(text)
    if context is None or context.owner is None:
        return
    confidence = min(
        0.90,
        context.bank_confidence + (0.05 if context.owner.field_anchored else 0.0),
    )
    evidence = ["structure:nearby-bank-account-owner", f"bank:{context.bank}"]
    if context.owner.field_anchored:
        evidence.append("anchor:owner-field")
    yield DetectionResult(
        label="PERSON_ACCOUNT_OWNER",
        text=context.owner.value,
        start=context.owner.start,
        end=context.owner.end,
        risk_level=RiskLevel.MEDIUM,
        confidence=confidence,
        evidence=evidence,
        legal_basis="개인정보보호법 제2조",
        extra={"role": "account_owner", "category": "일반개인정보"},
    )


def detect_custom_person_field(
    text: str,
    field_labels: tuple[str, ...],
) -> Iterator[DetectionResult]:
    """관리자 추가 이름 필드 바로 뒤의 2~4자 한글 값을 탐지한다.

    관리자 필드명은 실행 설정으로 전달받는다. 값의 형태는 ko-pii PERSON으로
    한 번 더 검증하되, 결과 위치는 실제 SearchView 문자열의 offset을 사용한다.
    """
    for field_label in field_labels:
        if not field_label:
            continue
        flexible_label = r"\s*".join(map(re.escape, field_label))
        pattern = re.compile(
            rf"{flexible_label}\s*(?:\([^)]{{1,12}}\))?\s*"
            rf"[:：·ㆍ.\-]?\s*"
            rf"(?P<value>[가-힣](?:\s*[가-힣]){{1,3}})"
        )
        for match in pattern.finditer(text):
            raw_value = match.group("value")
            compact_value = re.sub(r"\s", "", raw_value)
            if not any(
                result.text == compact_value
                for result in person.detect(f"성명: {compact_value}")
            ):
                continue
            yield DetectionResult(
                label="CUSTOM_PERSON_FIELD",
                text=raw_value,
                start=match.start("value"),
                end=match.end("value"),
                risk_level=RiskLevel.MEDIUM,
                confidence=0.85,
                evidence=[
                    "structure:custom-person-field",
                    f"field:{field_label}",
                ],
                legal_basis="개인정보보호법 제2조",
                extra={"field": field_label, "category": "일반개인정보"},
            )


def detect_address_field(text: str) -> Iterator[DetectionResult]:
    """좌표로 제한한 주소 필드 값 전체를 OCR 원문 변경 없이 후보로 반환한다.

    이 함수는 임의의 전체 문장에 실행하지 않는다. ``search_views``가 주소 label과
    값 열 정렬을 확인해 만든 ``address_field`` 보기에만 적용한다. 행정구역 표현과
    번지 숫자가 모두 있어야 하므로 단순 안내문은 주소로 만들지 않는다.
    """
    start = len(text) - len(text.lstrip())
    end = len(text.rstrip())
    if end <= start:
        return
    value = text[start:end]
    # 같은 표 행의 다음 필드나 날짜 열을 주소에 포함하지 않는다. 문자열을
    # 교정하지 않고 반환 범위의 끝만 줄이므로 원본 token/bbox 매핑은 유지된다.
    stop_indexes = [
        match.start()
        for pattern in (_ADDRESS_NEXT_FIELD_PATTERN, _DATE_COLUMN_PATTERN)
        for match in pattern.finditer(value)
        if match.start() > 0
    ]
    if stop_indexes:
        value = value[:min(stop_indexes)].rstrip()
        end = start + len(value)
    if not value:
        return
    region_match = _ADDRESS_REGION_PATTERN.search(value)
    if region_match is None or region_match.start() > 12:
        return
    # 날짜의 연·월·일을 번지로 오해하지 않도록 날짜 문자열을 제외한 복사본에서
    # 주소 숫자를 확인한다. 실제 DetectionResult에는 원문 value를 그대로 넣는다.
    number_check_text = _DATE_COLUMN_PATTERN.sub(" ", value)
    if _ADDRESS_NUMBER_PATTERN.search(number_check_text) is None:
        return

    yield DetectionResult(
        label="ADDRESS_FIELD",
        text=value,
        start=start,
        end=end,
        risk_level=RiskLevel.MEDIUM,
        confidence=0.85,
        evidence=["structure:address-field", "layout:aligned-value-lines"],
        legal_basis="개인정보보호법 제2조",
        extra={"category": "일반개인정보"},
    )
