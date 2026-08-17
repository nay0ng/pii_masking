"""OCR token으로 탐지 목적별 검색 문자열과 원본 위치 매핑을 만든다.

SearchView는 탐지 문자열 ``text``와 그 문자열의 각 문자가 어느 OCR token에서
왔는지 나타내는 매핑을 함께 가진다.

코드 검토 순서
1. ``SearchViewSettings``에서 줄 결합 범위, 숫자 형식 정리 수준과 사용자 필드명을
   확인한다.
2. ``SearchView``에서 탐지 문자열과 위치 매핑의 자료구조를 확인한다.
3. ``build_search_view()``에서 token을 문자열로 연결하는 기본 방식을 확인한다.
4. ``build_line_search_views()``에서 한 줄용 입력을 만드는 방식을 확인한다.
5. ``build_cross_line_search_views()``에서 인접한 여러 줄을 언제 연결하는지 확인한다.
6. 주소 필드 전용 범위는 ``build_address_field_search_view()``에서 확인한다.

이 파일은 개인정보를 판정하지 않는다. SearchView를 만들 뿐이며 실제 ko-pii와
AEGIS 호출, bbox 역매핑과 중복 제거는 ``detector_engine.py``가 담당한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re

from .models import OcrLine, OcrToken
from .ocr_preprocessing import build_text_index, normalize_for_search


# 공백으로 분리된 숫자·기호 조각을 붙여야 하는 형식 기반 탐지기만
# format_compact 보기를 사용한다. 이름·주소는 의미 경계가 중요하므로 제외한다.
FORMAT_COMPACT_CATEGORIES = frozenset(
    {
        "email_address",
        "phone_number",
        "url",
        "date",
        "account_card_number",
        "credential_secret",
        "resident_registration_number",
        "foreigner_registration_number",
        "passport_number",
        "driver_license_number",
        "business_registration_number",
        "corporate_registration_number",
        "health_insurance_number",
        "vehicle_plate_number",
        "ip_address",
        "land_lot_number",
        "prescription_number",
        "drug_code",
        "employee_number",
        "petition_number",
        "court_case_number",
        "postal_code",
        "document_number",
        "age",
        "height",
        "weight",
    }
)

# 구분자 복구와 OCR 숫자 혼동문자 복구는 숫자형 개인정보에만 적용한다.
# URL의 ``://``나 이메일의 영문자를 손상시키지 않기 위해 범위를 분리한다.
NUMBER_RECOVERY_CATEGORIES = frozenset(
    {
        "phone_number",
        "account_card_number",
        "resident_registration_number",
        "foreigner_registration_number",
        "driver_license_number",
        "business_registration_number",
        "corporate_registration_number",
        "health_insurance_number",
    }
)

# OCR이 숫자를 모양이 비슷한 영문자로 반환하는 경우에 사용할 대응표다.
# 문서 전체를 치환하지 않고, 위 NUMBER_RECOVERY_CATEGORIES에 포함된 숫자형
# 개인정보용 SearchView에서 가까운 곳에 실제 숫자가 있을 때만 적용한다.
OCR_DIGIT_CONFUSIONS = {
    "O": "0", "o": "0", "Q": "0",
    "I": "1", "i": "1", "l": "1", "L": "1", "|": "1", "!": "1",
    "Z": "2", "z": "2",
    "S": "5", "s": "5",
    "G": "6",
    "B": "8",
}

# 이메일 필드명과 값이 별도 OCR token으로 들어오면 SearchView가 둘을
# ``E-mailuser@example.com``으로 붙이지 않도록 경계를 보존한다. 단순 공백은
# ko-pii 정규화 과정에서 제거되므로 합성 콜론을 넣는다. 이 콜론은 원본 문자나
# bbox에 대응하지 않는다. 이메일 주소 안에서 분리된 아이디, @, 도메인과 마침표
# 조각은 기존처럼 계속 붙인다.
EMAIL_FIELD_LABELS = frozenset(
    {
        "email",
        "e-mail",
        "e_mail",
        "e.mail",
        "emailaddress",
        "e-mailaddress",
        "e_mailaddress",
        "e.mailaddress",
    }
)
EMAIL_VALUE_PATTERN = re.compile(
    r"^[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+"
    r"[A-Z]{2,63}$",
    re.IGNORECASE,
)

CONTEXT_TEXT_CATEGORIES = frozenset(
    {
        "nationality",
        "education_history",
        "academic_major",
        "job_position",
    }
)

# 전체 줄의 토큰 간격 보기는 일반 단어를 인명으로 오해하기 쉬운 PERSON을
# 제외한다. 이름은 콜론 주변 field_compact 보기에서 문맥이 제한된 상태로 찾는다.
TOKEN_SPACED_CATEGORIES = (
    FORMAT_COMPACT_CATEGORIES
    | CONTEXT_TEXT_CATEGORIES
    | frozenset({"address"})
)
FIELD_COMPACT_CATEGORIES = (
    FORMAT_COMPACT_CATEGORIES
    | CONTEXT_TEXT_CATEGORIES
    | frozenset({"address"})
)

# PERSON은 전체 줄에 허용하지 않고, 아래 label에서 시작하는 제한된 구간만 별도
# SearchView로 만든다. 긴 안내문 중간에 "신청인"이나 "성명"이 등장해도 줄 전체의
# 다른 일반 단어를 이름으로 오해하지 않게 하기 위함이다.
PERSON_CONTEXT_KEYWORDS = (
    "대표이사",
    "공동대표자",
    "대표자",
    "성명",
    "이름",
    "신청인",
    "신청자",
    "민원인",
    "청구인",
    "보호자",
    "대리인",
    "법정대리인",
    "계약자",
    "보험계약자",
    "피보험자",
    "수익자",
    "가입자",
    "예금주",
    "계좌주",
    "소유자",
    "채무자",
    "채권자",
    "담당자",
    "당사자",
    "원고",
    "피고",
)
PERSON_ONLY_CATEGORIES = frozenset({"person_name"})
STRUCTURED_BLOCK_CATEGORIES = frozenset({"account_card_number", "person_name"})
ADDRESS_FIELD_CATEGORIES = frozenset({"address"})
NON_PERSON_FIELD_PREFIXES = (
    "주민등록번호", "외국인등록번호", "법인등록번호", "사업자등록번호",
    "생년월일", "출생연월일", "생일", "주소", "소재지", "전화", "연락처", "휴대전화",
    "이메일", "전자우편", "계좌", "계좌번호", "여권번호", "면허번호",
)
ADDRESS_FIELD_LABELS = (
    "사업장소재지", "사업장주소", "본점소재지", "우편물수령지",
    "법인주소", "등록기준지", "출생장소", "거주지", "소재지", "주소",
    "사용본거지",
)
ADDRESS_STOP_FIELD_PREFIXES = (
    "사업의종류", "업종", "업태", "종목", "발급사유", "개업연월일",
    "법인등록번호", "사업자등록번호", "등록번호", "대표자", "성명",
    "생년월일", "전화", "팩스", "전자우편주소", "이메일",
)

# 표 셀의 두 번째 줄은 값 시작점이 아니라 필드 label 쪽으로 돌아가 시작하는 경우가 있다.
# 고정 px 대신 OCR 글자 높이에 비례해 들여쓰기 차이를 허용한다.
ADDRESS_CONTINUATION_X_TOLERANCE_HEIGHTS = 4
ADDRESS_CONTINUATION_VERTICAL_GAP_HEIGHTS = 1.25
ADDRESS_DATE_COLUMN_TOKEN = re.compile(
    r"^(?:19|20)\d{2}(?:[-./]\d{1,2}){2}$"
)
ADDRESS_CONTINUATION_TOKEN = re.compile(
    r"^(?:"
    r"\d+[-/]\d+|"
    r"\d+(?:[-/]\d+)?번지|"
    r"\d+(?:동|층|호)(?:[A-Za-z가-힣0-9]{0,30})|"
    r"[A-Za-z가-힣0-9]{1,30}(?:대로|번길|로|길)\d+(?:-\d+)?|"
    r"[A-Za-z가-힣0-9]{1,30}"
    r"(?:특별시|광역시|특별자치시|특별자치도|시|군|구|읍|면|동|리|"
    r"대로|번길|로|길|아파트|빌딩|타워|센터|오피스텔|상가|플라자)"
    r")"
)


class PatternStrictness(StrEnum):
    """숫자형 개인정보 후보를 만들 때 허용할 OCR 복구 범위."""

    EXACT = "exact"
    NORMALIZED = "normalized"
    RECOVERED = "recovered"
    OCR_TOLERANT = "ocr_tolerant"


@dataclass(frozen=True)
class SearchViewSettings:
    """문서 구조 탐색 범위와 관리자가 추가한 필드명을 보관한다.

    기본 필드명은 코드의 검증된 목록을 유지하고, 관리자 입력은 실행 시점에
    추가한다. 따라서 설정 변경 때문에 Python/C++ 엔진을 다시 빌드할 필요가 없다.
    """

    max_adjacent_lines: int = 3
    pattern_strictness: PatternStrictness = PatternStrictness.NORMALIZED
    checksum_validation_enabled: bool = True
    custom_person_field_labels: tuple[str, ...] = ()
    custom_address_field_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_adjacent_lines not in {1, 2, 3}:
            raise ValueError("max_adjacent_lines는 1, 2, 3 중 하나여야 합니다.")
        if not isinstance(self.pattern_strictness, PatternStrictness):
            object.__setattr__(
                self,
                "pattern_strictness",
                PatternStrictness(self.pattern_strictness),
            )
        object.__setattr__(
            self,
            "custom_person_field_labels",
            _clean_field_labels(self.custom_person_field_labels),
        )
        object.__setattr__(
            self,
            "custom_address_field_labels",
            _clean_field_labels(self.custom_address_field_labels),
        )

    @property
    def person_field_labels(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            (*PERSON_CONTEXT_KEYWORDS, *self.custom_person_field_labels)
        ))

    @property
    def address_field_labels(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            (*ADDRESS_FIELD_LABELS, *self.custom_address_field_labels)
        ))


def _clean_field_labels(labels: tuple[str, ...]) -> tuple[str, ...]:
    cleaned = []
    for label in labels:
        value = "".join(normalize_for_search(str(label)).split()).strip(":")
        if value and value not in cleaned:
            cleaned.append(value)
    return tuple(cleaned)


DEFAULT_SEARCH_VIEW_SETTINGS = SearchViewSettings()


@dataclass(frozen=True)
class SearchView:
    """탐지용 문자열 한 가지와 각 글자의 원본 위치를 함께 보관한다."""

    text: str
    char_to_raw: tuple[int | None, ...]
    char_to_token: tuple[int | None, ...]
    lines: tuple[OcrLine, ...]
    mode: str
    allowed_categories: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if not (
            len(self.text) == len(self.char_to_raw) == len(self.char_to_token)
        ):
            raise ValueError("SearchView 문자열과 위치 매핑의 길이가 다릅니다.")

    def raw_indexes_for_span(self, start: int, end: int) -> tuple[int, ...]:
        """검색 span에 실제로 대응하는 문서 raw 문자 위치를 반환한다."""
        if start < 0 or end <= start or end > len(self.text):
            raise ValueError(f"잘못된 SearchView span입니다: [{start}, {end})")
        return tuple(
            raw_index
            for raw_index in self.char_to_raw[start:end]
            if raw_index is not None
        )

    def token_ids_for_span(self, start: int, end: int) -> tuple[int, ...]:
        """검색 span에 연결된 token ID를 발견 순서대로 중복 없이 반환한다."""
        if start < 0 or end <= start or end > len(self.text):
            raise ValueError(f"잘못된 SearchView span입니다: [{start}, {end})")
        result: list[int] = []
        seen: set[int] = set()
        for token_id in self.char_to_token[start:end]:
            if token_id is not None and token_id not in seen:
                result.append(token_id)
                seen.add(token_id)
        return tuple(result)


def _is_format_piece(token: OcrToken) -> bool:
    """다른 토큰과 붙여서 번호·이메일 형식을 복원할 수 있는 조각인지 판단한다."""
    text = normalize_for_search(token.text)
    if not text:
        return False
    if any(char.isdigit() for char in text):
        return True
    if any(char in "@:/._+-" for char in text):
        return True
    return all(char.isascii() and (char.isalpha() or char in "@:/._+-") for char in text)


def _format_piece_flags(tokens: tuple[OcrToken, ...]) -> dict[int, bool]:
    """차량번호의 한글 용도 문자처럼 숫자 사이의 한 글자도 형식 조각으로 포함한다."""
    flags = [_is_format_piece(token) for token in tokens]
    for index, token in enumerate(tokens):
        text = normalize_for_search(token.text)
        if (
            not flags[index]
            and len(text) == 1
            and "가" <= text <= "힣"
            and index > 0
            and index + 1 < len(tokens)
            and flags[index - 1]
            and flags[index + 1]
        ):
            flags[index] = True
    return {token.token_id: flag for token, flag in zip(tokens, flags)}


def _email_field_value_boundaries(tokens: tuple[OcrToken, ...]) -> set[int]:
    """영문 이메일 필드명 다음에 이메일 값이 시작되는 token 위치를 찾는다."""
    boundaries: set[int] = set()
    normalized_tokens = tuple(normalize_for_search(token.text) for token in tokens)

    for label_start in range(len(tokens)):
        # 같은 위치에서 ``EMAIL``과 ``EMAIL ADDRESS``가 모두 일치하면 더 긴
        # 필드명을 사용한다. 그래야 ADDRESS를 이메일 아이디 조각으로 오해하지 않는다.
        value_index: int | None = None
        label_end_limit = min(len(tokens), label_start + 3)
        for label_end in range(label_start + 1, label_end_limit + 1):
            label = (
                "".join(normalized_tokens[label_start:label_end])
                .casefold()
                .rstrip(":：")
            )
            if label in EMAIL_FIELD_LABELS:
                value_index = label_end

        if value_index is None or value_index >= len(tokens):
            continue

        # 이메일 값 하나가 여러 token으로 나뉠 수 있으므로 뒤쪽 조각을 차례로
        # 붙여 완전한 이메일이 되는 구간이 있는지 확인한다. 이메일 뒤의 다른
        # 영문 필드까지 무조건 붙여 검사하지 않도록 최대 8개 token만 본다.
        value_end_limit = min(len(tokens), value_index + 8)
        for value_end in range(value_index + 1, value_end_limit + 1):
            compact_value = "".join(normalized_tokens[value_index:value_end])
            if EMAIL_VALUE_PATTERN.fullmatch(compact_value):
                boundaries.add(tokens[value_index].token_id)
                break

    return boundaries


def build_search_view(
    lines: tuple[OcrLine, ...],
    raw_offset: int,
    *,
    mode: str,
    allowed_categories: frozenset[str] | None = None,
) -> SearchView:
    """compact 인덱스를 바탕으로 목적에 맞는 합성 구분자를 넣은 검색 보기를 만든다."""
    if mode not in {
        "token_spaced", "format_compact", "field_compact", "structured_block",
        "address_field", "format_recovered", "format_ocr_tolerant",
        "table_field", "table_field_compact",
    }:
        raise ValueError(f"지원하지 않는 SearchView mode입니다: {mode}")

    tokens = tuple(token for line in lines for token in line.tokens)
    segment_index = build_text_index(tokens, lines=lines)
    token_to_line = {
        token.token_id: line.line_id
        for line in lines
        for token in line.tokens
    }
    format_flags = _format_piece_flags(tokens)
    email_value_boundaries = _email_field_value_boundaries(tokens)

    chars: list[str] = []
    char_to_raw: list[int | None] = []
    char_to_token: list[int | None] = []
    previous_token_id: int | None = None
    previous_line_id: int | None = None

    for normalized_index, char in enumerate(segment_index.normalized_text):
        token_id = segment_index.normalized_char_to_token[normalized_index]
        line_id = token_to_line[token_id]
        if previous_token_id is not None and token_id != previous_token_id:
            separator = ""
            if (
                mode in {"token_spaced", "format_compact", "field_compact"}
                and token_id in email_value_boundaries
            ):
                separator = "" if chars and chars[-1] in ":：" else ":"
            elif previous_line_id != line_id:
                # 주소 필드 보기는 좌표로 인접·정렬을 먼저 검증했으므로 작은
                # 탐지 구간 안에서만 OCR 줄바꿈을 공백 하나로 바꾼다.
                separator = " " if mode == "address_field" else "\n"
            elif mode in {"token_spaced", "structured_block", "address_field"}:
                separator = " "
            elif mode == "format_compact":
                if not (format_flags[previous_token_id] and format_flags[token_id]):
                    separator = " "

            if separator:
                chars.append(separator)
                char_to_raw.append(None)
                char_to_token.append(None)

        chars.append(char)
        char_to_raw.append(
            raw_offset + segment_index.normalized_char_to_raw[normalized_index]
        )
        char_to_token.append(token_id)
        previous_token_id = token_id
        previous_line_id = line_id

    return SearchView(
        text="".join(chars),
        char_to_raw=tuple(char_to_raw),
        char_to_token=tuple(char_to_token),
        lines=lines,
        mode=mode,
        allowed_categories=allowed_categories,
    )


def _numeric_neighbor(text: str, index: int, direction: int) -> bool:
    """구분자를 건너뛴 가까운 문자가 숫자인지 확인한다."""
    cursor = index + direction
    while 0 <= cursor < len(text) and text[cursor] in " \t\r\n-./":
        cursor += direction
    return 0 <= cursor < len(text) and text[cursor].isdigit()


def _recover_numeric_format_view(
    view: SearchView,
    *,
    ocr_tolerant: bool,
) -> SearchView:
    """숫자 후보의 중복 구분자와 제한적인 OCR 혼동문자를 복구한다.

    문자 삭제·치환 시에도 남은 각 글자의 raw/token 매핑을 그대로 복사하므로
    탐지 결과를 원본 bbox로 되돌릴 수 있다.
    """
    chars: list[str] = []
    char_to_raw: list[int | None] = []
    char_to_token: list[int | None] = []
    for index, original in enumerate(view.text):
        char = original
        if (
            char in "-./"
            and chars
            and chars[-1] == char
        ):
            continue
        if ocr_tolerant and char in OCR_DIGIT_CONFUSIONS:
            if (
                _numeric_neighbor(view.text, index, -1)
                or _numeric_neighbor(view.text, index, 1)
            ):
                char = OCR_DIGIT_CONFUSIONS[char]
        chars.append(char)
        char_to_raw.append(view.char_to_raw[index])
        char_to_token.append(view.char_to_token[index])
    return SearchView(
        text="".join(chars),
        char_to_raw=tuple(char_to_raw),
        char_to_token=tuple(char_to_token),
        lines=view.lines,
        mode=(
            "format_ocr_tolerant"
            if ocr_tolerant
            else "format_recovered"
        ),
        allowed_categories=NUMBER_RECOVERY_CATEGORIES,
    )


def _token_raw_offsets(tokens: tuple[OcrToken, ...]) -> tuple[int, ...]:
    """한 줄의 raw_text 안에서 각 토큰이 시작하는 상대 위치를 계산한다."""
    offsets: list[int] = []
    offset = 0
    for index, token in enumerate(tokens):
        offsets.append(offset)
        offset += len(token.text)
        if index < len(tokens) - 1:
            offset += 1
    return tuple(offsets)


def _compact_field_text(text: str) -> str:
    return "".join(
        char for char in normalize_for_search(text)
        if char.isalnum()
    )


def _edit_distance_at_most_one(first: str, second: str) -> bool:
    """긴 주소 label의 OCR 한 글자 삽입·누락·오인만 허용한다."""
    if abs(len(first) - len(second)) > 1:
        return False
    if first == second:
        return True
    if len(first) == len(second):
        return sum(a != b for a, b in zip(first, second)) == 1
    short, long = (first, second) if len(first) < len(second) else (second, first)
    return any(long[:index] + long[index + 1:] == short for index in range(len(long)))


def _matches_address_label(
    candidate: str,
    field_labels: tuple[str, ...],
    *,
    allow_short_fuzzy: bool = False,
) -> bool:
    if candidate in field_labels:
        return True
    return any(
        (
            len(label) >= 6
            or (
                allow_short_fuzzy
                and len(label) >= 2
                # ``주소``처럼 짧은 label의 한 글자 치환을 허용하면
                # ``문서``도 주소로 오인된다. 짧은 label은 OCR token 중복·누락에
                # 해당하는 삽입/삭제만 허용하고 동일 길이 치환은 허용하지 않는다.
                and len(candidate) != len(label)
            )
        )
        and _edit_distance_at_most_one(candidate, label)
        for label in field_labels
    )


def _address_label_range(
    tokens: tuple[OcrToken, ...],
    field_labels: tuple[str, ...],
    *,
    allow_short_fuzzy: bool = False,
) -> tuple[int, int] | None:
    """주소 label을 이루는 token 범위를 찾고 긴 label을 우선한다."""
    matches: list[tuple[int, int, int]] = []
    max_label_length = max(map(len, field_labels)) + 1
    for start in range(len(tokens)):
        joined = ""
        for end in range(start, min(len(tokens), start + 5)):
            joined += _compact_field_text(tokens[end].text)
            if _matches_address_label(
                joined,
                field_labels,
                allow_short_fuzzy=allow_short_fuzzy,
            ):
                matches.append((len(joined), start, end + 1))
            if len(joined) > max_label_length:
                break
    if not matches:
        return None

    best_match = matches[0]
    best_priority = (best_match[0], -best_match[1])
    for candidate_match in matches[1:]:
        candidate_priority = (candidate_match[0], -candidate_match[1])
        if candidate_priority > best_priority:
            best_match = candidate_match
            best_priority = candidate_priority

    _, start, end = best_match
    return start, end


def _starts_address_stop_field(token: OcrToken) -> bool:
    compact = _compact_field_text(token.text)
    return any(
        compact.startswith(prefix)
        or 0 <= compact.find(prefix) <= 4
        for prefix in ADDRESS_STOP_FIELD_PREFIXES
    )


def _tokens_before_stop_field(tokens: tuple[OcrToken, ...]) -> tuple[OcrToken, ...]:
    for index, token in enumerate(tokens):
        if _starts_address_stop_field(token):
            return tokens[:index]
    return tokens


def _tokens_before_date_column(tokens: tuple[OcrToken, ...]) -> tuple[OcrToken, ...]:
    """표의 주소 셀 오른쪽 등록일자 열이 직렬화되기 전에 잘라낸다."""
    for index, token in enumerate(tokens):
        compact = "".join(
            char for char in normalize_for_search(token.text) if not char.isspace()
        )
        if ADDRESS_DATE_COLUMN_TOKEN.fullmatch(compact):
            return tokens[:index]
    return tokens


def _looks_like_address_continuation(tokens: tuple[OcrToken, ...]) -> bool:
    """다음 표 줄이 주소 상세의 연속인지 문자열 형태로 확인한다.

    특정 문서의 ``자동차검사`` 같은 필드명을 나열하지 않는다. 구두점 token을
    건너뛴 첫 값이 행정구역·도로명·건물명 또는 번지·동·층·호 형태일 때만 주소
    확장을 허용한다. 따라서 새로운 일반 필드가 뒤따라도 주소 형태가 아니면
    자동으로 중단된다.
    """
    for token in tokens:
        compact = "".join(
            char
            for char in normalize_for_search(token.text)
            if char.isalnum() or char in "-/"
        )
        if not compact:
            continue
        if ADDRESS_DATE_COLUMN_TOKEN.fullmatch(compact):
            return False
        return ADDRESS_CONTINUATION_TOKEN.fullmatch(compact) is not None
    return False


def build_address_field_search_view(
    lines: tuple[OcrLine, ...],
    raw_offset: int,
    *,
    settings: SearchViewSettings = DEFAULT_SEARCH_VIEW_SETTINGS,
) -> SearchView | None:
    """주소 label과 값 열에 정렬된 최대 세 줄을 공백 연결 탐지 보기로 만든다."""
    if not lines or not lines[0].tokens:
        return None
    label_range = _address_label_range(
        lines[0].tokens,
        settings.address_field_labels,
        allow_short_fuzzy=(
            settings.pattern_strictness == PatternStrictness.OCR_TOLERANT
        ),
    )
    first_tokens = lines[0].tokens
    # 주소 필드명이 없는 행을 지역명만으로 주소 필드라고 추정하지 않는다.
    # 일반 주소 탐지는 ko-pii와 AEGIS가 담당하고, 이 구조 보기는 명시적인
    # 기본 필드명 또는 관리자 추가 필드명이 있을 때만 만든다.
    if label_range is None:
        return None
    _, label_end = label_range
    value_start = label_end
    while value_start < len(first_tokens) and _is_separator_only(
        first_tokens[value_start].text
    ):
        value_start += 1

    # 표 OCR은 ``사용본거지`` 같은 항목명 셀과 주소 값 셀을 서로 다른
    # 줄로 반환할 수 있다. 첫 줄에 label만 있으면 바로 다음 줄을 값 행으로
    # 사용한다. 실제 주소 여부는 이후 행정구역+주소 숫자 검증에서 결정한다.
    label_only_row = (
        label_range is not None and value_start >= len(first_tokens)
    )
    if label_only_row:
        if len(lines) < 2:
            return None
        value_line = lines[1]
        first_remainder = value_line.tokens
        first_before_stop_field = _tokens_before_stop_field(first_remainder)
        stopped_by_named_field = len(first_before_stop_field) < len(first_remainder)
        first_value_tokens = _tokens_before_date_column(first_before_stop_field)
        if not first_value_tokens:
            return None
        first_line_raw_length = len(build_text_index(first_tokens).raw_text)
        view_raw_offset = raw_offset + first_line_raw_length + 1
        continuation_lines = () if stopped_by_named_field else lines[2:]
        value_line_id = value_line.line_id
    else:
        first_remainder = first_tokens[value_start:]
        first_before_stop_field = _tokens_before_stop_field(first_remainder)
        stopped_by_named_field = len(first_before_stop_field) < len(first_remainder)
        first_value_tokens = _tokens_before_date_column(first_before_stop_field)
        if not first_value_tokens:
            return None
        token_offsets = _token_raw_offsets(first_tokens)
        view_raw_offset = raw_offset + token_offsets[value_start]
        continuation_lines = () if stopped_by_named_field else lines[1:]
        value_line_id = lines[0].line_id

    selected_lines: list[OcrLine] = [
        OcrLine(line_id=value_line_id, tokens=first_value_tokens)
    ]
    value_anchor = first_value_tokens[0]
    previous_tokens = first_value_tokens

    # 같은 줄에서 이미 다음 필드가 시작됐다면 이후 OCR 줄은 이 주소 값의
    # 연속으로 볼 수 없고, 생략된 token 때문에 raw offset도 연속하지 않는다.
    for line in continuation_lines:
        continuation = _tokens_before_stop_field(line.tokens)
        if (
            len(continuation) < len(line.tokens)
            and len(continuation) <= 1
        ):
            break
        if not continuation:
            break
        if not _looks_like_address_continuation(continuation):
            break
        continuation_start = continuation[0]
        token_height = max(
            1,
            value_anchor.box.y2 - value_anchor.box.y1,
            continuation_start.box.y2 - continuation_start.box.y1,
        )
        previous_bottom = max(token.box.y2 for token in previous_tokens)
        vertical_gap = continuation_start.box.y1 - previous_bottom
        if vertical_gap > token_height * ADDRESS_CONTINUATION_VERTICAL_GAP_HEIGHTS:
            break
        previous_left = min(token.box.x1 for token in previous_tokens)
        previous_right = max(token.box.x2 for token in previous_tokens)
        continuation_left = min(token.box.x1 for token in continuation)
        continuation_right = max(token.box.x2 for token in continuation)
        overlaps_while_returning_to_left = (
            continuation_left <= previous_left
            and
            min(previous_right, continuation_right)
            > max(previous_left, continuation_left)
        )
        if (
            not overlaps_while_returning_to_left
            and abs(continuation_start.box.x1 - value_anchor.box.x1)
            > token_height * ADDRESS_CONTINUATION_X_TOLERANCE_HEIGHTS
        ):
            break
        selected_lines.append(OcrLine(line_id=line.line_id, tokens=continuation))
        previous_tokens = continuation
        if len(continuation) < len(line.tokens):
            break

    return build_search_view(
        tuple(selected_lines),
        view_raw_offset,
        mode="address_field",
        allowed_categories=ADDRESS_FIELD_CATEGORIES,
    )


def _person_context_ranges(
    tokens: tuple[OcrToken, ...],
    field_labels: tuple[str, ...],
) -> tuple[tuple[int, int], ...]:
    """필드 label의 시작/끝 token을 찾되 조사로 이어지는 일반 문장은 뺀다."""
    ranges: list[tuple[int, int]] = []
    for start in range(len(tokens)):
        joined = ""
        for end in range(start, min(len(tokens), start + 4)):
            joined += normalize_for_search(tokens[end].text)
            candidate = joined.lstrip(".,·ㆍ:;()[]{}<>-_●■□※@#*")
            found = False
            for keyword in field_labels:
                if candidate == keyword:
                    found = True
                elif candidate.startswith(keyword):
                    remainder = candidate[len(keyword):]
                    found = bool(remainder) and not remainder[0].isalnum()
                if found:
                    ranges.append((start, end))
                    break
            if found or len(joined) > max(map(len, field_labels)) + 2:
                break
    return tuple(dict.fromkeys(ranges))


def _is_separator_only(text: str) -> bool:
    normalized = normalize_for_search(text)
    return bool(normalized) and all(not char.isalnum() for char in normalized)


def _starts_other_field(text: str) -> bool:
    # OCR token이 `, 주민등록번호`처럼 구두점과 다음 필드를 한 박스에 넣어도
    # 필드 경계를 인식한다.
    normalized = normalize_for_search(text).lstrip(".,·ㆍ:;()[]{}<>-_")
    return any(normalized.startswith(prefix) for prefix in NON_PERSON_FIELD_PREFIXES)


def _build_person_context_views(
    line: OcrLine,
    raw_offset: int,
    *,
    max_tokens: int = 7,
    settings: SearchViewSettings = DEFAULT_SEARCH_VIEW_SETTINGS,
) -> tuple[SearchView, ...]:
    """이름 label부터 가까운 값 token까지만 간격/compact 두 보기로 만든다."""
    token_offsets = _token_raw_offsets(line.tokens)
    views: list[SearchView] = []
    for start, label_end in _person_context_ranges(
        line.tokens,
        settings.person_field_labels,
    ):
        end_max = min(len(line.tokens), label_end + 1 + max_tokens)
        # label 뒤 구분기호는 건너뛰되, 첫 값이 다른 개인정보 필드라면 이 label은
        # 실제 값이 없는 안내문으로 보고 확장하지 않는다.
        first_value = label_end + 1
        while first_value < len(line.tokens) and _is_separator_only(
            line.tokens[first_value].text
        ):
            first_value += 1
        if first_value < len(line.tokens) and _starts_other_field(
            line.tokens[first_value].text
        ):
            end_max = label_end + 1

        for index in range(first_value + 1, end_max):
            if _starts_other_field(line.tokens[index].text):
                end_max = index
                break

        # label만 있는 짧은 보기부터 다음 필드 직전까지 단계적으로 검사한다.
        # 한 번에 긴 구간을 compact하면 이름과 다음 label이 붙기 때문이다.
        for end in range(label_end + 1, end_max + 1):
            partial = OcrLine(line_id=line.line_id, tokens=line.tokens[start:end])
            for mode in ("token_spaced", "field_compact"):
                views.append(
                    build_search_view(
                        (partial,),
                        raw_offset + token_offsets[start],
                        mode=mode,
                        allowed_categories=PERSON_ONLY_CATEGORIES,
                    )
                )
    return tuple(views)


def build_line_search_views(
    line: OcrLine,
    raw_offset: int,
    *,
    max_label_tokens: int = 5,
    max_value_tokens: int = 10,
    settings: SearchViewSettings = DEFAULT_SEARCH_VIEW_SETTINGS,
) -> tuple[SearchView, ...]:
    """한 줄에 대한 토큰 간격·형식 결합·필드 구간 검색 보기를 만든다."""
    views: list[SearchView] = [
        build_search_view(
            (line,),
            raw_offset,
            mode="token_spaced",
            allowed_categories=TOKEN_SPACED_CATEGORIES,
        )
    ]
    if settings.pattern_strictness != PatternStrictness.EXACT:
        format_view = build_search_view(
            (line,),
            raw_offset,
            mode="format_compact",
            allowed_categories=FORMAT_COMPACT_CATEGORIES,
        )
        views.append(format_view)
        if settings.pattern_strictness in {
            PatternStrictness.RECOVERED,
            PatternStrictness.OCR_TOLERANT,
        }:
            views.append(_recover_numeric_format_view(
                format_view,
                ocr_tolerant=False,
            ))
        if settings.pattern_strictness == PatternStrictness.OCR_TOLERANT:
            views.append(_recover_numeric_format_view(
                format_view,
                ocr_tolerant=True,
            ))

    tokens = line.tokens
    separator_indexes = tuple(
        index
        for index, token in enumerate(tokens)
        if ":" in normalize_for_search(token.text)
    )
    token_offsets = _token_raw_offsets(tokens)
    searched_windows: set[tuple[int, int]] = set()
    for separator_index in separator_indexes:
        start_min = max(0, separator_index - max_label_tokens)
        end_max = min(len(tokens), separator_index + 1 + max_value_tokens)
        for start in range(start_min, separator_index):
            for end in range(separator_index + 2, end_max + 1):
                key = (start, end)
                if key in searched_windows:
                    continue
                searched_windows.add(key)
                partial_line = OcrLine(line_id=line.line_id, tokens=tokens[start:end])
                if settings.pattern_strictness != PatternStrictness.EXACT:
                    views.append(
                        build_search_view(
                            (partial_line,),
                            raw_offset + token_offsets[start],
                            mode="field_compact",
                            allowed_categories=FIELD_COMPACT_CATEGORIES,
                        )
                    )

    views.extend(
        _build_person_context_views(
            line,
            raw_offset,
            settings=settings,
        )
    )

    # 동일 텍스트뿐 아니라 원본 위치까지 같을 때만 중복 보기로 판단한다.
    unique: dict[
        tuple[str, tuple[int | None, ...], frozenset[str] | None],
        SearchView,
    ] = {}
    for view in views:
        key = (view.text, view.char_to_raw, view.allowed_categories)
        unique.setdefault(key, view)
    return tuple(unique.values())


def build_cross_line_search_views(
    first: OcrLine,
    second: OcrLine,
    raw_offset: int,
    *,
    settings: SearchViewSettings = DEFAULT_SEARCH_VIEW_SETTINGS,
) -> tuple[SearchView, ...]:
    """인접 두 줄은 줄바꿈을 유지한 간격 보기와 형식 결합 보기만 만든다."""
    lines = (first, second)
    views = [
        build_search_view(
            lines,
            raw_offset,
            mode="token_spaced",
            allowed_categories=TOKEN_SPACED_CATEGORIES,
        )
    ]
    if settings.pattern_strictness != PatternStrictness.EXACT:
        format_view = build_search_view(
            lines,
            raw_offset,
            mode="format_compact",
            allowed_categories=FORMAT_COMPACT_CATEGORIES,
        )
        views.append(format_view)
        if settings.pattern_strictness in {
            PatternStrictness.RECOVERED,
            PatternStrictness.OCR_TOLERANT,
        }:
            views.append(_recover_numeric_format_view(
                format_view,
                ocr_tolerant=False,
            ))
        if settings.pattern_strictness == PatternStrictness.OCR_TOLERANT:
            views.append(_recover_numeric_format_view(
                format_view,
                ocr_tolerant=True,
            ))
    return tuple(views)
