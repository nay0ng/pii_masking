"""표 헤더와 바로 아래 값의 x좌표를 연결해 제한된 SearchView를 만든다.

일반적인 2줄 SearchView는 두 줄의 문자열을 이어 줄 뿐, 어느 값이 어느 표
열에 속하는지는 알지 못한다. 이 모듈은 한 줄에서 여러 개인정보 필드 헤더를
찾고, 다음 행에서 같은 x축 범위에 놓인 값만 선택한다.

선택한 ``필드명 + 값``은 개인정보를 여기서 확정하지 않고 SearchView로만
만든다. 실제 형식 검증은 ko-pii가, 이름·주소 같은 문맥 검증은 AEGIS가 맡는다.
따라서 특정 문서나 특정 실제 값에 맞춘 마스킹 규칙은 이 파일에 두지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import OcrLine, OcrToken, PreprocessedDocument
from .ocr_preprocessing import normalize_for_search
from .search_views import FORMAT_COMPACT_CATEGORIES, SearchView, SearchViewSettings


_FIXED_FIELD_LABELS = {
    "email_address": (
        "이메일", "이메일주소", "전자우편", "전자우편주소", "결과수신이메일",
        "전자세금계산서수신이메일",
        "전자세금계산서수신E-MAIL", "E-MAIL", "EMAIL",
    ),
    "phone_number": (
        "전화", "전화번호", "연락처", "휴대전화", "휴대전화번호", "휴대폰",
        "휴대폰번호", "핸드폰", "핸드폰번호", "자택전화", "자택전화번호",
        "직장전화", "직장전화번호", "자택연락처", "우선연락처",
        "담당자연락처", "대표자연락처", "사업장연락처", "고객연락처",
        "전화휴대폰번호", "직장번호", "법인휴대폰",
        "TEL", "PHONE", "MOBILE",
    ),
    "date": ("생년월일", "출생연월일", "출생일", "생일"),
    "account_card_number": (
        "계좌", "계좌번호", "결제계좌", "입금계좌", "사용자계좌번호",
        "가입자계좌번호", "퇴직연금계좌번호", "IRP계좌번호", "지급계좌번호",
        "수령계좌번호", "카드번호", "신용카드번호",
    ),
    "resident_registration_number": (
        "주민등록번호", "주민번호", "주민(법인)등록번호", "가입자주민번호",
        "가입자주민등록번호", "대표자주민번호", "가입자고객번호",
        "실명확인번호",
    ),
    "foreigner_registration_number": (
        "외국인등록번호", "외국인번호", "가입자고객번호", "실명확인번호",
    ),
    "passport_number": ("여권번호", "실명확인번호"),
    "driver_license_number": (
        "운전면허번호", "면허번호", "실명확인번호",
    ),
    "business_registration_number": ("사업자등록번호", "사업자번호"),
    "corporate_registration_number": (
        "법인등록번호", "법인번호", "주민(법인)등록번호",
    ),
    "health_insurance_number": ("건강보험증번호", "건강보험번호"),
    "vehicle_plate_number": ("차량번호", "자동차등록번호"),
    "postal_code": ("우편번호",),
    "employee_number": ("사번", "직원번호"),
}

_NEUTRAL_TABLE_HEADERS = frozenset({
    "구분", "관계", "성별", "분", "순번", "순위번호", "사항란", "등록일자", "접수번호",
})


@dataclass(frozen=True)
class _FieldHeader:
    """헤더 줄에서 찾은 필드명과 그 필드가 담당할 개인정보 유형."""

    categories: frozenset[str]
    label: str
    start: int
    end: int
    center_x: float
    height: int


def _normalized_label(text: str) -> str:
    """공백과 구두점을 제외해 분리 OCR된 필드명도 같은 값으로 비교한다."""
    normalized = "".join(
        char for char in normalize_for_search(text)
        if char.isalnum()
    )
    return normalized.upper()


def _field_labels_by_category(settings: SearchViewSettings) -> dict[str, tuple[str, ...]]:
    """고정 필드명과 실행 시 전달된 이름·주소 필드명을 하나로 합친다."""
    labels = dict(_FIXED_FIELD_LABELS)
    labels["person_name"] = settings.person_field_labels
    labels["address"] = settings.address_field_labels
    return labels


def _tokens_text(tokens: tuple[OcrToken, ...]) -> str:
    return _normalized_label("".join(token.text for token in tokens))


def _token_center_x(token: OcrToken) -> float:
    return (token.box.x1 + token.box.x2) / 2


def _find_field_headers(
    line: OcrLine,
    settings: SearchViewSettings,
) -> tuple[_FieldHeader, ...]:
    """한 줄에서 최대 네 token으로 분리된 개인정보 필드명을 찾는다."""
    label_to_categories: dict[str, set[str]] = {}
    for category, labels in _field_labels_by_category(settings).items():
        for label in labels:
            normalized = _normalized_label(label)
            if normalized:
                label_to_categories.setdefault(normalized, set()).add(category)

    headers: list[_FieldHeader] = []
    token_index = 0
    while token_index < len(line.tokens):
        matched_header: _FieldHeader | None = None
        max_end = min(len(line.tokens), token_index + 4)
        for end in range(max_end, token_index, -1):
            field_tokens = line.tokens[token_index:end]
            label = _tokens_text(field_tokens)
            categories = label_to_categories.get(label)
            if categories is None:
                continue
            left = min(token.box.x1 for token in field_tokens)
            right = max(token.box.x2 for token in field_tokens)
            height = max(
                1,
                max(token.box.y2 for token in field_tokens)
                - min(token.box.y1 for token in field_tokens),
            )
            matched_header = _FieldHeader(
                categories=frozenset(categories),
                label=label,
                start=token_index,
                end=end,
                center_x=(left + right) / 2,
                height=height,
            )
            break

        if matched_header is None:
            token_index += 1
            continue
        headers.append(matched_header)
        token_index = matched_header.end

    # 개인정보 헤더가 하나뿐이어도 ``구분`` 같은 별도 열 헤더가 있으면 표다.
    neutral_count = sum(
        _tokens_text((token,)) in _NEUTRAL_TABLE_HEADERS
        for token in line.tokens
    )
    if len(headers) < 2 and neutral_count == 0:
        return ()
    return tuple(sorted(headers, key=lambda header: header.center_x))


def _column_boundaries(
    header_line: OcrLine,
    headers: tuple[_FieldHeader, ...],
    index: int,
) -> tuple[float, float]:
    """현재 필드명 바로 앞·뒤 token 중심의 중간점을 열 경계로 사용한다."""
    header = headers[index]
    if header.start == 0:
        left = float("-inf")
    else:
        previous_center = _token_center_x(header_line.tokens[header.start - 1])
        left = (previous_center + header.center_x) / 2
    if header.end == len(header_line.tokens):
        right = float("inf")
    else:
        next_center = _token_center_x(header_line.tokens[header.end])
        right = (header.center_x + next_center) / 2
    return left, right


def _tokens_in_column(
    header_line: OcrLine,
    value_line: OcrLine,
    headers: tuple[_FieldHeader, ...],
    header_index: int,
    row_distance: int,
) -> tuple[OcrToken, ...]:
    """바로 아래 행에서 현재 헤더 열의 x축 범위에 들어오는 token을 고른다."""
    header = headers[header_index]
    header_bottom = max(
        token.box.y2 for token in header_line.tokens[header.start:header.end]
    )
    value_top = min(token.box.y1 for token in value_line.tokens)
    if value_top - header_bottom > header.height * 3 * row_distance:
        return ()

    left, right = _column_boundaries(header_line, headers, header_index)
    return tuple(
        token for token in value_line.tokens
        if left <= _token_center_x(token) <= right
    )


def _normalized_token_chars(
    document: PreprocessedDocument,
) -> dict[int, tuple[tuple[str, int], ...]]:
    """token별 검색 문자와 문서 raw 위치를 한 번 계산한다."""
    records: dict[int, list[tuple[str, int]]] = {}
    index = document.index
    for char, raw_index, token_id in zip(
        index.normalized_text,
        index.normalized_char_to_raw,
        index.normalized_char_to_token,
    ):
        records.setdefault(token_id, []).append((char, raw_index))
    return {token_id: tuple(values) for token_id, values in records.items()}


def _append_tokens(
    chars: list[str],
    raw_indexes: list[int | None],
    token_ids: list[int | None],
    tokens: tuple[OcrToken, ...],
    token_characters: dict[int, tuple[tuple[str, int], ...]],
    *,
    separator: str,
) -> None:
    """선택한 token의 문자와 실제 raw/token 위치를 SearchView 배열에 넣는다."""
    for token_index, token in enumerate(tokens):
        if token_index and separator:
            chars.append(separator)
            raw_indexes.append(None)
            token_ids.append(None)
        for char, raw_index in token_characters.get(token.token_id, ()):
            chars.append(char)
            raw_indexes.append(raw_index)
            token_ids.append(token.token_id)


def _build_table_field_view(
    header_line: OcrLine,
    value_line: OcrLine,
    header: _FieldHeader,
    value_tokens: tuple[OcrToken, ...],
    token_characters: dict[int, tuple[tuple[str, int], ...]],
    *,
    compact_value: bool,
) -> SearchView:
    """실제 raw 위치를 유지한 ``필드명: 값`` 전용 SearchView를 만든다."""
    chars: list[str] = []
    raw_indexes: list[int | None] = []
    token_ids: list[int | None] = []

    header_tokens = header_line.tokens[header.start:header.end]
    _append_tokens(
        chars, raw_indexes, token_ids, header_tokens, token_characters, separator="",
    )
    chars.extend((":", "\n"))
    raw_indexes.extend((None, None))
    token_ids.extend((None, None))
    _append_tokens(
        chars,
        raw_indexes,
        token_ids,
        value_tokens,
        token_characters,
        separator="" if compact_value else " ",
    )

    mode = "table_field_compact" if compact_value else "table_field"
    return SearchView(
        text="".join(chars),
        char_to_raw=tuple(raw_indexes),
        char_to_token=tuple(token_ids),
        lines=(header_line, value_line),
        mode=mode,
        allowed_categories=header.categories,
    )


def build_table_field_search_views(
    document: PreprocessedDocument,
    settings: SearchViewSettings,
) -> tuple[SearchView, ...]:
    """표 헤더와 같은 열의 다음 행 값을 유형 제한 SearchView로 반환한다."""
    token_characters = _normalized_token_chars(document)
    views: list[SearchView] = []
    for line_index, header_line in enumerate(document.lines):
        headers = _find_field_headers(header_line, settings)
        if not headers:
            continue
        for row_distance in range(1, settings.max_adjacent_lines + 1):
            value_index = line_index + row_distance
            if value_index >= len(document.lines):
                break
            value_line = document.lines[value_index]
            if _find_field_headers(value_line, settings):
                break
            for header_index, header in enumerate(headers):
                value_tokens = _tokens_in_column(
                    header_line, value_line, headers, header_index, row_distance,
                )
                if not value_tokens:
                    continue
                views.append(_build_table_field_view(
                    header_line,
                    value_line,
                    header,
                    value_tokens,
                    token_characters,
                    compact_value=False,
                ))
                if header.categories & FORMAT_COMPACT_CATEGORIES:
                    views.append(_build_table_field_view(
                        header_line,
                        value_line,
                        header,
                        value_tokens,
                        token_characters,
                        compact_value=True,
                    ))
    return tuple(views)
