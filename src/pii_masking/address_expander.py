"""탐지된 주소 범위를 OCR token과 bbox 기준으로 늘린다.

ko-pii가 주소의 행정구역까지만 반환한 경우 같은 줄 또는 바로 다음 줄의
동·층·호·건물명 token을 확인한다. 확장한 범위도 원본 token ID와 bbox를 그대로
사용하므로 마스킹 좌표를 다시 계산하지 않는다.

코드 검토는 ``extend_address_spans()``부터 시작한다. 이 함수가 이미 탐지된 주소
span마다 ``_address_extension_tokens()``를 호출하고, 같은 줄 또는 다음 줄의 token이
주소의 연속인지 확인한다. 이 파일은 새로운 주소를 처음부터 탐지하는 규칙이 아니라
이미 발견된 주소의 끝 범위를 보완하는 후처리다.
"""

from __future__ import annotations

from dataclasses import replace
import re

from .detector_engine import PiiSpan
from .models import OcrLine, OcrToken, PreprocessedDocument
from .ocr_preprocessing import normalize_for_search


_ADDRESS_DETAIL_TOKEN = re.compile(
    r"^(?:"
    r"\d+(?:[-/]\d+)?(?:번지)?|"
    # OCR은 ``201호(평촌동,``처럼 호수와 괄호 상세주소를 한 token으로
    # 묶기도 한다. 단위로 시작하는 경우에는 뒤의 동·건물명도 같은 주소
    # token으로 인정한다.
    r"\d+(?:동|층|호)(?:[가-힣A-Za-z0-9].*)?|"
    r"[가-힣A-Za-z0-9]{1,20}(?:아파트|빌딩|타워|센터|오피스텔|상가|플라자)"
    r")$"
)
_DATE_TOKEN = re.compile(r"^(?:19|20)\d{2}(?:[-./]\d{1,2}){1,2}$")


def _compact(text: str) -> str:
    return "".join(
        char
        for char in normalize_for_search(text)
        if not char.isspace() and char not in ",.;:·ㆍ()[]{}"
    )


def _is_address_detail_start(token: OcrToken) -> bool:
    return _ADDRESS_DETAIL_TOKEN.fullmatch(_compact(token.text)) is not None


def _line_by_id(document: PreprocessedDocument) -> dict[int, OcrLine]:
    return {line.line_id: line for line in document.lines}


def _raw_indexes_for_tokens(
    document: PreprocessedDocument,
    token_ids: set[int],
) -> list[int]:
    return [
        index
        for index, token_id in enumerate(document.index.raw_char_to_token)
        if token_id in token_ids
    ]


def _next_line_is_close(
    previous_tokens: tuple[OcrToken, ...],
    next_tokens: tuple[OcrToken, ...],
) -> bool:
    previous_bottom = max(token.box.y2 for token in previous_tokens)
    next_top = min(token.box.y1 for token in next_tokens)
    height = max(
        1,
        max(token.box.y2 - token.box.y1 for token in previous_tokens),
        max(token.box.y2 - token.box.y1 for token in next_tokens),
    )
    if next_top - previous_bottom > height * 1.5:
        return False
    previous_left = min(token.box.x1 for token in previous_tokens)
    previous_right = max(token.box.x2 for token in previous_tokens)
    next_left = min(token.box.x1 for token in next_tokens)
    next_right = max(token.box.x2 for token in next_tokens)
    overlaps_while_returning_to_left = (
        next_left <= previous_left
        and
        min(previous_right, next_right) > max(previous_left, next_left)
    )
    return (
        overlaps_while_returning_to_left
        or abs(next_left - previous_left) <= height * 4
    )


def _address_extension_tokens(
    document: PreprocessedDocument,
    span: PiiSpan,
    *,
    max_lines: int = 3,
) -> tuple[OcrToken, ...]:
    line_lookup = _line_by_id(document)
    token_lookup = {token.token_id: token for token in document.index.tokens}
    selected_ids = list(span.token_ids)
    selected_set = set(selected_ids)
    if not span.line_ids:
        return ()

    current_line_id = span.line_ids[-1]
    current_line = line_lookup[current_line_id]
    selected_in_line = [
        token for token in current_line.tokens if token.token_id in selected_set
    ]
    if not selected_in_line:
        return ()

    last_index = max(
        index
        for index, token in enumerate(current_line.tokens)
        if token.token_id in selected_set
    )
    same_line_tail = current_line.tokens[last_index + 1:]
    if same_line_tail and _is_address_detail_start(same_line_tail[0]):
        for token in same_line_tail:
            if _DATE_TOKEN.fullmatch(_compact(token.text)):
                break
            selected_ids.append(token.token_id)
            selected_set.add(token.token_id)
        selected_in_line = [
            token for token in current_line.tokens if token.token_id in selected_set
        ]

    previous_tokens = tuple(selected_in_line)
    followed_lines = 1
    while followed_lines < max_lines:
        next_line = line_lookup.get(current_line_id + 1)
        if next_line is None or not next_line.tokens:
            break
        if not _is_address_detail_start(next_line.tokens[0]):
            break
        if not _next_line_is_close(previous_tokens, next_line.tokens):
            break
        for token in next_line.tokens:
            if _DATE_TOKEN.fullmatch(_compact(token.text)):
                break
            selected_ids.append(token.token_id)
            selected_set.add(token.token_id)
        previous_tokens = next_line.tokens
        current_line_id += 1
        followed_lines += 1

    return tuple(
        token_lookup[token_id]
        for token_id in selected_ids
        if token_id in token_lookup
    )


def extend_address_spans(
    document: PreprocessedDocument,
    spans: tuple[PiiSpan, ...],
) -> tuple[PiiSpan, ...]:
    """Extend a confirmed address through aligned lot/building/unit tokens."""
    result: list[PiiSpan] = []
    for span in spans:
        if span.entity_type != "address":
            result.append(span)
            continue
        tokens = _address_extension_tokens(document, span)
        if len(tokens) <= len(span.token_ids):
            result.append(span)
            continue
        token_ids = {token.token_id for token in tokens}
        raw_indexes = _raw_indexes_for_tokens(document, token_ids)
        line_ids = tuple(
            line.line_id
            for line in document.lines
            if any(token.token_id in token_ids for token in line.tokens)
        )
        text = " ".join(token.text.strip() for token in tokens).strip()
        result.append(replace(
            span,
            text=text,
            normalized_text=normalize_for_search(text),
            line_ids=line_ids,
            raw_start=min(raw_indexes),
            raw_end=max(raw_indexes) + 1,
            token_ids=tuple(token.token_id for token in tokens),
            boxes=tuple(token.box for token in tokens),
            ocr_confidence=min(token.confidence for token in tokens),
            evidence=tuple(dict.fromkeys(
                (*span.evidence, "postprocess:address_unit_continuation")
            )),
        ))
    return tuple(result)
