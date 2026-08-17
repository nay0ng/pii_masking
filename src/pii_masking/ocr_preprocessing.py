"""OCR token을 실제 문서 좌표와 읽기 순서에 맞게 정리한다.

코드 검토 순서
1. ``normalize_for_search()``는 유니코드와 하이픈을 탐지하기 쉬운 문자로 바꾼다.
   원본 token 문자열 자체는 수정하지 않는다.
2. ``build_lines()``는 bbox의 세로 위치와 겹침을 이용해 token을 줄별로 묶고
   같은 줄에서는 x좌표 순서로 정렬한다.
3. ``build_text_index()``는 줄별 token을 ``raw_text``와 ``normalized_text``로
   연결하면서 각 문자가 어느 token에서 왔는지 인덱스를 저장한다.
4. ``preprocess_document()``가 좌표 배율 계산부터 위 결과 조립까지 총괄하여
   ``PreprocessedDocument``를 반환한다.

다음 단계인 ``search_views.py``는 이 결과로 탐지기 입력 문자열을 만들고,
``detector_engine.py``는 저장된 문자 위치를 token ID와 bbox로 역매핑한다.
"""

from __future__ import annotations

import math
import statistics
import unicodedata

from PIL import Image

from .models import (
    Box,
    DocumentIndex,
    OcrDocument,
    OcrLine,
    OcrToken,
    PreprocessedDocument,
)


NORMALIZED_DASHES = {
    "\u2010",  # hyphen
    "\u2011",  # non-breaking hyphen
    "\u2012",  # figure dash
    "\u2013",  # en dash
    "\u2014",  # em dash
    "\u2212",  # minus sign
    "\ufe63",  # small hyphen-minus
    "\uff0d",  # full-width hyphen-minus
}


def _normalized_characters(char: str):
    """원문 글자 하나를 검색용 문자 0개 이상으로 변환한다."""
    for normalized_char in unicodedata.normalize("NFKC", char):
        if normalized_char.isspace():
            continue
        if normalized_char in NORMALIZED_DASHES:
            normalized_char = "-"
        yield normalized_char


def normalize_for_search(text: str) -> str:
    """입력 문자열을 DocumentIndex.normalized_text와 같은 규칙으로 정규화한다."""
    return "".join(
        normalized_char
        for char in text
        for normalized_char in _normalized_characters(char)
    )


def _vertical_overlap_ratio(first: Box, second: Box) -> float:
    """
    두 박스가 세로 방향으로 얼마나 겹치는지 계산한다.

    작은 박스의 높이를 기준으로 계산하므로 글자 높이가 조금 달라도 같은
    줄에 있는 토큰을 안정적으로 묶을 수 있다.
    """
    overlap = max(0, min(first.y2, second.y2) - max(first.y1, second.y1))
    first_height = max(1, first.y2 - first.y1)
    second_height = max(1, second.y2 - second.y1)
    return overlap / min(first_height, second_height)


def _line_bounds(tokens: list[OcrToken]) -> Box:
    """현재 줄의 대표 중심선과 글자 높이를 나타내는 사각형을 만든다.

    모든 token의 y 최소/최대를 그대로 사용하면, 서로 다른 두 행을 걸치는 token
    하나가 줄 박스를 키우고 그 다음 행까지 연쇄적으로 합치는 문제가 생긴다.
    대표값에는 중앙값을 사용하여 표·다단 문서의 인접 행이 합쳐지는 것을 막는다.
    """
    centers = [(token.box.y1 + token.box.y2) / 2 for token in tokens]
    heights = [max(1, token.box.y2 - token.box.y1) for token in tokens]
    center = statistics.median(centers)
    height = statistics.median(heights)
    return Box(
        x1=min(token.box.x1 for token in tokens),
        y1=math.floor(center - height / 2),
        x2=max(token.box.x2 for token in tokens),
        y2=math.ceil(center + height / 2),
    )


def build_lines(
    tokens: tuple[OcrToken, ...],
    vertical_overlap_threshold: float = 0.6,
) -> tuple[OcrLine, ...]:
    """OCR 토큰을 세로 겹침률로 같은 줄에 묶고 읽기 순서로 정렬한다."""
    if not 0 < vertical_overlap_threshold <= 1:
        raise ValueError("vertical_overlap_threshold는 0보다 크고 1 이하여야 합니다.")

    groups: list[list[OcrToken]] = []

    # OCR 배열 순서를 그대로 믿지 않고 위→아래, 왼쪽→오른쪽 후보 순서로 정렬한다.
    candidate_records: list[tuple[float, int, int, OcrToken]] = []
    for token in tokens:
        vertical_center = (token.box.y1 + token.box.y2) / 2
        candidate_records.append(
            (vertical_center, token.box.x1, token.token_id, token)
        )
    candidate_records.sort()
    candidates = [record[3] for record in candidate_records]

    for token in candidates:
        best_group: list[OcrToken] | None = None
        best_overlap = 0.0
        for group in groups:
            # 여러 줄과 겹칠 때는 세로 겹침률이 가장 큰 줄을 선택한다.
            overlap = _vertical_overlap_ratio(token.box, _line_bounds(group))
            if overlap >= vertical_overlap_threshold and overlap > best_overlap:
                best_group = group
                best_overlap = overlap

        if best_group is None:
            # 기존 줄과 충분히 겹치지 않는 토큰은 새 줄의 시작이다.
            groups.append([token])
        else:
            best_group.append(token)

    # 각 줄 내부를 왼쪽→오른쪽 읽기 순서로 정렬한다.
    for group_index, group in enumerate(groups):
        token_records = [
            (token.box.x1, token.box.y1, token.token_id, token)
            for token in group
        ]
        token_records.sort()
        groups[group_index] = [record[3] for record in token_records]

    # 줄 전체는 위→아래 순서로 정렬한다.
    group_records: list[tuple[int, int, int, list[OcrToken]]] = []
    for group_index, group in enumerate(groups):
        top = min(token.box.y1 for token in group)
        left = min(token.box.x1 for token in group)
        group_records.append((top, left, group_index, group))
    group_records.sort()
    groups = [record[3] for record in group_records]

    return tuple(
        OcrLine(line_id=line_id, tokens=tuple(group))
        for line_id, group in enumerate(groups)
    )


def build_text_index(
    tokens: tuple[OcrToken, ...],
    lines: tuple[OcrLine, ...] | None = None,
) -> DocumentIndex:
    """
    OCR 토큰을 문자열로 연결하고, 각 글자의 원래 token ID를 기록한다.

    raw_text는 토큰 사이에 공백 하나를 넣고 줄 사이에는 줄바꿈을 넣는다.
    normalized_text는 NFKC 정규화 후 공백을 제거하고 대시를 '-'로 통일한다.
    """
    raw_chars: list[str] = []
    raw_char_to_token: list[int | None] = []

    # lines가 있으면 좌표로 복원한 읽기 순서를 사용하고, 없으면 입력 순서를 쓴다.
    token_rows = (
        tuple(line.tokens for line in lines)
        if lines is not None
        else (tokens,)
    )
    for row_index, row in enumerate(token_rows):
        if row_index:
            raw_chars.append("\n")
            raw_char_to_token.append(None)

        for token_index, token in enumerate(row):
            if token_index:
                raw_chars.append(" ")
                raw_char_to_token.append(None)

            # 토큰 안의 모든 글자를 동일한 token_id에 연결한다.
            raw_chars.extend(token.text)
            raw_char_to_token.extend([token.token_id] * len(token.text))

    raw_text = "".join(raw_chars)
    normalized_chars: list[str] = []
    normalized_char_to_raw: list[int] = []
    normalized_char_to_token: list[int] = []

    for raw_index, char in enumerate(raw_text):
        for normalized_char in _normalized_characters(char):
            normalized_chars.append(normalized_char)
            # 정규화된 글자 하나마다 원문에서 유래한 위치를 기억한다.
            normalized_char_to_raw.append(raw_index)
            token_id = raw_char_to_token[raw_index]
            if token_id is None:
                raise ValueError("정규화 문자에 연결할 OCR 토큰이 없습니다.")
            # 정규화 문자열에서 찾은 span을 token ID로 바로 변환할 수 있게 한다.
            normalized_char_to_token.append(token_id)

    return DocumentIndex(
        raw_text=raw_text,
        normalized_text="".join(normalized_chars),
        raw_char_to_token=tuple(raw_char_to_token),
        normalized_char_to_raw=tuple(normalized_char_to_raw),
        normalized_char_to_token=tuple(normalized_char_to_token),
        tokens=tokens,
    )


def preprocess_document(
    document: OcrDocument,
    vertical_overlap_threshold: float = 0.6,
    server_resize_target: int = 1280,
) -> PreprocessedDocument:
    """
    탐지 전에 필요한 좌표 검증, 줄 구성, 문자열 인덱싱을 한 번에 수행한다.

    현재 OCR 서버는 긴 변을 server_resize_target 크기로 축소한 뒤 OCR을 수행한다.
    이 함수가 끝난 뒤 index.tokens의 box는 모두 원본 이미지 좌표이다.
    """
    if not document.image_path.exists():
        raise FileNotFoundError(f"원본 이미지를 찾을 수 없습니다: {document.image_path}")

    with Image.open(document.image_path) as image:
        image_width, image_height = image.size

    # 응답에 실제 처리 이미지 크기가 있으면 하드코딩된 서버 크기보다 우선한다.
    if document.ocr_image_width and document.ocr_image_height:
        ocr_image_width = document.ocr_image_width
        ocr_image_height = document.ocr_image_height
    else:
        resize_target = document.ocr_resize_target or server_resize_target
        # table OCR 서버는 작은 이미지도 긴 변 960px로 확대한다.
        # 일반 OCR의 기존 1280px 규칙은 큰 이미지만 축소하므로 동작을 구분한다.
        should_resize = (
            document.ocr_resize_target is not None
            or max(image_width, image_height) >= resize_target
        )
        if should_resize:
            server_scale = resize_target / max(image_width, image_height)
            if image_height > image_width:
                ocr_image_width = int(image_width * server_scale)
                ocr_image_height = resize_target
            else:
                ocr_image_width = resize_target
                ocr_image_height = int(image_height * server_scale)
        else:
            ocr_image_width = image_width
            ocr_image_height = image_height

    scale_x = image_width / ocr_image_width
    scale_y = image_height / ocr_image_height
    mapped_tokens: list[OcrToken] = []
    for token in document.tokens:
        # 좌표 복원 전에 OCR 응답 좌표가 OCR 처리 이미지 안에 있는지 검증한다.
        if (
            token.box.x1 < 0
            or token.box.y1 < 0
            or token.box.x2 > ocr_image_width
            or token.box.y2 > ocr_image_height
        ):
            raise ValueError(
                f"token {token.token_id}의 OCR 좌표가 OCR 처리 이미지 범위를 "
                f"벗어났습니다. ocr_image={ocr_image_width}x{ocr_image_height}, "
                f"box={token.box}"
            )

        mapped_tokens.append(
            OcrToken(
                token_id=token.token_id,
                text=token.text,
                confidence=token.confidence,
                box=Box(
                    # 시작점은 floor, 끝점은 ceil을 사용해 글자 영역이 잘리지 않게 한다.
                    x1=max(0, math.floor(token.box.x1 * scale_x)),
                    y1=max(0, math.floor(token.box.y1 * scale_y)),
                    x2=min(image_width, math.ceil(token.box.x2 * scale_x)),
                    y2=min(image_height, math.ceil(token.box.y2 * scale_y)),
                ),
            )
        )

    lines = build_lines(
        tuple(mapped_tokens),
        vertical_overlap_threshold=vertical_overlap_threshold,
    )

    # 줄 재구성 과정에서 토큰이 사라지거나 두 줄에 중복되지 않았는지 확인한다.
    indexed_token_ids = [
        token.token_id
        for line in lines
        for token in line.tokens
    ]
    expected_token_ids = [token.token_id for token in document.tokens]
    if sorted(indexed_token_ids) != sorted(expected_token_ids):
        raise ValueError("줄 구성 과정에서 OCR 토큰이 누락되거나 중복되었습니다.")

    # 원본 좌표 토큰과 좌표 기반 줄 순서로 탐지용 문자열 인덱스를 만든다.
    index = build_text_index(tuple(mapped_tokens), lines=lines)
    return PreprocessedDocument(
        document=document,
        image_width=image_width,
        image_height=image_height,
        ocr_image_width=ocr_image_width,
        ocr_image_height=ocr_image_height,
        scale_x=scale_x,
        scale_y=scale_y,
        lines=lines,
        index=index,
    )
