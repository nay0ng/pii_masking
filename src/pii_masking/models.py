"""OCR 파싱 및 전처리 단계에서 공유하는 데이터 모델."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import unicodedata


@dataclass(frozen=True)
class Box:
    """원본 또는 OCR 처리 이미지 위의 축 정렬 사각형 좌표."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        if self.x1 > self.x2 or self.y1 > self.y2:
            raise ValueError(f"잘못된 boundingBox 좌표입니다: {self}")


@dataclass(frozen=True)
class OcrToken:
    """OCR이 인식한 단어 하나와 해당 단어의 위치 정보."""

    # word_boxes 배열의 순번이다. 이후 문자열 span을 박스로 되돌릴 때 사용한다.
    token_id: int
    text: str
    confidence: float
    box: Box


@dataclass(frozen=True)
class OcrDocument:
    """OCR 응답 한 건을 애플리케이션에서 다루기 쉬운 형태로 변환한 객체."""

    image_path: Path
    # 문서 종류를 모르는 입력도 처리해야 하므로 선택값으로만 보관한다.
    document_type: str | None
    # OCR 서버가 제공한 참고용 전체 문자열이다. 좌표 매핑에는 tokens를 사용한다.
    full_text: str
    tokens: tuple[OcrToken, ...]
    # 서버가 응답에 실제 OCR 처리 이미지 크기를 제공하면 그 값을 우선 사용한다.
    ocr_image_width: int | None = None
    ocr_image_height: int | None = None
    # 구버전 table OCR 응답처럼 크기 메타데이터가 없을 때의 호환용 긴 변 크기.
    ocr_resize_target: int | None = None


@dataclass(frozen=True)
class OcrLine:
    """좌표상 같은 줄에 있는 것으로 판단한 OCR 토큰 묶음."""

    line_id: int
    tokens: tuple[OcrToken, ...]

    @property
    def token_ids(self) -> tuple[int, ...]:
        """줄을 구성하는 토큰 ID를 읽기 순서대로 반환한다."""
        return tuple(token.token_id for token in self.tokens)

    @property
    def text(self) -> str:
        """사람이 확인하기 쉽도록 토큰 사이에 공백을 넣은 줄 문자열."""
        return " ".join(token.text for token in self.tokens)

    @property
    def compact_text(self) -> str:
        """
        필드명 비교에 사용할 공백 없는 줄 문자열.

        예: '대 표 자 : 홍 길 동' 형태로 분리된 OCR 토큰도
        '대표자:홍길동'으로 비교할 수 있다.
        """
        return "".join(
            char
            for token in self.tokens
            for char in unicodedata.normalize("NFKC", token.text)
            if not char.isspace()
        )


def _unique_token_ids(values: tuple[int | None, ...]) -> list[int]:
    """문자별 토큰 ID에서 None과 중복을 제거하되 발견 순서는 유지한다."""
    result: list[int] = []
    seen: set[int] = set()
    for token_id in values:
        if token_id is not None and token_id not in seen:
            result.append(token_id)
            seen.add(token_id)
    return result


@dataclass(frozen=True)
class DocumentIndex:
    """
    탐지용 문자열과 OCR 토큰을 양방향으로 연결하는 인덱스.

    raw_text
        줄바꿈과 토큰 사이 공백을 유지한 문자열이다.
    normalized_text
        공백을 제거하고 문자 형태를 통일한 검색용 문자열이다.
    raw_char_to_token
        raw_text의 각 글자가 어느 OCR 토큰에서 왔는지 기록한다.
    normalized_char_to_raw
        normalized_text의 각 글자를 raw_text의 원래 위치로 되돌리는 표이다.
    normalized_char_to_token
        normalized_text의 각 글자를 OCR 토큰으로 바로 연결하는 표이다.
    """

    raw_text: str
    normalized_text: str
    raw_char_to_token: tuple[int | None, ...]
    normalized_char_to_raw: tuple[int, ...]
    normalized_char_to_token: tuple[int, ...]
    tokens: tuple[OcrToken, ...]

    def token_ids_for_raw_span(self, start: int, end: int) -> list[int]:
        """raw_text의 [start, end) 범위와 겹치는 token ID를 반환한다."""
        if start < 0 or end < start or end > len(self.raw_text):
            raise ValueError(f"잘못된 raw span입니다: [{start}, {end})")

        return _unique_token_ids(self.raw_char_to_token[start:end])

    def raw_span_for_normalized_span(self, start: int, end: int) -> tuple[int, int]:
        """normalized_text의 [start, end) 범위를 raw_text 범위로 변환한다."""
        if start < 0 or end <= start or end > len(self.normalized_text):
            raise ValueError(f"잘못된 normalized span입니다: [{start}, {end})")

        raw_indexes = self.normalized_char_to_raw[start:end]
        return min(raw_indexes), max(raw_indexes) + 1

    def token_ids_for_normalized_span(self, start: int, end: int) -> list[int]:
        """정규화 문자열의 범위와 연결된 token ID를 반환한다."""
        if start < 0 or end <= start or end > len(self.normalized_text):
            raise ValueError(f"잘못된 normalized span입니다: [{start}, {end})")

        return _unique_token_ids(self.normalized_char_to_token[start:end])


@dataclass(frozen=True)
class PreprocessedDocument:
    """개인정보 탐지 직전 단계까지 준비를 마친 문서."""

    document: OcrDocument
    # 실제로 마스킹할 원본 이미지 크기
    image_width: int
    image_height: int
    # OCR 서버가 좌표를 계산할 때 사용한 축소 이미지 크기
    ocr_image_width: int
    ocr_image_height: int
    # OCR 좌표를 원본 좌표로 복원할 때 곱하는 배율
    scale_x: float
    scale_y: float
    lines: tuple[OcrLine, ...]
    index: DocumentIndex
