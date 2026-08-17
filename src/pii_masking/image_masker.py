"""최종 PiiSpan의 bbox를 원본 이미지에 검은 사각형으로 표시한다.

동일한 bbox는 한 번만 그리며, 설정한 여백을 적용한 뒤 이미지 경계를 넘지 않게
좌표를 제한한다. 이 파일은 개인정보를 탐지하지 않고 전달받은 좌표만 사용한다.

코드 검토 순서
1. ``render_masked_image()``가 PiiSpan의 중복 bbox를 제거하고 사각형을 그린다.
2. ``mask_document_image()``가 원본 이미지를 열고 결과 파일 저장까지 수행한다.
3. ``MaskedImageResult``에 저장 위치와 span·bbox 개수를 기록한다.

여기까지 도달한 span은 탐지와 정책 판단이 끝난 결과다. 이 파일에서는 문자열,
신뢰도와 체크섬을 다시 판단하지 않는다.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from .detector_engine import PiiSpan
from .models import Box, PreprocessedDocument
from .paths import IMAGE_DATA_DIR


@dataclass(frozen=True)
class MaskedImageResult:
    """이미지 한 장의 마스킹 저장 결과."""

    output_path: Path
    span_count: int
    box_count: int


def build_masked_output_path(
    document_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """원본의 상대 폴더 구조를 유지한 마스킹 이미지 경로를 만든다."""
    source = Path(document_path)
    root = Path(output_dir)

    # 안전하지 않은 상위 폴더(..) 또는 절대 경로는 출력 루트 아래에 그대로
    # 재현하지 않는다. 일반적인 상대 경로일 때만 원본 폴더 구조를 보존한다.
    if source.is_absolute():
        try:
            relative_parent = source.relative_to(IMAGE_DATA_DIR).parent
        except ValueError:
            relative_parent = Path()
    elif ".." in source.parts:
        relative_parent = Path()
    else:
        source_parts = source.parts
        if (
            len(source_parts) >= 2
            and source_parts[0].casefold() == "data"
            and source_parts[1].casefold() == "images"
        ):
            relative_parent = Path(*source_parts[2:-1])
        else:
            relative_parent = source.parent

    suffix = source.suffix or ".png"
    filename = f"{source.stem}_masked{suffix}"
    return root / relative_parent / filename


def _clamped_rectangle(
    box: Box,
    image_width: int,
    image_height: int,
    padding: int,
) -> tuple[int, int, int, int] | None:
    """bbox에 여백을 더하고 이미지 경계 안으로 제한한다."""
    x1 = max(0, box.x1 - padding)
    y1 = max(0, box.y1 - padding)
    x2 = min(image_width - 1, box.x2 + padding)
    y2 = min(image_height - 1, box.y2 + padding)
    if x1 > x2 or y1 > y2:
        return None
    return x1, y1, x2, y2


def render_masked_image(
    document: PreprocessedDocument,
    spans: Iterable[PiiSpan],
    *,
    padding: int = 2,
) -> tuple[Image.Image, int, int]:
    """마스킹 이미지를 메모리에서 만들고 ``(이미지, span 수, box 수)``를 반환한다.

    파일 저장이 필요 없는 Streamlit 미리보기와 실제 파일 저장 코드가 동일한
    좌표 계산을 사용하도록 분리한 함수다.
    """
    if padding < 0:
        raise ValueError("padding은 0 이상이어야 합니다.")

    source_path = document.document.image_path
    if not source_path.exists():
        raise FileNotFoundError(f"원본 이미지를 찾을 수 없습니다: {source_path}")

    span_items = tuple(spans)
    unique_boxes = tuple(dict.fromkeys(box for span in span_items for box in span.boxes))

    with Image.open(source_path) as source:
        image = source.convert("RGB")

    if image.size != (document.image_width, document.image_height):
        raise ValueError(
            "전처리에 사용한 원본 이미지 크기와 현재 이미지 크기가 다릅니다. "
            f"preprocessed={document.image_width}x{document.image_height}, "
            f"actual={image.width}x{image.height}"
        )

    draw = ImageDraw.Draw(image)
    drawn_count = 0
    for box in unique_boxes:
        rectangle = _clamped_rectangle(
            box,
            image_width=image.width,
            image_height=image.height,
            padding=padding,
        )
        if rectangle is None:
            continue
        draw.rectangle(rectangle, fill=(0, 0, 0))
        drawn_count += 1

    return image, len(span_items), drawn_count


def mask_document_image(
    document: PreprocessedDocument,
    spans: Iterable[PiiSpan],
    output_path: str | Path,
    *,
    padding: int = 2,
) -> MaskedImageResult:
    """모든 span의 bbox를 검은색으로 칠해 새 이미지로 저장한다."""
    image, span_count, box_count = render_masked_image(
        document,
        spans,
        padding=padding,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(output, quality=95, subsampling=0)
    else:
        image.save(output)

    return MaskedImageResult(
        output_path=output,
        span_count=span_count,
        box_count=box_count,
    )
