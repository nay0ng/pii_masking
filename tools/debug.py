"""OCR token 좌표를 이미지로 확인하는 개발용 도구.

특정 token의 bbox를 원본 이미지에 표시한다. 탐지나 마스킹 결과에는 영향을
주지 않으며, 좌표 복원 결과를 확인할 때만 사용한다.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pii_masking.models import OcrDocument
from pii_masking.ocr_preprocessing import preprocess_document


def render_token_debug_image(
    document: OcrDocument,
    token_id: int,
    output_path: str | Path,
) -> Path:
    """
    좌표 복원이 맞는지 눈으로 확인할 수 있도록 특정 토큰에 빨간 박스를 그린다.

    실제 마스킹 함수가 아니라 전처리 결과 검증을 위한 도구이다.
    """
    processed = preprocess_document(document)
    try:
        token = processed.index.tokens[token_id]
    except IndexError as exc:
        raise ValueError(
            f"token_id {token_id}가 범위를 벗어났습니다. "
            f"가능한 범위: 0~{len(document.tokens) - 1}"
        ) from exc

    if token.token_id != token_id:
        raise ValueError("토큰 목록과 token_id가 일치하지 않습니다.")
    if not document.image_path.exists():
        raise FileNotFoundError(f"원본 이미지를 찾을 수 없습니다: {document.image_path}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(document.image_path) as source:
        image = source.convert("RGB")

    if token.box.x2 > image.width or token.box.y2 > image.height:
        raise ValueError(
            "OCR 좌표가 원본 이미지 크기를 벗어났습니다. "
            f"image={image.width}x{image.height}, box={token.box}"
        )

    draw = ImageDraw.Draw(image)
    box_height = token.box.y2 - token.box.y1
    line_width = max(3, round(box_height * 0.12))
    draw.rectangle(
        [token.box.x1, token.box.y1, token.box.x2, token.box.y2],
        outline=(255, 0, 0),
        width=line_width,
    )

    label = f"token {token.token_id}: {token.text}"
    font = ImageFont.load_default()
    label_box = draw.textbbox((0, 0), label, font=font)
    label_width = label_box[2] - label_box[0]
    label_height = label_box[3] - label_box[1]
    label_x = token.box.x1
    label_y = max(0, token.box.y1 - label_height - 8)
    draw.rectangle(
        [
            label_x,
            label_y,
            min(image.width, label_x + label_width + 8),
            label_y + label_height + 6,
        ],
        fill=(255, 0, 0),
    )
    draw.text((label_x + 4, label_y + 3), label, fill=(255, 255, 255), font=font)

    image.save(output)
    return output
