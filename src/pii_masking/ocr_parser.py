"""OCR 서버 JSON 응답을 내부 OCR 문서 객체로 변환한다.

코드 검토 순서
1. ``parse_ocr_document()``가 OCR 응답에서 이미지 크기, ``full_text``,
   ``word_boxes``와 token 신뢰도를 꺼낸다.
2. 각 word box를 ``OcrToken``으로 만들고 잘못된 좌표를 검증한다.
3. token 목록을 ``OcrDocument``로 묶어 반환한다.
4. 반환된 문서는 ``preprocessing.preprocess_document()``로 전달된다.

이 파일은 OCR 서버가 반환한 내용을 읽기만 한다. 줄 재구성, 개인정보 탐지,
SearchView 생성과 이미지 마스킹은 수행하지 않는다.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from .models import Box, OcrDocument, OcrToken
from .paths import resolve_image_path


def _field_value(field_results: list[dict[str, Any]], name: str, field_id: str) -> Any:
    """
    OCR 필드를 displayName으로 찾고, 없으면 fieldId로 다시 찾는다.

    OCR 서버 설정에 따라 표시 이름이 달라질 가능성에 대비해 고정 fieldId도
    보조 키로 사용한다.
    """
    for field in field_results:
        if field.get("displayName") == name:
            return field.get("value")

    for field in field_results:
        if str(field.get("fieldId")) == field_id:
            return field.get("value")

    raise ValueError(f"OCR 응답에 {name!r} 필드가 없습니다.")


def _optional_field_value(
    field_results: list[dict[str, Any]],
    name: str,
    field_id: str,
) -> Any:
    """구버전 응답에도 사용할 수 있는 선택 필드를 읽는다."""
    try:
        return _field_value(field_results, name, field_id)
    except ValueError:
        return None


def parse_ocr_document(
    image_path: str,
    response: dict[str, Any],
    *,
    image_root: str | Path | None = None,
) -> OcrDocument:
    """이미지 한 장의 OCR 응답을 검증하고 OcrDocument로 변환한다."""
    if response.get("resultCode") != "0000":
        raise ValueError(
            f"OCR 처리에 실패한 응답입니다. resultCode={response.get('resultCode')!r}"
        )

    form_result = response.get("formResult")
    if not isinstance(form_result, dict):
        raise ValueError("OCR 응답에 formResult가 없습니다.")

    raw_document_type = form_result.get("type")
    # 현재 단계에서는 문서 종류에 의존하지 않는다. 값이 있으면 참고용으로만 보관한다.
    document_type = (
        raw_document_type
        if isinstance(raw_document_type, str) and raw_document_type
        else None
    )

    field_results = form_result.get("fieldResults")
    if not isinstance(field_results, list):
        raise ValueError("OCR 응답의 fieldResults가 목록이 아닙니다.")

    full_text = _field_value(field_results, "full_text", "801")
    word_boxes = _field_value(field_results, "word_boxes", "802")
    if not isinstance(full_text, str):
        raise ValueError("full_text가 문자열이 아닙니다.")
    if not isinstance(word_boxes, list):
        raise ValueError("word_boxes가 목록이 아닙니다.")

    raw_ocr_width = _optional_field_value(
        field_results,
        "ocr_image_width",
        "803",
    )
    raw_ocr_height = _optional_field_value(
        field_results,
        "ocr_image_height",
        "804",
    )
    try:
        ocr_image_width = int(raw_ocr_width) if raw_ocr_width is not None else None
        ocr_image_height = int(raw_ocr_height) if raw_ocr_height is not None else None
    except (TypeError, ValueError):
        ocr_image_width = None
        ocr_image_height = None

    # 기존 table OCR 결과는 803/804가 없지만 800번에 koiOcrResult가 있다.
    # 이 서버는 긴 변을 960px로 맞추므로 현재 저장된 결과도 올바르게 복원한다.
    is_table_ocr_response = any(
        isinstance(field, dict)
        and str(field.get("fieldId")) == "800"
        and "koiOcrResult" in field
        for field in field_results
    )

    tokens: list[OcrToken] = []
    seen_tokens: set[tuple[int, int, int, int, str]] = set()
    for token_id, item in enumerate(word_boxes):
        if not isinstance(item, dict):
            raise ValueError(f"word_boxes[{token_id}]가 객체가 아닙니다.")

        raw_box = item.get("boundingBox")
        if not isinstance(raw_box, dict):
            raise ValueError(f"word_boxes[{token_id}]에 boundingBox가 없습니다.")

        try:
            x1 = int(raw_box["x1"])
            y1 = int(raw_box["y1"])
            x2 = int(raw_box["x2"])
            y2 = int(raw_box["y2"])
            text = str(item["inferText"])
            confidence = float(item["inferConfidence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"word_boxes[{token_id}]의 형식이 잘못되었습니다.") from exc

        if x1 > x2 or y1 > y2:
            warnings.warn(
                f"word_boxes[{token_id}]의 좌표가 뒤집혀 있어 제외합니다: "
                f"text={text!r}, box=({x1}, {y1}, {x2}, {y2})",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        box = Box(x1=x1, y1=y1, x2=x2, y2=y2)

        duplicate_key = (box.x1, box.y1, box.x2, box.y2, text)
        if duplicate_key in seen_tokens:
            continue
        seen_tokens.add(duplicate_key)

        tokens.append(
            OcrToken(
                # 배열 순서를 고정 ID로 부여해 이후 모든 매핑 단계에서 유지한다.
                token_id=len(tokens),
                text=text,
                confidence=confidence,
                box=box,
            )
        )

    return OcrDocument(
        image_path=resolve_image_path(
            image_path,
            image_root=image_root,
        ),
        document_type=document_type,
        full_text=full_text,
        tokens=tuple(tokens),
        ocr_image_width=ocr_image_width,
        ocr_image_height=ocr_image_height,
        ocr_resize_target=(
            960
            if is_table_ocr_response
            and (ocr_image_width is None or ocr_image_height is None)
            else None
        ),
    )


def load_ocr_results(
    results_path: str | Path,
    *,
    skip_failed_responses: bool = False,
    image_root: str | Path | None = None,
) -> list[OcrDocument]:
    """여러 이미지의 OCR 응답이 담긴 JSON 파일 전체를 읽고 검증한다."""
    path = Path(results_path)
    with path.open(encoding="utf-8") as file:
        results = json.load(file)

    if not isinstance(results, dict):
        raise ValueError("OCR 결과 파일의 최상위 값은 객체여야 합니다.")

    documents: list[OcrDocument] = []
    errors: list[str] = []
    skipped: list[str] = []
    for image_path, response in results.items():
        if not isinstance(response, dict):
            errors.append(f"{image_path}: OCR 응답이 객체가 아닙니다.")
            continue
        if "error" in response:
            message = f"{image_path}: {response['error']}"
            (skipped if skip_failed_responses else errors).append(message)
            continue
        if response.get("resultCode") != "0000":
            message = (
                f"{image_path}: OCR 처리에 실패한 응답입니다. "
                f"resultCode={response.get('resultCode')!r}"
            )
            (skipped if skip_failed_responses else errors).append(message)
            continue
        try:
            documents.append(
                parse_ocr_document(
                    image_path,
                    response,
                    image_root=image_root,
                )
            )
        except ValueError as exc:
            # 한 건에서 바로 중단하지 않고 모든 오류를 모아 한 번에 보고한다.
            errors.append(f"{image_path}: {exc}")

    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"OCR 결과 일부를 읽지 못했습니다.\n{details}")

    if skipped:
        details = "\n".join(f"- {item}" for item in skipped)
        warnings.warn(
            "OCR 실패 문서는 마스킹에서 제외합니다.\n" + details,
            RuntimeWarning,
            stacklevel=2,
        )

    if not documents:
        raise ValueError("마스킹할 수 있는 정상 OCR 결과가 없습니다.")

    return documents
