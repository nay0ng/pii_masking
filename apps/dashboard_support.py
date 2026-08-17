"""Streamlit 화면에 표시할 표와 좌표 이미지를 만든다.

탐지 결과를 변경하지 않고, 이미 계산된 OCR token과 개인정보 span을 화면에서
확인할 수 있는 형태로 변환한다. 배포용 탐지 모듈에는 포함하지 않는다.

코드 검토 순서
1. ``build_document_search_views()``가 한 문서에 생성되는 모든 SearchView를 화면
   표시용 목록으로 모은다.
2. ``trace_ko_pii()``가 ko-pii 원시 후보와 최종 PiiSpan을 비교해 유지·중복·제외
   상태를 표시한다.
3. ``render_token_overlay()``는 OCR token과 재구성 line을 이미지 위에 표시한다.
4. ``render_span_overlay()``는 최종 span의 bbox와 번호를 이미지 위에 표시한다.
5. ``token_rows()``와 ``span_rows()``는 내부 객체를 Streamlit 표 행으로 바꾼다.
6. ``discover_result_files()``는 화면에서 선택할 OCR 결과 JSON을 찾는다.

이 함수들은 진단을 위해 같은 데이터를 다른 모양으로 보여 줄 뿐이다. 여기에서
만든 trace나 그림이 실제 탐지 결과와 마스킹 정책을 변경하지는 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pii_masking.detector_engine import PiiSpan
from pii_masking.ko_pii_detector import iter_adapter_detections
from pii_masking.models import PreprocessedDocument
from pii_masking.ocr_preprocessing import build_text_index
from pii_masking.search_views import (
    DEFAULT_SEARCH_VIEW_SETTINGS,
    STRUCTURED_BLOCK_CATEGORIES,
    SearchView,
    SearchViewSettings,
    build_address_field_search_view,
    build_cross_line_search_views,
    build_line_search_views,
    build_search_view,
)
from pii_masking.table_field_detectors import build_table_field_search_views


CATEGORY_NAMES = {
    "person_name": "이름", "address": "주소", "email_address": "이메일",
    "phone_number": "전화번호", "url": "URL", "date": "날짜",
    "account_card_number": "계좌·카드번호", "credential_secret": "인증 비밀정보",
    "resident_registration_number": "주민등록번호", "foreigner_registration_number": "외국인등록번호",
    "passport_number": "여권번호", "driver_license_number": "운전면허번호",
    "business_registration_number": "사업자등록번호", "corporate_registration_number": "법인등록번호",
    "health_insurance_number": "건강보험증번호", "vehicle_plate_number": "차량번호",
    "ip_address": "IP 주소",
    "land_lot_number": "토지번호",
    "prescription_number": "처방번호",
    "drug_code": "의약품 코드",
    "employee_number": "사번",
    "petition_number": "민원번호",
    "court_case_number": "사건번호",
    "postal_code": "우편번호",
    "document_number": "문서번호",
    "nationality": "국적",
    "education_history": "학력",
    "academic_major": "전공",
    "job_position": "직책",
    "age": "나이",
    "height": "신장",
    "weight": "체중",
    "username": "사용자 ID",
    "id_card_number": "신분증 번호",
    "time": "시간",
    "company_name": "회사명",
}

_COLORS = (
    (220, 38, 38), (37, 99, 235), (5, 150, 105), (217, 119, 6),
    (147, 51, 234), (219, 39, 119), (8, 145, 178),
)


@dataclass(frozen=True)
class DetectorTrace:
    """ko-pii의 원시 후보 하나가 최종 span까지 도달했는지 기록한다."""

    line_ids: tuple[int, ...]
    view_mode: str
    search_text: str
    category: str
    detected_text: str
    confidence: float
    rule_id: str
    start: int
    end: int
    token_ids: tuple[int, ...]
    outcome: str


def _line_offsets(document: PreprocessedDocument) -> dict[int, int]:
    offsets: dict[int, int] = {}
    raw_offset = 0
    for line in document.lines:
        offsets[line.line_id] = raw_offset
        raw_offset += len(build_text_index(line.tokens).raw_text) + 1
    return offsets


def build_document_search_views(
    document: PreprocessedDocument,
    *,
    search_settings: SearchViewSettings = DEFAULT_SEARCH_VIEW_SETTINGS,
) -> tuple[SearchView, ...]:
    """실제 탐지 엔진과 동일한 단일 줄/인접 두 줄 검색 입력을 만든다."""
    offsets = _line_offsets(document)
    views: list[SearchView] = list(
        build_table_field_search_views(document, search_settings)
    )
    for line in document.lines:
        views.extend(
            build_line_search_views(
                line,
                offsets[line.line_id],
                settings=search_settings,
            )
        )
        address_view = build_address_field_search_view(
            (line,),
            offsets[line.line_id],
            settings=search_settings,
        )
        if address_view is not None:
            views.append(address_view)
        views.append(
            build_search_view(
                (line,),
                offsets[line.line_id],
                mode="structured_block",
                allowed_categories=STRUCTURED_BLOCK_CATEGORIES,
            )
        )
    if search_settings.max_adjacent_lines >= 2:
        for first, second in zip(document.lines, document.lines[1:]):
            views.extend(
                build_cross_line_search_views(
                    first,
                    second,
                    offsets[first.line_id],
                    settings=search_settings,
                )
            )
            address_view = build_address_field_search_view(
                (first, second),
                offsets[first.line_id],
                settings=search_settings,
            )
            if address_view is not None:
                views.append(address_view)
            views.append(
                build_search_view(
                    (first, second),
                    offsets[first.line_id],
                    mode="structured_block",
                    allowed_categories=STRUCTURED_BLOCK_CATEGORIES,
                )
            )
    if search_settings.max_adjacent_lines >= 3:
        for first, second, third in zip(
            document.lines, document.lines[1:], document.lines[2:]
        ):
            lines = (first, second, third)
            views.append(
                build_search_view(
                    lines,
                    offsets[first.line_id],
                    mode="structured_block",
                    allowed_categories=STRUCTURED_BLOCK_CATEGORIES,
                )
            )
            address_view = build_address_field_search_view(
                (first, second, third),
                offsets[first.line_id],
                settings=search_settings,
            )
            if address_view is not None:
                views.append(address_view)
    return tuple(views)


def trace_ko_pii(
    document: PreprocessedDocument,
    final_spans: tuple[PiiSpan, ...],
    *,
    search_settings: SearchViewSettings = DEFAULT_SEARCH_VIEW_SETTINGS,
) -> tuple[DetectorTrace, ...]:
    """SearchView별 ko-pii 원시 결과와 필터/매핑/최종 유지 여부를 추적한다."""
    traces: list[DetectorTrace] = []
    final_keys = {(span.entity_type, span.token_ids) for span in final_spans}

    for view in build_document_search_views(
        document,
        search_settings=search_settings,
    ):
        for detection in iter_adapter_detections(
            view,
            search_settings=search_settings,
        ):
            category = detection.category
            result = detection.result
            start, end = int(result.start), int(result.end)
            token_ids: tuple[int, ...] = ()
            if start < 0 or end <= start or end > len(view.text):
                outcome = "위치 오류"
            else:
                token_ids = view.token_ids_for_span(start, end)
                if not token_ids:
                    outcome = "토큰 매핑 실패"
                elif (category, token_ids) in final_keys:
                    outcome = "최종 span 유지"
                elif any(
                    span.entity_type == category and set(span.token_ids) == set(token_ids)
                    for span in final_spans
                ):
                    outcome = "최종 span 유지"
                else:
                    outcome = "중복 제거/후처리"

            traces.append(DetectorTrace(
                line_ids=tuple(line.line_id for line in view.lines),
                view_mode=view.mode,
                search_text=detection.input_text,
                category=category,
                detected_text=str(result.text),
                confidence=float(result.confidence),
                rule_id=detection.rule_id,
                start=start,
                end=end,
                token_ids=token_ids,
                outcome=outcome,
            ))
    return tuple(traces)


def render_span_overlay(
    document: PreprocessedDocument,
    spans: tuple[PiiSpan, ...],
    *,
    selected_index: int | None = None,
    padding: int = 2,
) -> Image.Image:
    """탐지 bbox를 반투명 색상과 span 번호로 원본 위에 표시한다."""
    with Image.open(document.document.image_path) as source:
        base = source.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    for index, span in enumerate(spans):
        color = _COLORS[index % len(_COLORS)]
        is_selected = selected_index == index
        alpha, width = (105, 5) if is_selected else (55, 3)
        for box_index, box in enumerate(span.boxes):
            rectangle = (
                max(0, box.x1 - padding), max(0, box.y1 - padding),
                min(base.width - 1, box.x2 + padding), min(base.height - 1, box.y2 + padding),
            )
            draw.rectangle(rectangle, fill=(*color, alpha), outline=(*color, 255), width=width)
            if box_index == 0:
                label = str(index + 1)
                label_box = draw.textbbox((rectangle[0], rectangle[1]), label, font=font)
                draw.rectangle(
                    (label_box[0] - 3, max(0, label_box[1] - 3), label_box[2] + 3, label_box[3] + 3),
                    fill=(*color, 255),
                )
                draw.text((rectangle[0], max(0, rectangle[1])), label, fill="white", font=font)
    return Image.alpha_composite(base, overlay).convert("RGB")


def render_token_overlay(
    document: PreprocessedDocument,
    *,
    selected_token_ids: tuple[int, ...] = (),
) -> Image.Image:
    """OCR token bbox와 줄 번호를 표시하여 OCR/전처리 좌표를 검수한다."""
    with Image.open(document.document.image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    selected = set(selected_token_ids)
    font = ImageFont.load_default()
    for line in document.lines:
        for token in line.tokens:
            color = (220, 38, 38) if token.token_id in selected else (37, 99, 235)
            draw.rectangle(
                (token.box.x1, token.box.y1, token.box.x2, token.box.y2),
                outline=color, width=4 if token.token_id in selected else 2,
            )
        if line.tokens:
            first = line.tokens[0]
            draw.text((first.box.x1, max(0, first.box.y1 - 12)), f"L{line.line_id}", fill=(5, 100, 220), font=font)
    return image


def span_rows(spans: tuple[PiiSpan, ...]) -> list[dict[str, object]]:
    return [{
        "#": index + 1,
        "유형": CATEGORY_NAMES.get(span.entity_type, span.entity_type),
        "라벨": span.entity_type,
        "탐지 문자열": span.text.replace("\n", " ↵ "),
        "줄": ", ".join(map(str, span.line_ids)),
        "token IDs": ", ".join(map(str, span.token_ids)),
        "OCR 신뢰도": round(span.ocr_confidence, 3),
        "탐지 신뢰도": round(span.detector_confidence, 3),
        "위험도": span.risk_level,
        "탐지 근거": ", ".join(span.evidence),
        "법적 근거": span.legal_basis or "",
        "검수 필요": span.review_required,
        "규칙": span.rule_id,
        "box 수": len(span.boxes),
    } for index, span in enumerate(spans)]


def token_rows(document: PreprocessedDocument) -> list[dict[str, object]]:
    line_by_token = {
        token.token_id: line.line_id for line in document.lines for token in line.tokens
    }
    return [{
        "token ID": token.token_id,
        "line": line_by_token[token.token_id],
        "text": token.text,
        "OCR 신뢰도": round(token.confidence, 3),
        "x1": token.box.x1, "y1": token.box.y1, "x2": token.box.x2, "y2": token.box.y2,
    } for token in document.index.tokens]


def discover_result_files(root: str | Path) -> tuple[Path, ...]:
    """작업 폴더 최상위의 OCR 결과 JSON만 이름순으로 찾는다."""
    path = Path(root)
    return tuple(sorted(set(path.glob("*_results.json")) | set(path.glob("*_result.json"))))
