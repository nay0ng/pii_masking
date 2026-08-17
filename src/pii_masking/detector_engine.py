"""탐지기의 문자열 결과를 공통 PiiSpan으로 변환한다.

ko-pii와 AEGIS는 모두 SearchView 안의 문자 시작·종료 위치를 반환한다. 이
파일은 그 위치를 OCR token ID, 원본 문자열 위치, 이미지 bbox로 변환한다.
여러 SearchView에서 같은 개인정보가 반복 탐지된 경우에는 겹치는 결과도
여기에서 정리한다.

코드 검토 순서
1. ``ProviderMatch``는 ko-pii와 AEGIS가 공통으로 반환할 중간 탐지 형식이다.
2. ``PiiSpan``은 문자 위치, token ID, bbox와 탐지 근거를 가진 최종 공통 형식이다.
3. ``make_span_from_search_view()``가 SearchView 문자 범위를 token과 bbox로 바꾼다.
4. ``detect_with_provider()``가 한 탐지기를 여러 SearchView에 실행한다.
5. ``merge_pii_spans()``가 서로 겹치거나 반복된 결과를 정리한다.

이 파일은 정규식이나 NER 추론을 직접 수행하지 않는다. 탐지기마다 다른 반환값을
동일한 ``PiiSpan``으로 만들고 원본 이미지 좌표로 연결하는 공통 엔진이다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re
from typing import Any

from .models import Box, OcrLine, OcrToken, PreprocessedDocument
from .ocr_preprocessing import build_text_index, normalize_for_search
from .search_views import (
    DEFAULT_SEARCH_VIEW_SETTINGS,
    STRUCTURED_BLOCK_CATEGORIES,
    SearchView,
    SearchViewSettings,
    build_address_field_search_view,
    build_cross_line_search_views,
    build_line_search_views,
    build_search_view,
)
from .table_field_detectors import build_table_field_search_views


SpanValidator = Callable[[str], bool]


@dataclass(frozen=True)
class ProviderMatch:
    """ko-pii와 AEGIS 결과를 동일하게 처리하기 위한 중간 형식."""

    category: str
    start: int
    end: int
    text: str
    confidence: float
    rule_id: str
    review_required: bool = False
    risk_level: int | None = None
    evidence: tuple[str, ...] = ()
    legal_basis: str | None = None


DetectionProvider = Callable[[SearchView], list[ProviderMatch]]


@dataclass(frozen=True)
class PatternRule:
    """개인정보 패턴 하나를 표현하는 공통 규칙 포맷."""

    rule_id: str
    pattern: re.Pattern[str]
    # 문맥 키워드까지 포함해 검색한 경우 실제 값만 결과로 만들기 위한 그룹이다.
    value_group: int | str = 0
    validator: SpanValidator | None = None
    review_required: bool = False


@dataclass(frozen=True)
class DetectorSpec:
    """카테고리 하나와 그 카테고리에 적용할 패턴 목록."""

    category: str
    rules: tuple[PatternRule, ...] = ()


@dataclass(frozen=True)
class PiiSpan:
    """개인정보 문자열, 탐지 근거, 원본 이미지 위치를 저장하는 최종 형식."""

    entity_type: str
    text: str
    normalized_text: str
    line_ids: tuple[int, ...]
    normalized_start: int
    normalized_end: int
    raw_start: int
    raw_end: int
    token_ids: tuple[int, ...]
    boxes: tuple[Box, ...]
    ocr_confidence: float
    detector_confidence: float
    rule_id: str
    review_required: bool = False
    risk_level: int | None = None
    evidence: tuple[str, ...] = ()
    legal_basis: str | None = None


@dataclass(frozen=True)
class _LineContext:
    line: OcrLine
    raw_offset: int


def _line_contexts(document: PreprocessedDocument) -> tuple[_LineContext, ...]:
    """각 줄이 문서 전체 raw_text에서 시작하는 위치를 계산한다."""
    contexts: list[_LineContext] = []
    raw_offset = 0
    for line in document.lines:
        line_index = build_text_index(line.tokens)
        raw_end = raw_offset + len(line_index.raw_text)
        if document.index.raw_text[raw_offset:raw_end] != line_index.raw_text:
            raise ValueError("문서 인덱스와 줄 인덱스의 raw_text 순서가 일치하지 않습니다.")
        contexts.append(_LineContext(line=line, raw_offset=raw_offset))
        raw_offset = raw_end + 1
    return tuple(contexts)


def _box_around(tokens: tuple[OcrToken, ...]) -> Box:
    return Box(
        x1=min(token.box.x1 for token in tokens),
        y1=min(token.box.y1 for token in tokens),
        x2=max(token.box.x2 for token in tokens),
        y2=max(token.box.y2 for token in tokens),
    )


def _cross_line_match_is_close(
    first_tokens: tuple[OcrToken, ...],
    second_tokens: tuple[OcrToken, ...],
) -> bool:
    """인접 줄에서 이어 붙인 값이 좌표상으로도 가까운지 공통 기준으로 확인한다."""
    if not first_tokens or not second_tokens:
        return False

    first_box = _box_around(first_tokens)
    second_box = _box_around(second_tokens)
    max_height = max(
        1,
        first_box.y2 - first_box.y1,
        second_box.y2 - second_box.y1,
    )
    vertical_gap = max(0, second_box.y1 - first_box.y2)
    left_delta = abs(first_box.x1 - second_box.x1)
    right_delta = abs(first_box.x2 - second_box.x2)
    return vertical_gap <= max_height * 2.5 and min(left_delta, right_delta) <= max_height * 4


def _make_span(
    document: PreprocessedDocument,
    category: str,
    lines: tuple[OcrLine, ...],
    raw_offset: int,
    match: re.Match[str],
    rule: PatternRule,
) -> PiiSpan:
    """정규식 결과를 공통 PiiSpan 및 원본 bbox로 변환한다."""
    normalized_start, normalized_end = match.span(rule.value_group)
    return _make_span_from_offsets(
        document=document,
        category=category,
        lines=lines,
        raw_offset=raw_offset,
        normalized_start=normalized_start,
        normalized_end=normalized_end,
        normalized_text=match.group(rule.value_group),
        confidence=None,
        rule_id=rule.rule_id,
        review_required=rule.review_required,
    )


def _make_span_from_offsets(
    document: PreprocessedDocument,
    category: str,
    lines: tuple[OcrLine, ...],
    raw_offset: int,
    normalized_start: int,
    normalized_end: int,
    normalized_text: str,
    confidence: float | None,
    rule_id: str,
    review_required: bool,
) -> PiiSpan:
    """어떤 탐지 소스든 동일한 offset→token→bbox 변환을 거치게 한다."""
    segment_index = build_text_index(
        tuple(token for line in lines for token in line.tokens),
        lines=lines,
    )
    local_raw_start, local_raw_end = segment_index.raw_span_for_normalized_span(
        normalized_start,
        normalized_end,
    )
    raw_start = raw_offset + local_raw_start
    raw_end = raw_offset + local_raw_end
    token_ids = tuple(
        segment_index.token_ids_for_normalized_span(normalized_start, normalized_end)
    )
    token_lookup = {
        token.token_id: token
        for line in lines
        for token in line.tokens
    }
    matched_tokens = tuple(token_lookup[token_id] for token_id in token_ids)
    matched_line_ids = tuple(
        line.line_id
        for line in lines
        if any(token.token_id in token_ids for token in line.tokens)
    )

    return PiiSpan(
        entity_type=category,
        text=document.index.raw_text[raw_start:raw_end],
        normalized_text=normalized_text,
        line_ids=matched_line_ids,
        normalized_start=normalized_start,
        normalized_end=normalized_end,
        raw_start=raw_start,
        raw_end=raw_end,
        token_ids=token_ids,
        boxes=tuple(token.box for token in matched_tokens),
        ocr_confidence=min(token.confidence for token in matched_tokens),
        detector_confidence=1.0 if confidence is None else confidence,
        rule_id=rule_id,
        review_required=review_required,
    )


def _matches_for_segment(
    document: PreprocessedDocument,
    spec: DetectorSpec,
    lines: tuple[OcrLine, ...],
    raw_offset: int,
    *,
    cross_line_boundary: int | None = None,
) -> list[PiiSpan]:
    """한 줄 또는 인접 두 줄에 모든 규칙을 동일한 방식으로 실행한다."""
    segment_index = build_text_index(
        tuple(token for line in lines for token in line.tokens),
        lines=lines,
    )
    spans: list[PiiSpan] = []

    for rule in spec.rules:
        for match in rule.pattern.finditer(segment_index.normalized_text):
            value = match.group(rule.value_group)
            if not value:
                continue
            if rule.validator is not None and not rule.validator(value):
                continue

            if cross_line_boundary is not None:
                # 문맥 키워드는 첫 줄, 실제 값은 둘째 줄에 있을 수도 있으므로
                # 결과 그룹이 아니라 정규식 전체가 줄 경계를 넘는지 확인한다.
                full_start, full_end = match.span(0)
                if not full_start < cross_line_boundary < full_end:
                    continue
                full_token_ids = set(
                    segment_index.token_ids_for_normalized_span(full_start, full_end)
                )
                first_tokens = tuple(
                    token for token in lines[0].tokens if token.token_id in full_token_ids
                )
                second_tokens = tuple(
                    token for token in lines[1].tokens if token.token_id in full_token_ids
                )
                if not _cross_line_match_is_close(first_tokens, second_tokens):
                    continue

            spans.append(
                _make_span(
                    document=document,
                    category=spec.category,
                    lines=lines,
                    raw_offset=raw_offset,
                    match=match,
                    rule=rule,
                )
            )
    return spans


CATEGORY_PRIORITY: dict[str, int] = {
    "resident_registration_number": 100,
    "corporate_registration_number": 95,
    "foreigner_registration_number": 90,
    "passport_number": 85,
    "driver_license_number": 85,
    "health_insurance_number": 85,
    "business_registration_number": 80,
    "account_card_number": 75,
    "phone_number": 60,
    "email_address": 60,
    "url": 55,
    "date": 50,
    "vehicle_plate_number": 50,
    "address": 30,
    "person_name": 20,
}

def calculate_span_priority(span: PiiSpan) -> tuple[float, int, int]:
    """Span의 신뢰도, 카테고리 우선순위, 길이(범위 크기)를 기준으로 가중치 튜플을 반환한다."""
    detector_conf = getattr(span, "detector_confidence", 0.0) or 0.0
    entity_type = getattr(span, "entity_type", "")
    priority_score = CATEGORY_PRIORITY.get(entity_type, 0)
    length = span.raw_end - span.raw_start

    return (detector_conf, priority_score, length)


def safe_convert_to_tuple(items: Any) -> tuple[Any, ...]:
    """list 또는 Iterable 객체를 safe하게 tuple로 변환한다."""
    if not items:
        return ()
    if isinstance(items, tuple):
        return items
    if isinstance(items, list):
        return tuple(items)
    try:
        return tuple(items)
    except TypeError:
        return ()


def _ensure_hashable_span(span: PiiSpan) -> PiiSpan:
    """PiiSpan 내부의 list 타입 필드들을 모두 tuple로 변환하여 완전한 Hashable 객체로 만든다."""
    token_ids_tuple = safe_convert_to_tuple(getattr(span, "token_ids", ()))
    line_ids_tuple = safe_convert_to_tuple(getattr(span, "line_ids", ()))
    boxes_tuple = safe_convert_to_tuple(getattr(span, "boxes", ()))
    evidence_tuple = safe_convert_to_tuple(getattr(span, "evidence", ()))

    return PiiSpan(
        entity_type=span.entity_type,
        text=span.text,
        normalized_text=getattr(span, "normalized_text", "") or "",
        raw_start=span.raw_start,
        raw_end=span.raw_end,
        normalized_start=getattr(span, "normalized_start", span.raw_start),
        normalized_end=getattr(span, "normalized_end", span.raw_end),
        token_ids=token_ids_tuple,
        line_ids=line_ids_tuple,
        boxes=boxes_tuple,
        ocr_confidence=getattr(span, "ocr_confidence", 0.0) or 0.0,
        detector_confidence=getattr(span, "detector_confidence", 0.0) or 0.0,
        rule_id=span.rule_id,
        review_required=span.review_required,
        risk_level=span.risk_level,
        evidence=evidence_tuple,
        legal_basis=span.legal_basis,
    )


def extract_normalized_bounds(group: list[PiiSpan], min_start: int, max_end: int) -> tuple[int, int]:
    """그룹 내 Span들로부터 normalized_start와 normalized_end의 최소/최대 범위를 안전하게 추출한다."""
    norm_starts = [
        getattr(s, "normalized_start", None)
        for s in group
        if getattr(s, "normalized_start", None) is not None
    ]
    norm_ends = [
        getattr(s, "normalized_end", None)
        for s in group
        if getattr(s, "normalized_end", None) is not None
    ]

    min_norm_start = min(norm_starts) if norm_starts else min_start
    max_norm_end = max(norm_ends) if norm_ends else max_end

    return min_norm_start, max_norm_end


def _filter_exact_and_contained_spans(spans: list[PiiSpan]) -> list[PiiSpan]:
    """동일하거나 다른 Span에 완전히 포함되는(Sub-span) 영역을 1차로 제거한다."""
    exact_map: dict[tuple[str, int, int, str], PiiSpan] = {}
    for span in spans:
        key = (span.entity_type, span.raw_start, span.raw_end, span.rule_id)
        exact_map.setdefault(key, span)

    sorted_spans = sorted(
        exact_map.values(),
        key=lambda s: (s.raw_start, -s.raw_end, -calculate_span_priority(s)[0]),
    )

    filtered: list[PiiSpan] = []
    for current in sorted_spans:
        is_contained = False
        for existing in filtered:
            if (
                existing.entity_type == current.entity_type
                and existing.raw_start <= current.raw_start
                and existing.raw_end >= current.raw_end
            ):
                is_contained = True
                break

        if not is_contained:
            filtered.append(current)

    return filtered


def _merge_fragmented_spans_by_token(spans: list[PiiSpan]) -> list[PiiSpan]:
    """
    동일한 OCR 토큰(token_ids)을 공유하거나 근접한 동일 카테고리 파편 Span을 하나로 병합한다.

    예: 동일 토큰 id=145 내에 분할 탐지된 'home'[747, 751]과 '.go.kr'[754, 760]을
        전체 범위인 [747, 760]으로 병합한다.
    """
    if not spans:
        return []

    grouped: dict[tuple[tuple[int, ...], str], list[PiiSpan]] = {}
    non_mergeable: list[PiiSpan] = []

    for span in spans:
        tokens = safe_convert_to_tuple(span.token_ids)
        if tokens:
            grouped.setdefault((tokens, span.entity_type), []).append(span)
        else:
            non_mergeable.append(span)

    merged_spans: list[PiiSpan] = list(non_mergeable)

    for (tokens, entity_type), group in grouped.items():
        if len(group) == 1:
            merged_spans.append(group[0])
            continue

        group.sort(key=lambda s: s.raw_start)

        min_start = min(s.raw_start for s in group)
        max_end = max(s.raw_end for s in group)
        min_norm_start, max_norm_end = extract_normalized_bounds(group, min_start, max_end)

        max_conf = max(getattr(s, "detector_confidence", 0.0) or 0.0 for s in group)
        max_ocr_conf = max(getattr(s, "ocr_confidence", 0.0) or 0.0 for s in group)

        base_span = group[0]
        merged_text = "".join(s.text for s in group)
        merged_norm_text = "".join(getattr(s, "normalized_text", "") or "" for s in group)

        merged_span = PiiSpan(
            entity_type=entity_type,
            text=merged_text,
            normalized_text=merged_norm_text,
            raw_start=min_start,
            raw_end=max_end,
            normalized_start=min_norm_start,
            normalized_end=max_norm_end,
            token_ids=safe_convert_to_tuple(tokens),
            line_ids=safe_convert_to_tuple(base_span.line_ids),
            boxes=safe_convert_to_tuple(base_span.boxes),
            ocr_confidence=max_ocr_conf,
            detector_confidence=max_conf,
            rule_id=base_span.rule_id,
            review_required=base_span.review_required,
            risk_level=base_span.risk_level,
            evidence=safe_convert_to_tuple(base_span.evidence),
            legal_basis=base_span.legal_basis,
        )
        merged_spans.append(merged_span)

    return merged_spans


def _resolve_location_conflicts(spans: list[PiiSpan]) -> list[PiiSpan]:
    """동일한 위치(시작/끝/토큰)에 복수의 카테고리가 매핑된 경우 최우선 순위 하나만 선별한다."""
    by_location: dict[tuple[int, int, tuple[int, ...]], list[PiiSpan]] = {}
    for span in spans:
        tokens_tuple = safe_convert_to_tuple(span.token_ids)
        key = (span.raw_start, span.raw_end, tokens_tuple)
        by_location.setdefault(key, []).append(span)

    resolved: list[PiiSpan] = []
    for items in by_location.values():
        best_span = items[0]
        best_priority = calculate_span_priority(best_span)

        for candidate in items[1:]:
            candidate_priority = calculate_span_priority(candidate)
            if candidate_priority > best_priority:
                best_span = candidate
                best_priority = candidate_priority

        resolved.append(best_span)

    return resolved


def _resolve_overlapping_spans(spans: list[PiiSpan]) -> list[PiiSpan]:
    """영역이 부분적으로 겹치는(Overlap) Span 간 충돌을 해결한다."""
    if not spans:
        return []

    sorted_spans = sorted(spans, key=lambda s: (s.raw_start, s.raw_end))
    results: list[PiiSpan] = []

    for current in sorted_spans:
        if not results:
            results.append(current)
            continue

        previous = results[-1]
        if current.raw_start < previous.raw_end:
            prev_priority = calculate_span_priority(previous)
            curr_priority = calculate_span_priority(current)

            if curr_priority > prev_priority:
                results[-1] = current
        else:
            results.append(current)

    return results


def _deduplicate_spans(spans: list[PiiSpan]) -> tuple[PiiSpan, ...]:
    """같은 값의 중복을 제거하고, 파편화 병합 및 위치 충돌을 해결하여 정제된 PiiSpan 튜플을 반환한다."""
    if not spans:
        return ()

    # 1. 입력받은 모든 PiiSpan의 list 구조 필드들을 Hashable(tuple) 상태로 정제
    hashable_spans = [_ensure_hashable_span(s) for s in spans]

    # 2. 완전 중복 및 하위 포함 span(sub-span) 제거
    step1_spans = _filter_exact_and_contained_spans(hashable_spans)

    # 3. 동일 토큰(token_ids) 공유 파편 Span 병합
    step2_spans = _merge_fragmented_spans_by_token(step1_spans)

    # 4. 완전 동일 위치 카테고리 충돌 해소
    step3_spans = _resolve_location_conflicts(step2_spans)

    # 5. 오버랩 충돌 해소
    step4_spans = _resolve_overlapping_spans(step3_spans)

    # 6. 문서 내 원문 위치(raw_start) 기준 정렬 및 불변 필드화 완료 객체 생성
    final_spans = [
        _ensure_hashable_span(s)
        for s in sorted(
            step4_spans,
            key=lambda s: (s.raw_start, s.raw_end, s.entity_type),
        )
    ]

    return tuple(final_spans)

def merge_pii_spans(*span_groups: tuple[PiiSpan, ...]) -> tuple[PiiSpan, ...]:
    """여러 탐지기의 PiiSpan을 공통 중복·번호유형 충돌 규칙으로 병합한다."""
    return _deduplicate_spans([
        span
        for group in span_groups
        for span in group
    ])


def detect_with_spec(
    document: PreprocessedDocument,
    spec: DetectorSpec,
) -> tuple[PiiSpan, ...]:
    """
    모든 카테고리에 동일한 탐지 파이프라인을 적용한다.

    1. 각 줄을 검색한다.
    2. 인접 두 줄을 이어 검색하되 실제 정규식이 줄 경계를 넘고 좌표가 가까울 때만 허용한다.
    3. normalized span을 raw span, token ID, 원본 bbox로 변환한다.
    """
    if not spec.rules:
        return ()

    contexts = _line_contexts(document)
    spans: list[PiiSpan] = []

    for context in contexts:
        spans.extend(
            _matches_for_segment(
                document,
                spec,
                (context.line,),
                context.raw_offset,
            )
        )

    for first, second in zip(contexts, contexts[1:]):
        first_index = build_text_index(first.line.tokens)
        spans.extend(
            _matches_for_segment(
                document,
                spec,
                (first.line, second.line),
                first.raw_offset,
                cross_line_boundary=len(first_index.normalized_text),
            )
        )

    return _deduplicate_spans(spans)


def make_span_from_search_view(
    document: PreprocessedDocument,
    view: SearchView,
    result: ProviderMatch,
) -> PiiSpan | None:
    """SearchView 결과를 공통 raw span, token ID, bbox 결과로 변환한다."""
    if result.start < 0 or result.end <= result.start or result.end > len(view.text):
        return None
    if (
        view.allowed_categories is not None
        and result.category not in view.allowed_categories
    ):
        return None

    raw_indexes = view.raw_indexes_for_span(result.start, result.end)
    token_ids = view.token_ids_for_span(result.start, result.end)
    if not raw_indexes or not token_ids:
        return None

    raw_start = min(raw_indexes)
    raw_end = max(raw_indexes) + 1
    token_lookup = {
        token.token_id: token
        for line in view.lines
        for token in line.tokens
    }
    matched_tokens = tuple(
        token_lookup[token_id]
        for token_id in token_ids
        if token_id in token_lookup
    )
    if not matched_tokens:
        return None
    matched_line_ids = tuple(
        line.line_id
        for line in view.lines
        if any(token.token_id in token_ids for token in line.tokens)
    )
    span_text = document.index.raw_text[raw_start:raw_end]
    if view.mode in {"address_field", "table_field", "table_field_compact"}:
        # 표에서는 주소 첫 줄과 연속 줄 사이에 등록일자/접수번호 열이 직렬화될 수 있다.
        # 실제 마스킹 token만 사용해 표시 문자열을 만들고 다른 열의 값은 제외한다.
        span_text = " ".join(token.text.strip() for token in matched_tokens).strip()

    return PiiSpan(
        entity_type=result.category,
        text=span_text,
        normalized_text=normalize_for_search(view.text[result.start:result.end]),
        line_ids=matched_line_ids,
        normalized_start=result.start,
        normalized_end=result.end,
        raw_start=raw_start,
        raw_end=raw_end,
        token_ids=token_ids,
        boxes=tuple(token.box for token in matched_tokens),
        ocr_confidence=min(token.confidence for token in matched_tokens),
        detector_confidence=result.confidence,
        rule_id=result.rule_id,
        review_required=result.review_required,
        risk_level=result.risk_level,
        evidence=result.evidence,
        legal_basis=result.legal_basis,
    )


def _provider_matches_for_view(
    document: PreprocessedDocument,
    provider: DetectionProvider,
    view: SearchView,
    *,
    cross_line: bool = False,
) -> list[PiiSpan]:
    """외부 탐지 결과를 SearchView가 보관한 원본 위치로 되돌린다."""
    spans: list[PiiSpan] = []
    line_boundary = view.text.find("\n") if cross_line else -1
    for result in provider(view):
        if cross_line:
            # 첫 줄에만 있는 결과는 단일 줄 보기에서 이미 처리했다.
            if line_boundary < 0 or result.end <= line_boundary:
                continue
            token_ids = set(view.token_ids_for_span(result.start, result.end))
            first_tokens = tuple(
                token for token in view.lines[0].tokens if token.token_id in token_ids
            )
            second_tokens = tuple(
                token for token in view.lines[1].tokens if token.token_id in token_ids
            )
            # 값은 둘째 줄에만 있고 첫 줄의 필드명이 anchor일 수도 있다.
            if not first_tokens:
                first_tokens = view.lines[0].tokens
            if not _cross_line_match_is_close(first_tokens, second_tokens):
                continue

        span = make_span_from_search_view(document, view, result)
        if span is not None:
            spans.append(span)
    return spans


def detect_with_provider(
    document: PreprocessedDocument,
    provider: DetectionProvider,
    *,
    search_settings: SearchViewSettings = DEFAULT_SEARCH_VIEW_SETTINGS,
) -> tuple[PiiSpan, ...]:
    """여러 SearchView를 통해 ko-pii를 실행하고 결과를 하나의 span 형식으로 합친다."""
    contexts = _line_contexts(document)
    spans: list[PiiSpan] = []

    # 일반 줄 연결에는 표의 x축 열 관계가 없다. 헤더와 같은 열에 놓인 값만
    # 선택한 제한 SearchView를 ko-pii와 AEGIS에 공통으로 전달한다.
    for view in build_table_field_search_views(document, search_settings):
        spans.extend(_provider_matches_for_view(document, provider, view))

    for context in contexts:
        for view in build_line_search_views(
            context.line,
            context.raw_offset,
            settings=search_settings,
        ):
            spans.extend(_provider_matches_for_view(document, provider, view))
        address_view = build_address_field_search_view(
            (context.line,),
            context.raw_offset,
            settings=search_settings,
        )
        if address_view is not None:
            spans.extend(_provider_matches_for_view(document, provider, address_view))
        structured_view = build_search_view(
            (context.line,),
            context.raw_offset,
            mode="structured_block",
            allowed_categories=STRUCTURED_BLOCK_CATEGORIES,
        )
        spans.extend(_provider_matches_for_view(document, provider, structured_view))

    if search_settings.max_adjacent_lines >= 2:
        for first, second in zip(contexts, contexts[1:]):
            for view in build_cross_line_search_views(
                first.line,
                second.line,
                first.raw_offset,
                settings=search_settings,
            ):
                spans.extend(
                    _provider_matches_for_view(
                        document,
                        provider,
                        view,
                        cross_line=True,
                    )
                )
            address_view = build_address_field_search_view(
                (first.line, second.line),
                first.raw_offset,
                settings=search_settings,
            )
            if address_view is not None:
                spans.extend(
                    _provider_matches_for_view(document, provider, address_view)
                )
            structured_view = build_search_view(
                (first.line, second.line),
                first.raw_offset,
                mode="structured_block",
                allowed_categories=STRUCTURED_BLOCK_CATEGORIES,
            )
            spans.extend(
                _provider_matches_for_view(document, provider, structured_view)
            )

    if search_settings.max_adjacent_lines >= 3:
        for first, second, third in zip(contexts, contexts[1:], contexts[2:]):
            lines = (first.line, second.line, third.line)
            view = build_search_view(
                lines,
                first.raw_offset,
                mode="structured_block",
                allowed_categories=STRUCTURED_BLOCK_CATEGORIES,
            )
            spans.extend(_provider_matches_for_view(document, provider, view))
            address_view = build_address_field_search_view(
                (first.line, second.line, third.line),
                first.raw_offset,
                settings=search_settings,
            )
            if address_view is not None:
                spans.extend(
                    _provider_matches_for_view(document, provider, address_view)
                )

    return _deduplicate_spans(spans)
