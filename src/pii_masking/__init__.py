"""OCR 개인정보 탐지·마스킹 모듈에서 외부에 공개하는 기능."""

from .categories import (
    CATEGORY_BY_LABEL,
    PII_CATEGORIES,
    PiiCategory,
)
from .image_masker import (
    MaskedImageResult,
    build_masked_output_path,
    mask_document_image,
)
from .pii_detectors import (
    DETECTOR_BY_CATEGORY,
    PiiSpan,
    detect_all_pii_spans,
    extract_rrn_spans,
    implemented_category_labels,
)
from .detector_engine import merge_pii_spans
from .detection_policy import (
    DetectionPolicy,
    DetectorExecutionMode,
    DetectorMergeMode,
    GateAction,
    PolicyRunResult,
    ReviewHandling,
    SpanPolicyDecision,
    combine_policy_spans,
    decide_ner_execution,
    evaluate_span_policy,
)
from .models import (
    Box,
    DocumentIndex,
    OcrDocument,
    OcrLine,
    OcrToken,
    PreprocessedDocument,
)
from .policy import (
    DEFAULT_POLICY_PATH,
    MaskingSensitivity,
    MaskingPolicyConfig,
    SensitivityPreset,
    get_sensitivity_preset,
    load_masking_policy,
    save_masking_policy,
)
from .ocr_parser import load_ocr_results, parse_ocr_document
from .ocr_preprocessing import (
    NORMALIZED_DASHES,
    build_lines,
    build_text_index,
    normalize_for_search,
    preprocess_document,
)
from .detection_pipeline import detect_document_pii
from .search_views import PatternStrictness, SearchViewSettings

__all__ = [
    "Box",
    "CATEGORY_BY_LABEL",
    "DETECTOR_BY_CATEGORY",
    "DocumentIndex",
    "DetectionPolicy",
    "DetectorExecutionMode",
    "DetectorMergeMode",
    "GateAction",
    "MaskedImageResult",
    "MaskingSensitivity",
    "MaskingPolicyConfig",
    "NORMALIZED_DASHES",
    "OcrDocument",
    "OcrLine",
    "OcrToken",
    "PII_CATEGORIES",
    "PiiCategory",
    "PiiSpan",
    "PolicyRunResult",
    "PatternStrictness",
    "ReviewHandling",
    "SpanPolicyDecision",
    "PreprocessedDocument",
    "SearchViewSettings",
    "SensitivityPreset",
    "build_lines",
    "build_masked_output_path",
    "build_text_index",
    "combine_policy_spans",
    "detect_all_pii_spans",
    "detect_document_pii",
    "decide_ner_execution",
    "evaluate_span_policy",
    "extract_rrn_spans",
    "implemented_category_labels",
    "get_sensitivity_preset",
    "load_ocr_results",
    "load_masking_policy",
    "mask_document_image",
    "merge_pii_spans",
    "normalize_for_search",
    "parse_ocr_document",
    "preprocess_document",
    "save_masking_policy",
    "DEFAULT_POLICY_PATH",
]
