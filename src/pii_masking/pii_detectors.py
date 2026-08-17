"""개인정보 카테고리를 유형별 탐지 함수와 연결한다.

코드 검토 순서
1. ``DETECTOR_SPECS``에서 내부 카테고리와 ko-pii 라벨의 대응을 확인한다.
2. ``_detect()``가 한 카테고리를 ``ko_pii_adapter``에 전달한다.
3. ``CATEGORY_DETECTORS``가 외부에 제공할 카테고리별 함수를 등록한다.
4. ``detect_all_pii_spans()``가 한 문서에서 선택된 카테고리를 모두 탐지한다.

대부분의 카테고리는 ko-pii 결과를 공통 ``PiiSpan``으로 받는다. 표의 세로 필드는
공통 SearchView 단계에서 모든 유형에 연결하고, 주소에는 상세주소 bbox 확장 후처리를 한다.
실제 정규식과 체크섬 판단은 ko-pii와 ``ko_pii_detector.py``에서 수행한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .categories import PII_CATEGORIES
from .detector_engine import PiiSpan, merge_pii_spans
from .ko_pii_detector import (
    KO_PII_CATEGORY_SPECS,
    detect_ko_pii_categories,
    detect_ko_pii_spans,
)
from .models import PreprocessedDocument
from .search_views import (
    DEFAULT_SEARCH_VIEW_SETTINGS,
    SearchViewSettings,
)
from .address_expander import extend_address_spans


# 외부에 보여 주는 설정 이름은 기존 코드와 호환되도록 유지한다.
DETECTOR_SPECS = KO_PII_CATEGORY_SPECS


def _detect(document: PreprocessedDocument, category: str) -> tuple[PiiSpan, ...]:
    """지정한 카테고리를 ko-pii로 탐지하고 유형별 보완을 적용한다."""
    spans = detect_ko_pii_spans(document, DETECTOR_SPECS[category])
    if category == "address":
        spans = extend_address_spans(document, spans)
    return spans


def detect_person_name(document: PreprocessedDocument) -> tuple[PiiSpan, ...]:
    return _detect(document, "person_name")


def detect_address(document: PreprocessedDocument) -> tuple[PiiSpan, ...]:
    return _detect(document, "address")


def detect_email_address(document: PreprocessedDocument) -> tuple[PiiSpan, ...]:
    return _detect(document, "email_address")


def detect_phone_number(document: PreprocessedDocument) -> tuple[PiiSpan, ...]:
    return _detect(document, "phone_number")


def detect_url(document: PreprocessedDocument) -> tuple[PiiSpan, ...]:
    return _detect(document, "url")


def detect_date(document: PreprocessedDocument) -> tuple[PiiSpan, ...]:
    return _detect(document, "date")


def detect_account_card_number(document: PreprocessedDocument) -> tuple[PiiSpan, ...]:
    return _detect(document, "account_card_number")


def detect_credential_secret(document: PreprocessedDocument) -> tuple[PiiSpan, ...]:
    return _detect(document, "credential_secret")


def detect_resident_registration_number(document: PreprocessedDocument) -> tuple[PiiSpan, ...]:
    return _detect(document, "resident_registration_number")


def extract_rrn_spans(document: PreprocessedDocument) -> tuple[PiiSpan, ...]:
    """기존 호출부와 호환되는 주민등록번호 탐지 함수."""
    return detect_resident_registration_number(document)


def detect_foreigner_registration_number(document: PreprocessedDocument) -> tuple[PiiSpan, ...]:
    return _detect(document, "foreigner_registration_number")


def detect_passport_number(document: PreprocessedDocument) -> tuple[PiiSpan, ...]:
    return _detect(document, "passport_number")


def detect_driver_license_number(document: PreprocessedDocument) -> tuple[PiiSpan, ...]:
    return _detect(document, "driver_license_number")


def detect_business_registration_number(document: PreprocessedDocument) -> tuple[PiiSpan, ...]:
    return _detect(document, "business_registration_number")


def detect_corporate_registration_number(document: PreprocessedDocument) -> tuple[PiiSpan, ...]:
    return _detect(document, "corporate_registration_number")


def detect_health_insurance_number(document: PreprocessedDocument) -> tuple[PiiSpan, ...]:
    return _detect(document, "health_insurance_number")


def detect_vehicle_plate_number(document: PreprocessedDocument) -> tuple[PiiSpan, ...]:
    return _detect(document, "vehicle_plate_number")


def detect_ip_address(document: PreprocessedDocument) -> tuple[PiiSpan, ...]:
    return _detect(document, "ip_address")


_NAMED_DETECTOR_BY_CATEGORY = {
    "person_name": detect_person_name,
    "address": detect_address,
    "email_address": detect_email_address,
    "phone_number": detect_phone_number,
    "url": detect_url,
    "date": detect_date,
    "account_card_number": detect_account_card_number,
    "credential_secret": detect_credential_secret,
    "resident_registration_number": detect_resident_registration_number,
    "foreigner_registration_number": detect_foreigner_registration_number,
    "passport_number": detect_passport_number,
    "driver_license_number": detect_driver_license_number,
    "business_registration_number": detect_business_registration_number,
    "corporate_registration_number": detect_corporate_registration_number,
    "health_insurance_number": detect_health_insurance_number,
    "vehicle_plate_number": detect_vehicle_plate_number,
    "ip_address": detect_ip_address,
}


@dataclass(frozen=True)
class CategoryDetector:
    """별도 함수가 없는 ko-pii 카테고리를 공통 탐지 경로에 연결한다."""

    category: str

    def __call__(self, document: PreprocessedDocument) -> tuple[PiiSpan, ...]:
        return _detect(document, self.category)


DETECTOR_BY_CATEGORY = {
    category.label: _NAMED_DETECTOR_BY_CATEGORY.get(category.label)
    or CategoryDetector(category.label)
    for category in PII_CATEGORIES
}

PII_DETECTORS = tuple(DETECTOR_BY_CATEGORY.values())


def implemented_category_labels() -> tuple[str, ...]:
    return tuple(
        category
        for category, spec in DETECTOR_SPECS.items()
        if spec.labels
    )


def detect_all_pii_spans(
    document: PreprocessedDocument,
    *,
    search_settings: SearchViewSettings = DEFAULT_SEARCH_VIEW_SETTINGS,
) -> tuple[PiiSpan, ...]:
    """등록된 모든 카테고리를 한 번의 SearchView 순회로 탐지한다."""
    spans = detect_ko_pii_categories(
        document,
        tuple(DETECTOR_SPECS.values()),
        search_settings=search_settings,
    )
    return extend_address_spans(document, merge_pii_spans(spans))
