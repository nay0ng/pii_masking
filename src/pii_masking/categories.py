"""프로젝트에서 목표로 하는 개인정보 카테고리 정의."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PiiCategory:
    """외부 결과에 사용할 고정 개인정보 라벨과 표시 정보."""

    label: str
    korean_name: str
    is_unique_identifier: bool = False


PII_CATEGORIES: tuple[PiiCategory, ...] = (
    PiiCategory("person_name", "이름"),
    PiiCategory("address", "주소"),
    PiiCategory("email_address", "이메일"),
    PiiCategory("phone_number", "전화번호"),
    PiiCategory("url", "URL"),
    PiiCategory("date", "날짜 (생년월일 등)"),
    PiiCategory("account_card_number", "계좌·카드번호"),
    PiiCategory("credential_secret", "API 키·토큰·비밀번호"),
    PiiCategory("resident_registration_number", "주민등록번호", True),
    PiiCategory("foreigner_registration_number", "외국인등록번호", True),
    PiiCategory("passport_number", "여권번호", True),
    PiiCategory("driver_license_number", "운전면허번호", True),
    PiiCategory("business_registration_number", "사업자등록번호"),
    PiiCategory("corporate_registration_number", "법인등록번호"),
    PiiCategory("health_insurance_number", "건강보험증번호", True),
    PiiCategory("vehicle_plate_number", "차량번호"),
    PiiCategory("ip_address", "IP 주소"),
    PiiCategory("land_lot_number", "토지번호"),
    PiiCategory("prescription_number", "처방번호"),
    PiiCategory("drug_code", "의약품 코드"),
    PiiCategory("employee_number", "사번"),
    PiiCategory("petition_number", "민원번호"),
    PiiCategory("court_case_number", "사건번호"),
    PiiCategory("postal_code", "우편번호"),
    PiiCategory("document_number", "문서번호"),
    PiiCategory("nationality", "국적"),
    PiiCategory("education_history", "학력"),
    PiiCategory("academic_major", "전공"),
    PiiCategory("job_position", "직책"),
    PiiCategory("age", "나이"),
    PiiCategory("height", "신장"),
    PiiCategory("weight", "체중"),
)

CATEGORY_BY_LABEL = {category.label: category for category in PII_CATEGORIES}
