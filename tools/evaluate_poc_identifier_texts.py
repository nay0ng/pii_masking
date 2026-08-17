"""PoC 고유식별정보 4종을 합성 OCR 텍스트로 일괄 점검한다.

이 스크립트는 이미지 OCR 자체를 평가하지 않는다. JSONL에 적힌 줄과 token을
OCR 서버가 반환했다고 가정하고 가상 bbox를 만든 뒤, 현재 프로젝트의 전처리,
SearchView 생성, ko-pii·자체 규칙 탐지, 정책 필터까지 실행한다.

평가 대상:
    - 주민등록번호
    - 외국인등록번호
    - 여권번호
    - 운전면허번호
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
KO_PII_SRC_DIR = PROJECT_ROOT / "ko-pii" / "src"
for import_path in (SRC_DIR, KO_PII_SRC_DIR):
    path_text = str(import_path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from ko_pii.checksum.rrn_checksum import compute_check_digit
from ko_pii.checksum.corp_reg_checksum import (
    compute_check_digit as compute_corporate_check_digit,
)
from pii_masking.detection_pipeline import detect_document_pii
from pii_masking.detection_policy import (
    DetectorExecutionMode,
    GateAction,
    ReviewHandling,
)
from pii_masking.models import Box, OcrDocument, OcrToken
from pii_masking.ocr_preprocessing import preprocess_document
from pii_masking.policy import MaskingPolicyConfig
from pii_masking.search_views import PatternStrictness, SearchViewSettings


POC_CATEGORIES = (
    "resident_registration_number",
    "foreigner_registration_number",
    "passport_number",
    "driver_license_number",
)

CATEGORY_NAMES = {
    "resident_registration_number": "주민등록번호",
    "foreigner_registration_number": "외국인등록번호",
    "passport_number": "여권번호",
    "driver_license_number": "운전면허번호",
}

STRICTNESS_VALUES = tuple(value.value for value in PatternStrictness)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="합성 OCR 텍스트로 PoC 고유식별정보 4종을 일괄 평가합니다.",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "sample_data" / "poc_identifier_text_cases.jsonl",
        help="합성 OCR 텍스트 JSONL 경로",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "poc_identifier_text_evaluation",
        help="상세 CSV와 요약 JSON을 저장할 폴더",
    )
    parser.add_argument(
        "--pattern-strictness",
        choices=(*STRICTNESS_VALUES, "all"),
        default="ocr_tolerant",
        help="검색 문자열 복구 수준. all은 네 수준을 모두 비교합니다.",
    )
    parser.add_argument(
        "--checksum",
        choices=("on", "off"),
        default="off",
        help="체크섬 Gate 사용 여부. PoC 데모 권장 기본값은 off입니다.",
    )
    parser.add_argument(
        "--generate-cases",
        action="store_true",
        help="내장 합성 케이스로 JSONL을 다시 만든 뒤 평가합니다.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="합성 JSONL만 만들고 평가는 실행하지 않습니다.",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="필수 탐지·오탐 방지 기대값과 다르면 종료 코드 1을 반환합니다.",
    )
    return parser.parse_args()


def make_valid_registration_number(
    birth: str,
    gender_digit: str,
    serial_five: str,
) -> str:
    """주민·외국인등록번호용 체크섬 유효 합성 번호를 만든다."""
    first_twelve = birth + gender_digit + serial_five
    check_digit = compute_check_digit(first_twelve)
    return f"{birth}-{gender_digit}{serial_five}{check_digit}"


def make_corporate_collision_number(
    front_six: str,
    seventh_digit: str,
    serial_five: str,
) -> str:
    """법인 체크섬은 맞지만 외국인번호처럼 보이는 합성 충돌값을 만든다."""
    first_twelve = front_six + seventh_digit + serial_five
    check_digit = compute_corporate_check_digit(first_twelve)
    return f"{front_six}-{seventh_digit}{serial_five}{check_digit}"


def fullwidth(text: str) -> str:
    """ASCII 숫자와 영문자를 OCR에서 나올 수 있는 전각 문자로 바꾼다."""
    converted = []
    for char in text:
        if "0" <= char <= "9" or "A" <= char <= "Z" or "a" <= char <= "z":
            converted.append(chr(ord(char) + 0xFEE0))
        elif char == "-":
            converted.append("－")
        else:
            converted.append(char)
    return "".join(converted)


def token(text: str, x: int | None = None) -> str | dict[str, Any]:
    if x is None:
        return text
    return {"text": text, "x": x}


def case(
    case_id: str,
    scenario: str,
    expected_categories: list[str],
    ocr_lines: list[list[str | dict[str, Any]]],
    description: str,
    expectation: str = "required",
    expected_categories_checksum_off: list[str] | None = None,
) -> dict[str, Any]:
    item = {
        "case_id": case_id,
        "scenario": scenario,
        "expectation": expectation,
        "expected_categories": expected_categories,
        "ocr_lines": ocr_lines,
        "description": description,
        "synthetic": True,
    }
    if expected_categories_checksum_off is not None:
        item["expected_categories_checksum_off"] = expected_categories_checksum_off
    return item


def registration_cases(
    prefix: str,
    category: str,
    field_label: str,
    values: list[str],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, value in enumerate(values, start=1):
        front, back = value.split("-")
        base_id = f"{prefix}_{index:02d}"
        cases.extend(
            [
                case(
                    f"{base_id}_plain",
                    "plain",
                    [category],
                    [[value]],
                    "필드명 없이 번호만 한 token으로 인식",
                ),
                case(
                    f"{base_id}_field_inline",
                    "field_inline",
                    [category],
                    [[field_label, ":", value]],
                    "필드명 오른쪽에 번호가 있는 일반 형식",
                ),
                case(
                    f"{base_id}_field_above",
                    "field_above",
                    [category],
                    [[field_label], [value]],
                    "필드명 아래 줄에 번호가 있는 표 형식",
                ),
                case(
                    f"{base_id}_split_tokens",
                    "split_tokens",
                    [category],
                    [[field_label, ":", front, "-", back]],
                    "앞자리·하이픈·뒷자리를 서로 다른 token으로 인식",
                ),
                case(
                    f"{base_id}_digits_as_tokens",
                    "digits_as_tokens",
                    [category],
                    [[field_label, ":", *list(front), "-", *list(back)]],
                    "각 숫자를 별도 token으로 인식",
                ),
                case(
                    f"{base_id}_dot_separator",
                    "separator_variant",
                    [category],
                    [[f"{front}.{back}"]],
                    "하이픈을 점으로 인식",
                ),
                case(
                    f"{base_id}_slash_separator",
                    "separator_variant",
                    [category],
                    [[f"{front}/{back}"]],
                    "하이픈을 슬래시로 인식",
                ),
                case(
                    f"{base_id}_fullwidth",
                    "unicode_normalization",
                    [category],
                    [[fullwidth(value)]],
                    "전각 숫자와 전각 하이픈으로 인식",
                ),
                case(
                    f"{base_id}_duplicate_hyphen",
                    "duplicate_separator",
                    [category],
                    [[field_label, ":", f"{front}--{back}"]],
                    "하이픈을 두 번 인식",
                ),
                case(
                    f"{base_id}_cross_line_number",
                    "cross_line",
                    [category],
                    [[field_label, ":", front], [token("-", 250), token(back, 270)]],
                    "번호 앞자리와 뒷자리가 인접한 두 줄로 분리",
                ),
            ]
        )

    sample = values[0]
    front, back = sample.split("-")
    confused = front[:3] + "S" + front[4:] + "-" + back
    cases.append(
        case(
            f"{prefix}_ocr_confusion",
            "ocr_digit_confusion",
            [category],
            [[field_label, ":", confused]],
            "숫자 5를 영문 S로 인식; ocr_tolerant에서 복구 대상",
        )
    )
    cases.append(
        case(
            f"{prefix}_missing_digit",
            "missing_character",
            [category],
            [[field_label, ":", sample[:-1]]],
            "숫자 한 자가 OCR에서 완전히 누락됨; 현재 자동 추론하지 않음",
            expectation="known_limit",
        )
    )
    return cases


def passport_cases(values: list[str]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, value in enumerate(values, start=1):
        prefix_length = 2 if value[:2] in {"PP", "PM", "PS", "PO", "PD", "PR", "PT"} else 1
        prefix = value[:prefix_length]
        digits = value[prefix_length:]
        base_id = f"passport_{index:02d}"
        cases.extend(
            [
                case(base_id + "_plain", "plain", ["passport_number"], [[value]], "여권번호 한 token"),
                case(base_id + "_field_inline", "field_inline", ["passport_number"], [["여권번호", ":", value]], "여권번호 필드 오른쪽 값"),
                case(base_id + "_field_above", "field_above", ["passport_number"], [["여권번호"], [value]], "필드 아래 줄 값"),
                case(base_id + "_split_tokens", "split_tokens", ["passport_number"], [["여권번호", ":", prefix, digits]], "접두어와 숫자가 별도 token"),
                case(base_id + "_digits_as_tokens", "digits_as_tokens", ["passport_number"], [["여권번호", ":", prefix, *list(digits)]], "숫자가 한 글자씩 분리"),
                case(base_id + "_fullwidth", "unicode_normalization", ["passport_number"], [[fullwidth(value)]], "전각 영문·숫자로 인식"),
                case(base_id + "_sentence", "embedded_sentence", ["passport_number"], [["신청인의", "여권번호는", value, "입니다"]], "일반 문장 안의 여권번호"),
            ]
        )

    sample = values[0]
    cases.append(
        case(
            "passport_lowercase_prefix",
            "letter_case_error",
            ["passport_number"],
            [["여권번호", ":", sample.lower()]],
            "OCR이 영문 접두어를 소문자로 인식; 현재 대문자화하지 않음",
            expectation="known_limit",
        )
    )
    cases.append(
        case(
            "passport_ocr_digit_confusion",
            "ocr_digit_confusion",
            ["passport_number"],
            [["여권번호", ":", sample[:4] + "S" + sample[5:]]],
            "여권 숫자를 S로 인식; 현재 숫자 복구 대상 카테고리에 여권은 없음",
            expectation="known_limit",
        )
    )
    cases.append(
        case(
            "passport_missing_digit",
            "missing_character",
            ["passport_number"],
            [["여권번호", ":", sample[:-1]]],
            "여권번호 숫자 한 자 누락",
            expectation="known_limit",
        )
    )
    foreign_examples = (
        ("foreign_passport_two_letters", "AB1234567"),
        ("foreign_passport_nine_digits", "123456789"),
        ("foreign_passport_mixed", "C01X23456"),
        ("foreign_passport_long_mixed", "N7A2345678"),
    )
    for case_id, value in foreign_examples:
        cases.append(
            case(
                case_id + "_field_anchored",
                "foreign_passport_field_anchored",
                ["passport_number"],
                [["여권번호", ":", value]],
                "한국 여권 prefix 규칙 밖의 해외 발급 여권번호 형식 예시; 국가별 규칙 검토 필요",
                expectation="known_limit",
            )
        )
        cases.append(
            case(
                case_id + "_standalone",
                "foreign_passport_standalone",
                ["passport_number"],
                [[value]],
                "필드명·국가·문서 문맥 없이 해외 여권번호 값만 존재; 일반 문서번호와 구분 어려움",
                expectation="known_limit",
            )
        )
    return cases


def driver_license_cases(values: list[str]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, value in enumerate(values, start=1):
        parts = value.split("-")
        compact = "".join(parts)
        base_id = f"driver_{index:02d}"
        cases.extend(
            [
                case(base_id + "_plain", "plain", ["driver_license_number"], [[value]], "하이픈 포함 운전면허번호"),
                case(base_id + "_field_inline", "field_inline", ["driver_license_number"], [["운전면허번호", ":", value]], "필드 오른쪽 운전면허번호"),
                case(base_id + "_field_above", "field_above", ["driver_license_number"], [["면허번호"], [value]], "필드 아래 줄 운전면허번호"),
                case(base_id + "_split_tokens", "split_tokens", ["driver_license_number"], [["운전면허번호", ":", parts[0], "-", parts[1], "-", parts[2], "-", parts[3]]], "구성요소와 하이픈이 별도 token"),
                case(base_id + "_compact_with_anchor", "compact_anchor", ["driver_license_number"], [["운전면허번호", ":", compact]], "하이픈이 없지만 바로 앞 필드명이 있음"),
                case(base_id + "_fullwidth", "unicode_normalization", ["driver_license_number"], [[fullwidth(value)]], "전각 숫자·하이픈"),
                case(base_id + "_duplicate_hyphen", "duplicate_separator", ["driver_license_number"], [["운전면허번호", ":", value.replace("-", "--", 1)]], "첫 하이픈을 중복 인식"),
            ]
        )

    sample = values[0]
    parts = sample.split("-")
    cases.append(
        case(
            "driver_compact_without_anchor",
            "negative_collision",
            [],
            [["관리번호", ":", "".join(parts)]],
            "하이픈 없는 12자리 숫자에 운전면허 필드가 없어 오탐 방지",
        )
    )
    cases.append(
        case(
            "driver_invalid_region",
            "invalid_format",
            [],
            [["운전면허번호", ":", "99-20-123456-78"]],
            "허용되지 않은 지방경찰청 코드",
        )
    )
    cases.append(
        case(
            "driver_ocr_digit_confusion",
            "ocr_digit_confusion",
            ["driver_license_number"],
            [["운전면허번호", ":", sample.replace("5", "S", 1)]],
            "숫자 5를 S로 인식; ocr_tolerant에서 복구 대상",
        )
    )
    cases.append(
        case(
            "driver_missing_digit",
            "missing_character",
            ["driver_license_number"],
            [["운전면허번호", ":", sample[:-1]]],
            "숫자 한 자 누락",
            expectation="known_limit",
        )
    )
    return cases


def build_cases() -> list[dict[str, Any]]:
    rrns = [
        make_valid_registration_number("900101", "1", "23456"),
        make_valid_registration_number("851231", "2", "34567"),
        make_valid_registration_number("010305", "3", "45678"),
        make_valid_registration_number("120229", "4", "56789"),
    ]
    frns = [
        make_valid_registration_number("800215", "5", "12345"),
        make_valid_registration_number("991231", "6", "23456"),
        make_valid_registration_number("010101", "7", "34567"),
        make_valid_registration_number("120229", "8", "45678"),
    ]
    passports = [
        "M12345678",
        "S87654321",
        "G24681357",
        "PP13572468",
        "PO11223344",
        "PD55667788",
    ]
    driver_licenses = [
        "11-20-123456-78",
        "12-99-654321-09",
        "18-07-112233-44",
        "28-24-987654-32",
    ]
    corporate_collision = make_corporate_collision_number(
        "110111",
        "5",
        "43210",
    )

    cases: list[dict[str, Any]] = []
    cases.extend(registration_cases("rrn", "resident_registration_number", "주민등록번호", rrns))
    cases.extend(registration_cases("frn", "foreigner_registration_number", "외국인등록번호", frns))
    cases.extend(passport_cases(passports))
    cases.extend(driver_license_cases(driver_licenses))

    cases.extend(
        [
            case(
                "negative_rrn_bad_date",
                "invalid_date",
                [],
                [["주민등록번호", ":", "991332-1234567"]],
                "존재하지 않는 생년월일. 체크섬 off에서는 6-7 자리 형태를 우선 마스킹",
                expected_categories_checksum_off=["resident_registration_number"],
            ),
            case(
                "negative_frn_bad_date",
                "invalid_date",
                [],
                [["외국인등록번호", ":", "991332-5234567"]],
                "존재하지 않는 생년월일. 체크섬 off에서는 6-7 자리 형태를 우선 마스킹",
                expected_categories_checksum_off=["foreigner_registration_number"],
            ),
            case("negative_passport_zero", "placeholder", [], [["여권번호", ":", "M00000000"]], "전체 0인 여권 placeholder"),
            case("negative_passport_prefix", "invalid_format", [], [["여권번호", ":", "X12345678"]], "지원하지 않는 한국 여권 접두어"),
            case("negative_order_number", "negative_collision", [], [["주문번호", ":", "11-20-123456-78"]], "운전면허처럼 보이는 주문번호이지만 하이픈 형식상 현재 탐지 가능", expectation="known_limit"),
            case("negative_long_digits", "invalid_boundary", [], [["주민번호", ":", "19001011234567890"]], "더 긴 숫자열 내부의 13자리"),
            case(
                "collision_corporate_number_with_field",
                "frn_corporate_collision",
                [],
                [["법인등록번호", ":", corporate_collision]],
                "법인 체크섬이 맞고 앞 6자리와 7번째 숫자는 외국인등록번호처럼 보이는 충돌값",
            ),
            case(
                "collision_corporate_number_without_field",
                "frn_corporate_collision",
                [],
                [[corporate_collision]],
                "필드 문맥이 없는 6자리-7자리 값은 법인번호와 외국인번호를 안전하게 구분하기 어려움",
                expectation="known_limit",
            ),
            case(
                "mixed_all_four",
                "multi_entity_document",
                list(POC_CATEGORIES),
                [
                    ["주민등록번호", ":", rrns[0]],
                    ["외국인등록번호", ":", frns[0]],
                    ["여권번호", ":", passports[0]],
                    ["운전면허번호", ":", driver_licenses[0]],
                ],
                "한 문서에 PoC 4종이 모두 포함",
            ),
        ]
    )
    return cases


def write_cases(path: Path, cases: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for item in cases:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            required = {"case_id", "scenario", "expectation", "expected_categories", "ocr_lines"}
            missing = required - set(item)
            if missing:
                raise ValueError(f"{path}:{line_number} 필수 항목 누락: {sorted(missing)}")
            unknown = set(item["expected_categories"]) - set(POC_CATEGORIES)
            if unknown:
                raise ValueError(f"{path}:{line_number} 알 수 없는 유형: {sorted(unknown)}")
            cases.append(item)
    return cases


def build_document(case_data: dict[str, Any], image_path: Path) -> OcrDocument:
    tokens: list[OcrToken] = []
    full_text_lines = []
    token_id = 0
    for line_index, line in enumerate(case_data["ocr_lines"]):
        x = 30
        y = 30 + line_index * 48
        line_texts = []
        for entry in line:
            if isinstance(entry, str):
                text_value = entry
                requested_x = None
            else:
                text_value = str(entry["text"])
                requested_x = entry.get("x")
            if requested_x is not None:
                x = int(requested_x)
            width = max(14, len(text_value) * 15)
            tokens.append(
                OcrToken(
                    token_id=token_id,
                    text=text_value,
                    confidence=0.99,
                    box=Box(x1=x, y1=y, x2=x + width, y2=y + 26),
                )
            )
            token_id += 1
            x += width + 14
            line_texts.append(text_value)
        full_text_lines.append(" ".join(line_texts))

    return OcrDocument(
        image_path=image_path,
        document_type=None,
        full_text="\n".join(full_text_lines),
        tokens=tuple(tokens),
        ocr_image_width=2400,
        ocr_image_height=1200,
    )


def make_policy(strictness: str, checksum_enabled: bool) -> tuple[MaskingPolicyConfig, SearchViewSettings]:
    policy = MaskingPolicyConfig(
        execution_mode=DetectorExecutionMode.RULE_ONLY,
        selected_categories=frozenset(POC_CATEGORIES),
        pattern_strictness=PatternStrictness(strictness),
        minimum_confidence=0.0,
        minimum_risk_level=1,
        checksum_validation_enabled=checksum_enabled,
        checksum_invalid_action=GateAction.MASK,
        minimum_context_evidence_count=0,
        missing_context_action=GateAction.MASK,
        review_handling=ReviewHandling.MASK_ALL,
        max_adjacent_lines=3,
    )
    search_settings = SearchViewSettings(
        max_adjacent_lines=3,
        pattern_strictness=PatternStrictness(strictness),
        checksum_validation_enabled=checksum_enabled,
    )
    return policy, search_settings


def evaluate_case(
    case_data: dict[str, Any],
    image_path: Path,
    strictness: str,
    checksum_enabled: bool,
) -> dict[str, Any]:
    document = build_document(case_data, image_path)
    processed = preprocess_document(document)
    policy, search_settings = make_policy(strictness, checksum_enabled)
    result = detect_document_pii(processed, policy, search_settings, ner=None)

    detected_categories = sorted({span.entity_type for span in result.final_spans})
    expected_key = (
        "expected_categories"
        if checksum_enabled
        else "expected_categories_checksum_off"
    )
    expected_categories = sorted(
        case_data.get(expected_key, case_data["expected_categories"])
    )
    expectation = case_data["expectation"]
    passed = None if expectation == "known_limit" else detected_categories == expected_categories

    span_records = []
    for span in result.final_spans:
        if span.entity_type not in POC_CATEGORIES:
            continue
        span_records.append(
            {
                "category": span.entity_type,
                "text": span.text,
                "rule_id": span.rule_id,
                "confidence": round(span.detector_confidence, 4),
                "review_required": span.review_required,
                "token_ids": list(span.token_ids),
                "evidence": list(span.evidence),
            }
        )

    return {
        "case_id": case_data["case_id"],
        "scenario": case_data["scenario"],
        "description": case_data.get("description", ""),
        "expectation": expectation,
        "strictness": strictness,
        "checksum": "on" if checksum_enabled else "off",
        "expected_categories": expected_categories,
        "detected_categories": detected_categories,
        "passed": passed,
        "ocr_text": " ↵ ".join(" ".join(str(item.get("text", "")) if isinstance(item, dict) else item for item in line) for line in case_data["ocr_lines"]),
        "spans": span_records,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strictness_values = sorted({row["strictness"] for row in rows})
    summaries = {}
    for strictness in strictness_values:
        selected = [row for row in rows if row["strictness"] == strictness]
        scored = [row for row in selected if row["passed"] is not None]
        passed = sum(row["passed"] is True for row in scored)
        required_positive = [row for row in scored if row["expected_categories"]]
        required_negative = [row for row in scored if not row["expected_categories"]]
        category_summary = {}
        for category in POC_CATEGORIES:
            expected_rows = [
                row
                for row in scored
                if category in row["expected_categories"]
            ]
            unexpected_rows = [
                row
                for row in scored
                if category not in row["expected_categories"]
                and category in row["detected_categories"]
            ]
            category_summary[category] = {
                "name": CATEGORY_NAMES[category],
                "expected_cases": len(expected_rows),
                "detected_expected_cases": sum(
                    category in row["detected_categories"]
                    for row in expected_rows
                ),
                "unexpected_detection_cases": len(unexpected_rows),
            }

        scenario_summary = {}
        for scenario in sorted({row["scenario"] for row in selected}):
            scenario_rows = [
                row
                for row in selected
                if row["scenario"] == scenario and row["passed"] is not None
            ]
            if not scenario_rows:
                continue
            scenario_summary[scenario] = {
                "scored_cases": len(scenario_rows),
                "passed_cases": sum(row["passed"] is True for row in scenario_rows),
            }

        summaries[strictness] = {
            "total_cases": len(selected),
            "scored_cases": len(scored),
            "passed_cases": passed,
            "failed_cases": len(scored) - passed,
            "pass_rate": round(passed / len(scored), 4) if scored else None,
            "required_positive_cases": len(required_positive),
            "positive_passed": sum(row["passed"] is True for row in required_positive),
            "required_negative_cases": len(required_negative),
            "negative_passed": sum(row["passed"] is True for row in required_negative),
            "known_limit_cases": sum(row["expectation"] == "known_limit" for row in selected),
            "by_category": category_summary,
            "by_scenario": scenario_summary,
        }
    return summaries


def write_results(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "details.json"
    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "details.csv"

    details_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "case_id",
                "scenario",
                "expectation",
                "strictness",
                "checksum",
                "expected_categories",
                "detected_categories",
                "passed",
                "ocr_text",
                "description",
            ),
        )
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["expected_categories"] = ",".join(row["expected_categories"])
            csv_row["detected_categories"] = ",".join(row["detected_categories"])
            csv_row.pop("spans")
            writer.writerow(csv_row)


def print_summary(summary: dict[str, Any], rows: list[dict[str, Any]], output_dir: Path) -> None:
    print("\n=== PoC 고유식별정보 4종 합성 텍스트 평가 ===")
    for strictness, item in summary.items():
        print(
            f"[{strictness}] {item['passed_cases']}/{item['scored_cases']} 통과 "
            f"({item['pass_rate'] * 100:.1f}%) | "
            f"필수 탐지 {item['positive_passed']}/{item['required_positive_cases']} | "
            f"오탐 방지 {item['negative_passed']}/{item['required_negative_cases']} | "
            f"알려진 한계 {item['known_limit_cases']}"
        )

    failures = [row for row in rows if row["passed"] is False]
    if failures:
        print("\n--- 기대값 불일치 ---")
        for row in failures[:30]:
            print(
                f"- {row['case_id']} [{row['strictness']}] "
                f"expected={row['expected_categories']} detected={row['detected_categories']}"
            )
        if len(failures) > 30:
            print(f"... 나머지 {len(failures) - 30}건은 details.csv에서 확인")
    print(f"\n결과 폴더: {output_dir.resolve()}")
    print("주의: 이 수치는 합성 OCR 문자열 탐지 결과이며 실제 OCR 정확도나 이미지 bbox 품질이 아닙니다.")


def main() -> None:
    args = parse_args()
    if args.generate_cases or args.generate_only or not args.cases.exists():
        generated_cases = build_cases()
        write_cases(args.cases, generated_cases)
        print(f"합성 케이스 생성: {args.cases.resolve()} ({len(generated_cases)}건)")
    if args.generate_only:
        return

    cases = load_cases(args.cases)
    strictness_values = (
        STRICTNESS_VALUES
        if args.pattern_strictness == "all"
        else (args.pattern_strictness,)
    )
    checksum_enabled = args.checksum == "on"

    rows: list[dict[str, Any]] = []
    with TemporaryDirectory(prefix="poc_pii_text_") as temp_dir:
        image_path = Path(temp_dir) / "synthetic_canvas.png"
        Image.new("RGB", (2400, 1200), color="white").save(image_path)
        for strictness in strictness_values:
            print(f"평가 중: strictness={strictness}, checksum={args.checksum}")
            for index, case_data in enumerate(cases, start=1):
                rows.append(
                    evaluate_case(
                        case_data,
                        image_path,
                        strictness,
                        checksum_enabled,
                    )
                )
                if index % 25 == 0 or index == len(cases):
                    print(f"  {index}/{len(cases)}")

    summary = {
        "case_file": str(args.cases.resolve()),
        "checksum_validation_enabled": checksum_enabled,
        "categories": list(POC_CATEGORIES),
        "note": "합성 OCR 문자열과 가상 bbox를 이용한 탐지 파이프라인 평가이며 OCR 모델 평가는 아님",
        "strictness": summarize(rows),
    }
    write_results(args.output_dir, rows, summary)
    print_summary(summary["strictness"], rows, args.output_dir)

    if args.fail_on_mismatch and any(row["passed"] is False for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
