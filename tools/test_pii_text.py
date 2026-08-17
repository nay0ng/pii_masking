"""터미널에서 짧은 문자열의 ko-pii·AEGIS 판정을 비교한다.

이미지와 OCR JSON을 거치지 않는 텍스트 전용 점검 도구다. AEGIS의 원시 라벨을
그대로 보여주고, IDCARD와 DRIVERLICENSENUM은 PoC 유형으로 사용해도 되는지
형식 검증 결과를 별도로 표시한다.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import re
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENGINE_SRC = PROJECT_ROOT / "src"
KO_PII_SRC = PROJECT_ROOT / "ko-pii" / "src"
for source_dir in (ENGINE_SRC, KO_PII_SRC):
    source_text = str(source_dir)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

from ko_pii import detect_all
from ko_pii.checksum.rrn_checksum import is_valid_checksum
from pii_masking.aegis_detector import AegisOnnxNer


AEGIS_ENTITY_TYPES = (
    "IDCARD",
    "EMAIL",
    "TELEPHONENUM",
    "DRIVERLICENSENUM",
)

KO_PII_LABELS = (
    "RRN",
    "FRN",
    "PASSPORT",
    "DRIVER_LICENSE",
    "EMAIL",
    "PHONE",
)

RRN_CENTURY = {
    "1": 1900,
    "2": 1900,
    "3": 2000,
    "4": 2000,
    "9": 1800,
    "0": 1800,
}

FRN_CENTURY = {
    "5": 1900,
    "6": 1900,
    "7": 2000,
    "8": 2000,
}

DRIVER_KEYWORDS = ("운전면허", "면허번호", "면허증")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="문자열의 ko-pii·AEGIS 개인정보 탐지 결과 비교",
    )
    parser.add_argument(
        "--text",
        help="검사할 문자열. 생략하면 여러 문장을 입력하는 대화형 모드로 실행합니다.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="AEGIS 모델 폴더",
    )
    parser.add_argument(
        "--provider",
        choices=("cpu", "cuda"),
        default="cpu",
        help="AEGIS ONNX 실행 장치",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--threshold-adjustment",
        type=float,
        default=0.0,
        help="AEGIS 내부 유형별 기준값에 더할 값. PoC 권장값은 0.0입니다.",
    )
    return parser.parse_args()


def decode_birth_date(front: str, century: int) -> date | None:
    try:
        birth = date(
            century + int(front[0:2]),
            int(front[2:4]),
            int(front[4:6]),
        )
    except ValueError:
        return None
    if birth > date.today():
        return None
    return birth


def classify_idcard(value: str) -> tuple[str | None, str]:
    """AEGIS IDCARD를 주민번호 또는 외국인번호 후보로 재분류한다."""
    digits = re.sub(r"\D", "", value)
    if len(digits) != 13:
        return None, "숫자 13자리가 아니므로 주민·외국인번호로 확정할 수 없음"

    front = digits[:6]
    discriminator = digits[6]
    century = RRN_CENTURY.get(discriminator)
    category = "resident_registration_number"
    korean_name = "주민등록번호"
    if century is None:
        century = FRN_CENTURY.get(discriminator)
        category = "foreigner_registration_number"
        korean_name = "외국인등록번호"
    if century is None:
        return None, f"7번째 숫자 {discriminator!r}가 주민·외국인 구분 코드가 아님"

    birth = decode_birth_date(front, century)
    if birth is None:
        return None, "앞 6자리와 구분 코드로 계산한 생년월일이 잘못됐거나 미래임"

    checksum = "정상" if is_valid_checksum(digits) else "불일치"
    reason = (
        f"{korean_name} 형식, 생년월일={birth.isoformat()}, "
        f"체크섬={checksum}(PoC 기본 정책에서는 필수 아님)"
    )
    return category, reason


def validate_driver_license(value: str, source_text: str) -> tuple[bool, str]:
    """AEGIS 면허번호 후보가 현재 ko-pii 형식 조건을 만족하는지 확인한다."""
    compact_value = value.strip()
    match = re.fullmatch(
        r"([0-9]{2})-([0-9]{2})-([0-9]{6})-([0-9]{2})",
        compact_value,
    )
    if match is None and re.fullmatch(r"[0-9]{12}", compact_value):
        has_keyword = any(keyword in source_text for keyword in DRIVER_KEYWORDS)
        if not has_keyword:
            return False, "하이픈 없는 12자리는 면허 필드명이 필요함"
        match = re.fullmatch(
            r"([0-9]{2})([0-9]{2})([0-9]{6})([0-9]{2})",
            compact_value,
        )
    if match is None:
        return False, "XX-YY-NNNNNN-CC 형식이 아님"

    region = int(match.group(1))
    if not 11 <= region <= 28:
        return False, f"지역코드 {region:02d}가 허용 범위 11~28 밖임"
    return True, f"운전면허번호 형식과 지역코드 {region:02d} 확인"


def print_aegis_results(text: str, ner: AegisOnnxNer, adjustment: float) -> None:
    started = time.perf_counter()
    detections = ner.detect(
        text,
        scope="all",
        threshold_adjustment=adjustment,
        enabled_entity_types=AEGIS_ENTITY_TYPES,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    print(f"\n[AEGIS 원본 결과] {elapsed_ms:.2f} ms")
    if not detections:
        print("- 탐지 결과 없음")
        return

    for detection in detections:
        entity = "+".join(detection.entity_types)
        print(
            f"- {entity} | text={detection.text!r} | "
            f"confidence={detection.confidence:.4f}"
        )
        if "IDCARD" in detection.entity_types:
            category, reason = classify_idcard(detection.text)
            if category is None:
                print(f"  재분류: 제외 | {reason}")
            else:
                print(f"  재분류: {category} | {reason}")
        if "DRIVERLICENSENUM" in detection.entity_types:
            valid, reason = validate_driver_license(detection.text, text)
            result = "채택" if valid else "제외"
            print(f"  형식 검증: {result} | {reason}")


def print_ko_pii_results(text: str) -> None:
    started = time.perf_counter()
    detections = detect_all(text, include=KO_PII_LABELS, normalize=True)
    elapsed_ms = (time.perf_counter() - started) * 1000

    print(f"\n[ko-pii 결과] {elapsed_ms:.2f} ms")
    if not detections:
        print("- 탐지 결과 없음")
        return
    for detection in detections:
        evidence = ", ".join(detection.evidence) or "없음"
        print(
            f"- {detection.label} | text={detection.text!r} | "
            f"confidence={detection.confidence:.2f} | evidence={evidence}"
        )


def inspect_text(text: str, ner: AegisOnnxNer, adjustment: float) -> None:
    print("\n" + "=" * 78)
    print(f"입력: {text}")
    print_aegis_results(text, ner, adjustment)
    print_ko_pii_results(text)


def run_interactive(ner: AegisOnnxNer, adjustment: float) -> None:
    print("\n검사할 문자열을 입력하세요. 종료하려면 q 또는 exit를 입력합니다.")
    while True:
        try:
            text = input("\nPII> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text.casefold() in {"q", "quit", "exit"}:
            return
        if not text:
            continue
        inspect_text(text, ner, adjustment)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")

    args = parse_args()
    print(f"AEGIS 모델 로딩: {args.model_dir} ({args.provider})")
    ner = AegisOnnxNer(
        args.model_dir,
        execution_provider=args.provider,
        device_id=args.device_id,
    )
    print(f"ONNX Runtime provider: {', '.join(ner.session_providers)}")

    if args.text is not None:
        inspect_text(args.text, ner, args.threshold_adjustment)
    else:
        run_interactive(ner, args.threshold_adjustment)


if __name__ == "__main__":
    main()
