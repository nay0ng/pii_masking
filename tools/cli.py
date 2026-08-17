"""OCR 파싱·탐지·마스킹 결과를 터미널에서 확인하는 개발용 실행 파일.

제품 코드의 진입점은 ``run_full_pipeline.py``이다. 이 파일은 token 좌표와 중간
문자열을 직접 확인해야 할 때 사용하며 배포용 탐지 모듈에는 포함하지 않는다.
"""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pii_masking.aegis_detector import detect_aegis_spans
from pii_masking.categories import CATEGORY_BY_LABEL, PII_CATEGORIES
from pii_masking.detector_engine import merge_pii_spans
from pii_masking.pii_detectors import detect_all_pii_spans, implemented_category_labels
from pii_masking.image_masker import build_masked_output_path, mask_document_image
from pii_masking.ocr_parser import load_ocr_results
from pii_masking.paths import RESULTS_DIR
from pii_masking.ocr_preprocessing import normalize_for_search, preprocess_document
from tools.debug import render_token_debug_image


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR 결과 JSON의 구조를 확인합니다.")
    parser.add_argument("results_json", help="예: business_results.json")
    parser.add_argument("--debug-token", type=int, help="첫 번째 문서에서 이 token_id의 좌표를 이미지에 표시합니다.",)
    parser.add_argument("--output", default=str(RESULTS_DIR / "diagnostics" / "debug_token.png"), help="디버그 이미지 저장 경로",)
    parser.add_argument("--find", help="첫 번째 문서의 정규화 문자열에서 찾을 값",)
    parser.add_argument("--show-lines", action="store_true", help="첫 번째 문서를 좌표 기준으로 재구성한 줄을 출력합니다.",)
    parser.add_argument("--mask-output-dir", default=str(RESULTS_DIR / "masked"), help="마스킹된 이미지 저장 폴더 (기본값: results/masked)",)
    parser.add_argument("--mask-padding", type=int, default=2, help="OCR bbox 바깥으로 추가할 마스킹 여백 픽셀 (기본값: 2)",)
    parser.add_argument("--without-aegis", action="store_true", help="AEGIS를 제외하고 ko-pii·자체 규칙 결과만 사용합니다.",)
    args = parser.parse_args()

    # 배치 OCR에서는 일부 문서만 실패할 수 있다. 실패 문서는 경고로 알리고,
    # 정상 응답을 받은 나머지 문서의 탐지와 마스킹은 계속 진행한다.
    documents = load_ocr_results(args.results_json, skip_failed_responses=True)
    print(f"문서 수: {len(documents)}")
    for document in documents:
        print(
            f"- {document.image_path} | "
            f"type={document.document_type or '-'} | tokens={len(document.tokens)}"
        )

    implemented = implemented_category_labels()
    print(
        f"PII 카테고리: 목표 {len(PII_CATEGORIES)}개 | "
        f"현재 탐지 {len(implemented)}개 "
        f"({', '.join(implemented)})"
    )

    # 별도 탐지 옵션 없이 기본 실행에서 모든 문서를 전처리하고,
    # 현재 레지스트리에 연결된 개인정보 탐지기를 전부 실행한다.
    processed_documents = tuple(preprocess_document(document) for document in documents)
    total_span_count = 0
    masked_results = []
    for processed in processed_documents:
        rule_spans = detect_all_pii_spans(processed)
        aegis_spans = (
            ()
            if args.without_aegis
            else detect_aegis_spans(processed, scope="semantic")
        )
        spans = merge_pii_spans(rule_spans, aegis_spans)
        total_span_count += len(spans)
        if spans:
            print(f"\n문서: {processed.document.image_path}")
            for span in spans:
                category = CATEGORY_BY_LABEL[span.entity_type]
                print(
                    f"- {category.korean_name}({span.entity_type}) | "
                    f"lines={list(span.line_ids)} | "
                    f"raw_span=[{span.raw_start}, {span.raw_end}) | "
                    f"text={span.text!r} | token_ids={list(span.token_ids)} | "
                    f"boxes={list(span.boxes)} | "
                    f"ocr_confidence={span.ocr_confidence:.2f} | "
                    f"detector_confidence={span.detector_confidence:.2f} | "
                    f"rule={span.rule_id} | "
                    f"review_required={span.review_required}"
                )

        masked_output = build_masked_output_path(
            processed.document.image_path,
            args.mask_output_dir,
        )
        masked_results.append(
            mask_document_image(
                processed,
                spans,
                masked_output,
                padding=args.mask_padding,
            )
        )
    print(f"\n전체 개인정보 span 수: {total_span_count}")
    print(f"마스킹 이미지 저장: {len(masked_results)}개")
    for result in masked_results:
        print(
            f"- {result.output_path} | "
            f"spans={result.span_count} | boxes={result.box_count}"
        )

    if args.debug_token is not None:
        output = render_token_debug_image(
            documents[0],
            token_id=args.debug_token,
            output_path=args.output,
        )
        token = documents[0].tokens[args.debug_token]
        print(f"디버그 이미지: {output}")
        print(f"표시한 토큰: {token.text!r}, 좌표={token.box}")

    if args.find:
        processed = processed_documents[0]
        index = processed.index
        normalized_query = normalize_for_search(args.find)
        start = index.normalized_text.find(normalized_query)
        if start < 0:
            print(f"검색 실패: {args.find!r}")
        else:
            end = start + len(normalized_query)
            raw_start, raw_end = index.raw_span_for_normalized_span(start, end)
            token_ids = index.token_ids_for_normalized_span(start, end)
            print(f"검색 값: {args.find!r}")
            print(f"normalized span: [{start}, {end})")
            print(f"raw span: [{raw_start}, {raw_end})")
            print(f"연결된 token IDs: {token_ids}")
            print(f"원본 토큰: {[index.tokens[item].text for item in token_ids]}")

    if args.show_lines:
        processed = processed_documents[0]
        print(
            f"원본 이미지: {processed.image_width}x{processed.image_height}, "
            f"OCR 처리 이미지: "
            f"{processed.ocr_image_width}x{processed.ocr_image_height}, "
            f"좌표 배율: ({processed.scale_x:.4f}, {processed.scale_y:.4f}), "
            f"재구성한 줄 수: {len(processed.lines)}"
        )
        for line in processed.lines:
            print(f"line {line.line_id:02d} {list(line.token_ids)}: {line.text}")


if __name__ == "__main__":
    main()
