# `pii_masking` 소스코드 안내

이 폴더는 OCR JSON을 읽어 개인정보를 탐지하고, 탐지 문자열을 원본 이미지의
좌표로 되돌려 마스킹하는 핵심 엔진이다.

배치 실행 방법은 프로젝트 루트의 [`README.md`](../../README.md)를 참고하고,
엔진의 세부 규칙과 SearchView 설명은
[`ENGINE_COMPLETE_GUIDE.md`](../../docs/architecture/ENGINE_COMPLETE_GUIDE.md)를 참고한다.

## 1. 전체 처리 흐름

```text
OCR JSON
  ↓ ocr_parser.py
OcrDocument와 OCR token
  ↓ ocr_preprocessing.py
줄·읽기 순서·문자 위치가 정리된 PreprocessedDocument
  ↓ search_views.py / table_field_detectors.py
탐지용 문자열 SearchView와 원본 token 위치 대응표
  ├─ ko_pii_detector.py + structure_rules.py
  └─ aegis_detector.py
  ↓ detector_engine.py
token ID와 bbox가 포함된 공통 PiiSpan
  ↓ detection_policy.py
규칙·NER 결과 병합 및 관리자 정책 적용
  ↓ image_masker.py
원본 이미지 좌표 마스킹
```

전체 실행 순서를 코드로 보려면 [`detection_pipeline.py`](detection_pipeline.py)의
`detect_document_pii()`부터 읽는 것이 가장 쉽다.

## 2. 추천 코드 검토 순서

처음부터 모든 파일을 읽을 필요는 없다. 아래 순서대로 이동하면 한 문서가 처리되는
과정을 따라갈 수 있다.

1. [`detection_pipeline.py`](detection_pipeline.py): 규칙 탐지, AEGIS 실행, 결과 병합 순서
2. [`models.py`](models.py): OCR 문서와 최종 탐지 결과가 사용하는 기본 자료구조
3. [`ocr_parser.py`](ocr_parser.py): OCR JSON을 `OcrDocument`로 변환
4. [`ocr_preprocessing.py`](ocr_preprocessing.py): token 좌표로 줄과 문자 위치 구성
5. [`search_views.py`](search_views.py): 탐지기에 전달할 문자열과 역매핑 정보 생성
6. [`pii_detectors.py`](pii_detectors.py): 개인정보 유형별 공개 탐지 함수
7. [`ko_pii_detector.py`](ko_pii_detector.py): ko-pii 실행 결과를 공통 형식으로 변환
8. [`aegis_detector.py`](aegis_detector.py): AEGIS ONNX NER 실행
9. [`detector_engine.py`](detector_engine.py): 문자 span을 token과 bbox로 역매핑
10. [`detection_policy.py`](detection_policy.py): 관리자 정책으로 최종 결과 결정
11. [`image_masker.py`](image_masker.py): 최종 bbox를 이미지에 그림

## 3. 파일별 역할

### 실행 흐름과 외부 공개 기능

| 파일 | 역할 |
|---|---|
| [`__init__.py`](__init__.py) | 외부 코드에서 사용할 클래스와 함수를 공개한다. 내부 구현 전체를 직접 import하지 않도록 공개 범위를 정한다. |
| [`detection_pipeline.py`](detection_pipeline.py) | 문서 한 건의 규칙 탐지, AEGIS 탐지, 정책 병합을 순서대로 실행한다. 이미지 렌더링은 하지 않는다. |
| [`pii_detectors.py`](pii_detectors.py) | `detect_address()`, `detect_person_name()`처럼 개인정보 유형별 진입 함수를 제공한다. 실제 규칙은 이 파일에 직접 작성하지 않는다. |
| [`categories.py`](categories.py) | 엔진 내부 개인정보 유형의 이름과 표시 정보를 정의한다. |
| [`paths.py`](paths.py) | 설정, 이미지, OCR JSON, 모델, 결과 폴더의 표준 경로와 기존 상대 경로를 실제 파일 경로로 해석한다. |

### OCR 입력과 전처리

| 파일 | 역할 |
|---|---|
| [`models.py`](models.py) | `Box`, `OcrToken`, `OcrLine`, `OcrDocument`, `DocumentIndex`, `PreprocessedDocument`를 정의한다. |
| [`ocr_parser.py`](ocr_parser.py) | OCR 서버 JSON의 `word_boxes`, 문자열, 신뢰도와 이미지 정보를 읽는다. 실패한 OCR 응답은 경고 또는 오류로 구분한다. |
| [`ocr_preprocessing.py`](ocr_preprocessing.py) | token bbox의 세로 겹침과 좌표를 이용해 줄과 읽기 순서를 재구성한다. NFKC 정규화와 문자별 raw/token 대응표도 만든다. |
| [`search_views.py`](search_views.py) | OCR 원문을 수정하지 않고 탐지 목적별 문자열을 만든다. 각 문자에서 원본 raw 위치와 token ID로 돌아가는 배열을 함께 유지한다. |
| [`table_field_detectors.py`](table_field_detectors.py) | 표 헤더와 같은 열 아래의 값을 bbox로 연결해 `table_field` SearchView를 만든다. 값을 개인정보로 확정하지 않고 ko-pii 또는 AEGIS가 다시 판단하게 한다. |

### 개인정보 탐지기 연결

| 파일 | 역할 |
|---|---|
| [`ko_pii_detector.py`](ko_pii_detector.py) | SearchView를 ko-pii `detect_all()`에 전달하고 ko-pii 라벨·문자 위치·근거·점수를 공통 형식으로 바꾼다. OCR 길이를 유지하는 일부 보정과 정책상 허용된 번호 복구도 이 경계에서 처리한다. |
| [`aegis_detector.py`](aegis_detector.py) | AEGIS Personal PII NER의 tokenizer와 ONNX Runtime session을 로드하고 BIO 결과를 내부 개인정보 유형으로 변환한다. |
| [`structure_rules.py`](structure_rules.py) | ko-pii의 단일 문자열 규칙만으로 표현하기 어려운 최소 문서 구조를 보완한다. 현재는 인접 1~3줄의 은행·계좌·선택적 예금주, 관리자 추가 이름 필드, 좌표로 제한된 주소 필드를 처리한다. |

`structure_rules.py`는 ko-pii 원본 코드가 아니다. 이 프로젝트가 OCR의 줄과 bbox
구조를 활용하기 위해 추가한 규칙이다. 특정 파일명이나 실제 개인정보 값에 맞춘
예외는 넣지 않고, 여러 문서에서 반복 가능한 구조만 유지한다.

### span과 좌표 처리

| 파일 | 역할 |
|---|---|
| [`detector_engine.py`](detector_engine.py) | ko-pii와 AEGIS의 서로 다른 반환값을 `ProviderMatch`로 받아 공통 `PiiSpan`으로 변환한다. SearchView 문자 범위를 OCR token ID, raw span과 bbox로 역매핑하고 중복 결과를 정리한다. |
| [`address_expander.py`](address_expander.py) | 탐지된 주소 뒤에 이어지는 동·층·호 같은 주소성 token을 좌표와 문자열 조건으로 제한적으로 확장한다. 새로운 개인정보 유형을 판정하는 파일은 아니다. |
| [`image_masker.py`](image_masker.py) | 최종 `PiiSpan.boxes`를 원본 이미지 범위 안으로 제한하고 마스킹 사각형을 그린다. |

### 관리자 설정과 결과 정책

| 파일 | 역할 |
|---|---|
| [`policy.py`](policy.py) | `policy.json`을 읽고 저장하는 설정 계층이다. 민감도 preset, 선택 유형, AEGIS 범위, 체크섬과 문맥 설정을 `MaskingPolicyConfig`로 만든다. |
| [`detection_policy.py`](detection_policy.py) | 설정을 실제 판단에 적용하는 실행 계층이다. 규칙과 NER 중 무엇을 실행할지, OR/AND 병합, 유형·위험도·체크섬·문맥·검수 후보 처리 방식을 결정한다. |

두 정책 파일의 차이는 다음과 같다.

```text
policy.py
→ JSON과 관리자 입력을 Python 설정 객체로 변환

detection_policy.py
→ 그 설정을 실제 PiiSpan에 적용해 mask/review/exclude 결정
```

## 4. 핵심 자료구조

### `OcrToken`

OCR이 인식한 단어나 문자 묶음 하나다. 다음 정보를 가진다.

- token ID
- OCR 문자열
- OCR 신뢰도
- 원본 이미지 bbox

### `PreprocessedDocument`

OCR token을 줄과 읽기 순서로 정리한 문서다. 원본 문자열, 탐지용 정규화 문자열,
각 문자가 어느 token에서 왔는지 확인할 수 있는 `DocumentIndex`를 포함한다.

### `SearchView`

탐지기에 실제로 전달하는 문자열 한 개다. 문자열만 저장하는 것이 아니라 각 문자의
원본 raw 위치와 token ID도 함께 저장한다. 그래서 공백을 정리한 문자열에서 찾은
개인정보도 원본 bbox로 돌아갈 수 있다.

주요 mode는 다음과 같다.

| mode | 용도 |
|---|---|
| `token_spaced` | OCR token 사이를 한 칸 유지한 기본 탐지 문자열 |
| `format_compact` | 번호·이메일처럼 형식을 구성하는 token 조각만 붙인 문자열 |
| `field_compact` | 이름 필드 주변처럼 제한된 한 필드 구간을 붙인 문자열 |
| `table_field` | 표 헤더와 같은 열 아래의 값을 연결한 문자열 |
| `address_field` | 주소 필드와 좌표가 맞는 인접 주소 줄을 연결한 문자열 |
| `structured_block` | 인접 1~3줄에서 은행·계좌번호·선택적 예금주를 확인하는 제한 문자열 |

### `ProviderMatch`

ko-pii와 AEGIS 결과를 같은 방식으로 처리하기 위한 중간 형식이다. 개인정보 유형,
SearchView 안의 시작·끝 위치, 문자열, 탐지 점수와 규칙 ID를 가진다.

### `PiiSpan`

마스킹 직전의 공통 탐지 결과다. 다음 정보가 들어 있다.

- 개인정보 유형과 탐지 문자열
- raw 문자 범위와 줄 ID
- OCR token ID
- 원본 이미지 bbox 목록
- OCR 신뢰도와 탐지기 점수
- 규칙 ID, 근거, 위험도와 검수 여부

## 5. ko-pii와 AEGIS가 결합되는 위치

두 탐지기는 서로의 결과를 입력으로 사용하지 않는다.

```text
같은 PreprocessedDocument
  ├─ ko-pii + 최소 문서 구조 규칙 → rule_spans
  └─ AEGIS NER                    → ner_spans

rule_spans + ner_spans
  ↓ detection_policy.py
중복·정책 처리 후 final_spans
```

- ko-pii는 번호 형식, 키워드, 사전, 날짜와 체크섬 검증에 적합하다.
- AEGIS는 문맥에서 이름·주소 등의 NER 범위를 찾는다.
- `detector_engine.py`가 두 결과를 같은 `PiiSpan` 형식으로 맞춘다.
- `detection_policy.py`가 OR, AND, 규칙 우선 실행과 최종 정책을 적용한다.

ko-pii 결과의 `confidence`는 규칙 근거에 부여된 점수이고, AEGIS의 confidence는
모델 출력 확률이다. 의미가 다르므로 두 값을 단순히 같은 정확도 수치로 비교하면
안 된다. OCR 신뢰도, ko-pii·자체 규칙 점수, AEGIS 모델 confidence와 최종 정책의
관계는 프로젝트 루트 [`README.md`](../../README.md)의
“화면에 표시되는 신뢰도와 최종 마스킹 판단” 절에 예시와 함께 정리되어 있다.

## 6. 기능을 수정할 때 어디를 봐야 하는가

| 변경하려는 내용 | 먼저 볼 파일 |
|---|---|
| OCR JSON 필드나 bbox 형식 | `ocr_parser.py`, `models.py` |
| 줄 묶기·읽기 순서·정규화 | `ocr_preprocessing.py` |
| token을 붙이는 방법이나 인접 줄 범위 | `search_views.py` |
| 표 헤더 아래 값 연결 | `table_field_detectors.py` |
| ko-pii 라벨 매핑·체크섬 생략 정책 | `ko_pii_detector.py` |
| AEGIS 라벨·threshold·CPU/CUDA | `aegis_detector.py` |
| 은행·계좌·예금주 같은 문서 구조 | `structure_rules.py` |
| 탐지 문자를 bbox로 바꾸는 방법 | `detector_engine.py` |
| 규칙·NER 실행 방식과 mask/review/exclude | `detection_policy.py` |
| `policy.json` 읽기·저장·민감도 preset | `policy.py` |
| 주소 뒤 동·층·호 확장 | `address_expander.py` |
| 마스크 사각형과 여백 | `image_masker.py` |

## 7. 코드 경계에서 지켜야 할 사항

1. OCR 원문과 bbox는 전처리에서 임의로 수정하지 않는다.
2. 탐지용 문자열을 변경하면 `char_to_raw`, `char_to_token` 길이와 위치도 반드시
   함께 유지한다.
3. 탐지기 adapter는 결과를 `ProviderMatch`로 변환하고, bbox 계산은
   `detector_engine.py`에서 공통으로 처리한다.
4. 관리자 설정 때문에 ko-pii나 C++ 코드를 매번 수정하지 않도록, 변경 가능한 값은
   `MaskingPolicyConfig`와 `SearchViewSettings`를 통해 전달한다.
5. 특정 테스트 이미지의 문구나 좌표를 직접 조건으로 넣지 않는다.
6. 새 구조 규칙은 ko-pii 또는 AEGIS로 처리할 수 없는 이유와 오탐 방지 조건을
   함께 설명할 수 있어야 한다.

## 8. 이 폴더에서 직접 실행하지 않는 것

이 폴더는 import되는 라이브러리 코드다. 전체 실행과 화면 확인은 상위 경로에서
수행한다.

- 배치 실행: `../../run_full_pipeline.py`
- 파이프라인 확인 화면: `../../apps/pipeline_viewer.py`
- 관리자 정책 실험 화면: `../../apps/streamlit_app.py`
- 기본 설정: `../../config/policy.json`
