# OCR 개인정보 탐지·마스킹 파이프라인

OCR 서버가 생성한 JSON과 원본 이미지를 입력받아 다음 작업을 수행하는 전달용
프로젝트입니다.

```text
OCR token·bbox 파싱
→ 좌표 복원과 줄 재구성
→ ko-pii 규칙 탐지와 AEGIS NER 탐지
→ 탐지 결과를 원본 bbox로 역매핑
→ 정책 적용과 이미지 마스킹
```

이 프로젝트에는 OCR 모델과 OCR 서버 호출 코드는 포함되어 있지 않습니다.
입력으로 OCR 결과 JSON과 해당 원본 이미지가 필요합니다.

현재 엔진의 자체 규칙, SearchView, ko-pii·AEGIS 연결, 정책, 후처리 전체 설명은
[`docs/architecture/ENGINE_COMPLETE_GUIDE.md`](docs/architecture/ENGINE_COMPLETE_GUIDE.md)에 정리되어 있습니다.

세 가지 방식으로 확인할 수 있습니다.

1. `apps/pipeline_viewer.py`: 단계별 결과를 화면에서 검수
2. `run_full_pipeline.py`: OCR 결과부터 최종 마스킹까지 배치 실행
3. `run_aegis.py`: 문자열만 AEGIS NER 모델에 전달해 결과 확인

## 검수 화면 실행

전달받은 사람이 확인할 기본 화면은 **`apps/pipeline_viewer.py`**입니다.
`apps/streamlit_app.py`가 아니라 다음 명령으로 실행합니다.

```powershell
cd C:\전달받은경로\ocr_pii_masking_pipeline

uv venv .venv --python 3.12

uv pip install --python .\.venv\Scripts\python.exe `
  -r .\requirements-full-cpu.txt

.\.venv\Scripts\python.exe -m streamlit run `
  .\apps\pipeline_viewer.py
```

브라우저가 자동으로 열리지 않으면 터미널에 표시된 `Local URL`로 접속합니다.
기본 샘플은 별도 경로 수정 없이 다음 값을 사용합니다.

- OCR 결과 폴더: `sample_data`
- 원본 이미지 폴더: `sample_data`
- 정책 JSON: `config/policy.json`
- AEGIS 모델 폴더: `models`
- OCR 결과 파일: `test_docs_ocr_results.json`

화면은 실제 처리 순서와 동일하게 구성되어 있습니다.

1. **OCR:** 서버의 `full_text`, OCR token, bbox와 OCR 신뢰도
2. **전처리·SearchView:** 재구성 line과 탐지기에 전달된 문자열
3. **개인정보 탐지:** ko-pii 후보, AEGIS 원시 예측·threshold, 최종 정책 결과
4. **span·bbox:** 탐지 문자열이 연결된 원본 이미지 bbox
5. **최종 마스킹:** 원본 이미지와 마스킹 결과 비교

`apps/streamlit_app.py`는 더 많은 관리자 실험 항목을 확인하기 위한 내부 상세
화면입니다. 전달 결과를 검수하거나 코드 흐름을 확인할 때는
`apps/pipeline_viewer.py`를 기준으로 봅니다.

### 뷰어 코드 검토 순서

`apps/pipeline_viewer.py`의 `main()`부터 시작해 다음 함수를 Ctrl+클릭합니다.

```text
pipeline_viewer.main()
→ load_and_process()
→ load_ocr_results()
→ preprocess_document()
→ detect_document_pii()
→ detect_all_pii_spans() / detect_aegis_spans()
→ create_policy_run_result()
→ render_masked_image()
```

화면 표와 bbox 미리보기만 만드는 함수는 `apps/dashboard_support.py`에 있습니다.
실제 탐지·정책·마스킹 엔진은 모두 `src/pii_masking`에 있습니다.

## 전체 프로세스 빠른 실행

PowerShell에서 아래 명령을 순서대로 실행합니다.

```powershell
cd C:\전달받은경로\ocr_pii_masking_pipeline

uv venv .venv --python 3.12

uv pip install --python .\.venv\Scripts\python.exe `
  -r .\requirements-full-cpu.txt

.\.venv\Scripts\python.exe .\run_full_pipeline.py `
  --ocr-results .\sample_data\test_docs_ocr_results.json `
  --image-root .\sample_data `
  --policy .\config\policy.json `
  --output-dir .\sample_full_output `
  --provider cpu

Get-Content -Raw -Encoding UTF8 `
  .\sample_full_output\pipeline_results.json
```

정상 실행되면 다음 결과가 생성됩니다.

- `sample_full_output/pipeline_results.json`
- `sample_full_output/masked/0228train285_masked.jpg`

주요 입력 인자는 다음과 같습니다.

| 인자 | 필수 | 설명 |
|---|---:|---|
| `--ocr-results` | ✓ | OCR 서버 결과가 저장된 JSON 파일 |
| `--image-root` |  | OCR JSON 안의 상대 이미지 경로를 찾을 기준 폴더 |
| `--policy` |  | 적용할 관리자 정책 JSON. 생략하면 `config/policy.json` 사용 |
| `--output-dir` |  | 결과 JSON과 마스킹 이미지를 저장할 폴더 |
| `--provider` |  | AEGIS 실행 장치. `cpu` 또는 `cuda` |

다른 정책을 비교하려면 정책 파일만 바꿔서 실행할 수 있습니다.

```powershell
python .\run_full_pipeline.py `
  --ocr-results .\sample_data\test_docs_ocr_results.json `
  --image-root .\sample_data\test_docs `
  --policy .\config\policy_sensitive.json `
  --output-dir .\results\test_docs_sensitive `
  --provider cpu
```

실제로 사용한 정책 파일 경로는 결과의
`pipeline_results.json → pipeline.policy_file`에도 저장됩니다.

전체 처리 순서:

```text
OCR 결과 JSON + 원본 이미지
→ OCR token·bbox 파싱
→ 원본 이미지 좌표 복원
→ 좌표 기준 줄 재구성
→ 탐지용 문자열과 문자 위치 생성
→ ko-pii·문맥 규칙 탐지
→ AEGIS NER 탐지
→ 탐지 결과 병합
→ token ID·bbox 역매핑
→ 마스킹 이미지와 결과 JSON 저장
```

### 숫자형 개인정보의 OCR 혼동문자 복구

OCR은 숫자를 모양이 비슷한 영문자나 기호로 반환할 수 있습니다. 이 파이프라인은
원본 OCR 결과를 수정하지 않고, 개인정보 탐지기에 전달할 별도의 SearchView에서만
다음 문자를 제한적으로 숫자로 복구합니다.

| OCR 문자 | 탐지용 숫자 |
|---|---:|
| `O`, `o`, `Q` | `0` |
| `I`, `i`, `l`, `L`, `|`, `!` | `1` |
| `Z`, `z` | `2` |
| `S`, `s` | `5` |
| `G` | `6` |
| `B` | `8` |

예를 들어 OCR이 다음처럼 반환한 경우:

```text
OCR 원본: 900S01-I234B67
탐지 입력: 900501-1234867
```

복구된 각 문자는 원래 OCR token과 bbox 연결을 그대로 유지합니다. 따라서 ko-pii가
복구 문자열에서 개인정보를 찾으면 실제 마스킹은 원본 이미지의
`900S01-I234B67` 위치에 적용됩니다.

오탐을 줄이기 위해 다음 조건에서만 복구합니다.

- `pattern_strictness`가 `ocr_tolerant`일 때만 실행합니다.
- 혼동문자 가까이에 실제 숫자가 있는 경우에만 치환합니다.
- 전화번호, 계좌·카드번호, 주민등록번호, 외국인등록번호, 운전면허번호,
  사업자등록번호, 법인등록번호, 건강보험증번호에만 적용합니다.
- 이메일, URL, 이름, 주소에는 적용하지 않습니다.
- 여권번호의 `S`처럼 정상 영문 접두어를 손상시킬 수 있어 여권번호에는 적용하지
  않습니다.

웹 백엔드에서는 민감도 화면을 제공하지 않아도 다음 설정을 고정하여 사용할 수
있습니다.

```python
policy_config = MaskingPolicyConfig(
    pattern_strictness=PatternStrictness.OCR_TOLERANT,
)
```

이 복구는 빠진 숫자를 새로 추정하는 기능이 아닙니다. OCR이 문자를 완전히
누락했거나 숫자를 서로 다른 숫자로 잘못 읽은 경우는 복구하지 못합니다.

단독 NER 실행과 전체 파이프라인은 별도의 모델 구현을 갖지 않습니다.
둘 다 `src/pii_masking/aegis_detector.py`의 같은 AEGIS ONNX 실행 코드를
사용합니다. 규칙 탐지 → AEGIS 실행 여부 결정 → 결과 병합 과정도
`src/pii_masking/detection_pipeline.py`의 `detect_document_pii()` 한 곳에서 실행됩니다.

`3. 개인정보 탐지 → AEGIS → 모델 원시 예측·threshold`에서는 모델이 예측한
token entity의 confidence와 적용 threshold를 함께 표시합니다. 따라서 모델이
아예 예측하지 않은 경우와 예측했지만 threshold 때문에 제외된 경우를 구분할 수
있습니다. 기본 화면은 threshold 제외 예측만 보여줍니다.

### 화면에 표시되는 신뢰도와 최종 마스킹 판단

먼저 결론부터 정리하면, 화면의 숫자가 `0.8`이라고 해서 “개인정보일 확률이
80%”라는 뜻은 아닙니다. 현재 파이프라인에는 서로 의미가 다른 값이 함께 있으며,
출처를 확인하지 않고 숫자만 비교하면 안 됩니다.

| 값 | 누가 만드는가 | 의미 | 현재 사용 방법 |
|---|---|---|---|
| OCR 신뢰도 | OCR 서버 | 이미지 글자를 해당 token 문자열로 읽은 정도 | `PiiSpan.ocr_confidence`에 저장하고 화면에 표시한다. 현재 자동 마스킹 필터로 사용하지 않는다. |
| ko-pii 규칙 점수 | ko-pii의 유형별 탐지 코드 | 해당 규칙이 확인한 형식·체크섬·사전·문맥 근거의 강도 | `PiiSpan.detector_confidence`에 저장한다. 확률이나 실제 정확도가 아니다. |
| 자체 구조 규칙 점수 | 이 프로젝트의 `structure_rules.py` 또는 제한적 복구 코드 | 은행·계좌·예금주, 관리자 필드, 주소 필드 같은 구조 근거의 강도 | ko-pii 결과와 같은 `detector_confidence` 칸에 저장하지만 출처는 `context:` 규칙 ID로 구분한다. |
| AEGIS 모델 confidence | AEGIS ONNX 모델과 후처리 코드 | NER token의 softmax 확률을 한 span 단위로 평균 낸 값 | 라벨별 threshold를 넘는지 판단한 뒤 `PiiSpan.detector_confidence`에 저장한다. |

#### 1. OCR 신뢰도

OCR 신뢰도는 개인정보 탐지 점수가 아닙니다. 예를 들어 OCR 서버가
`900101-1234567`을 token 네 개로 반환했고 신뢰도가 각각 `0.98`, `0.91`,
`0.87`, `0.95`라면, 해당 PII span의 OCR 신뢰도는 가장 낮은 값인 `0.87`로
저장됩니다.

```text
OCR token 신뢰도: 0.98, 0.91, 0.87, 0.95
PiiSpan.ocr_confidence = 0.87
```

이 값은 “주민등록번호일 가능성”이 아니라 “OCR이 이 글자를 제대로 읽었는가”에
관한 값입니다. 현재 정책 코드는 OCR 신뢰도가 낮다는 이유만으로 span을 제외하지
않고, 진단 화면에서 OCR 문제를 구분할 수 있도록 저장만 합니다.

#### 2. ko-pii 규칙 점수

ko-pii는 AI 분류 모델이 아니라 유형별 규칙 탐지기입니다. 모든 유형에 공통으로
적용되는 하나의 confidence 계산식이 없고, 각 규칙 파일이 조건을 통과했을 때
반환할 점수를 직접 정합니다.

따라서 다음 두 값을 같은 방식으로 해석하면 안 됩니다.

```text
주민등록번호 confidence 0.7
→ 생년월일과 성별·세기 코드는 유효하지만 체크섬이 맞지 않음

문맥형 주소 confidence 0.6
→ 주소 키워드와 주소 사전 근거로 찾은 문맥형 주소
```

둘 다 숫자가 낮지만 `0.7 = 70%`, `0.6 = 60%`라는 의미가 아닙니다. 서로 다른
탐지기가 서로 다른 조건을 표현하기 위해 정한 규칙 점수입니다.

현재 사용하는 주요 ko-pii 점수는 다음과 같습니다.

| 개인정보 유형 | 확인 조건 | ko-pii 점수 |
|---|---|---:|
| 주민·외국인등록번호 | 날짜·세기/성별 코드·체크섬 통과 | `1.0` |
| 주민·외국인등록번호 | 날짜·세기/성별 코드는 유효하지만 체크섬 불일치 | `0.7` |
| 사업자·법인등록번호 | 지정 형식과 체크섬 통과 | `1.0` |
| 카드번호 | 카드 형식과 Luhn 체크섬 통과 | `1.0` |
| 이메일·전화번호·IP 주소 | 해당 탐지기의 전체 형식 통과 | `1.0` |
| 생년월일 | 사용한 날짜 표현과 문맥 규칙에 따라 | `0.95`, `0.9`, `0.85` |
| 여권·계좌·건강보험번호·URL | 해당 유형의 형식 또는 필수 문맥 통과 | `0.9` |
| 운전면허·팩스·차량번호 | 해당 유형의 형식 통과 | `0.85` |
| 도로명 주소 | 행정구역·도로명·주소 사전 조건 통과 | `0.8` |
| 지번 주소 | 행정구역·지번·주소 사전 조건 통과 | `0.75` |
| 문맥형 주소 | 주소 키워드와 사전 조건 통과 | `0.6` |
| 단독 행정구역 주소 | 단독 행정구역 조건 통과 | `0.7` |
| 이름 | 필드명·성씨·직책·조사·음절·주변 이름 등의 근거를 합산 | 후보마다 다름 |

이름 점수는 고정값이 아닙니다. 예를 들어 `대표자: 홍길동`처럼 강한 필드명이
있거나 직책·성씨·이름 음절 등의 근거가 추가되면 점수가 올라갑니다. 반대로 일반
한글 단어와 구분할 근거가 부족하면 threshold를 통과하지 못해 결과가 나오지 않을
수 있습니다.

#### 3. 이 프로젝트의 자체 구조 규칙 점수

다음 점수는 ko-pii가 만든 값이 아닙니다. OCR 문서 구조를 보완하기 위해 현재
프로젝트 코드가 부여합니다.

| 자체 결과 | 조건 | 자체 점수 |
|---|---|---:|
| 체크섬 생략 번호 후보 | 관리자가 체크섬 검사를 끈 상태에서 유형별 번호 형식을 확인한다. 사업자·법인·카드는 강한 필드명도 요구한다. | `0.65` |
| 관리자 추가 이름 필드 | 등록한 필드명 뒤 값을 찾고 ko-pii PERSON으로 재검증 | `0.85` |
| 좌표로 제한한 주소 필드 | 주소 필드와 값 열을 확인하고 행정구역·주소 숫자를 검증 | `0.85` |
| 은행·계좌 구조의 계좌번호 | 인접 1~3줄에서 은행명과 10~16자리 계좌 형식을 함께 확인 | 기본 `0.60`~`0.85`, 계좌 필드 확인 시 최대 `0.90` |
| 은행·계좌 구조의 예금주 | 위 구조가 성립하고 예금주 후보를 ko-pii PERSON으로 재검증 | 기본 `0.60`~`0.85`, 예금주 필드 확인 시 최대 `0.90` |

`table_field`는 자체적으로 이름을 확정하거나 `0.82` 같은 점수를 만들지 않습니다.
표 헤더와 같은 열 아래의 값을 제한된 SearchView로 구성한 다음, ko-pii 또는
AEGIS가 그 값을 다시 판단합니다. 따라서 최종 점수는 실제로 결과를 만든 탐지기의
점수입니다.

규칙 ID로 점수의 출처를 구분할 수 있습니다.

```text
ko_pii:RRN
→ ko-pii 주민등록번호 규칙이 만든 결과

context:ACCOUNT_CONTEXT
→ 이 프로젝트의 은행·계좌 구조 규칙이 만든 결과

ner:aegis-v2:SURNAME+GIVENNAME
→ AEGIS가 만든 이름 결과
```

#### 4. AEGIS 모델 confidence와 threshold

AEGIS는 모델이므로 여기의 confidence는 규칙 점수와 달리 softmax 출력에서
계산됩니다. 먼저 모델 token마다 BIO 라벨과 확률을 구하고, 라벨별 threshold를
통과한 연속 token을 하나의 개인정보 span으로 합칩니다. 최종 span confidence는
합쳐진 token 확률의 평균입니다. 다만 이 값도 별도 calibration 평가 없이 실제
정답 확률이나 문서 전체 정확도로 해석하면 안 됩니다.

```text
홍  B-GIVENNAME  0.91
길  I-GIVENNAME  0.88
동  I-GIVENNAME  0.86

AEGIS span confidence = (0.91 + 0.88 + 0.86) / 3 = 0.883
```

AEGIS 모델 파일 자체에 프로젝트 운영 threshold가 들어 있는 것은 아닙니다.
`aegis_detector.py`의 `ENTITY_THRESHOLDS`에 이름 `0.85`, 주소 구성요소
`0.80` 또는 `0.90` 등 라벨별 기준을 별도로 정했습니다. 민감도 설정의
`aegis_threshold_adjustment`가 이 기준을 올리거나 내립니다.

```text
기본 이름 threshold 0.85
민감도 조정 -0.05
실제 적용 threshold 0.80
```

모델이 이름을 전혀 예측하지 않은 경우와, 이름으로 예측했지만 threshold 아래라
제외된 경우는 서로 다릅니다. Pipeline Viewer의 AEGIS 원시 예측 표에서 예측값,
적용 threshold와 통과 여부를 함께 확인할 수 있습니다.

#### 5. 점수와 최종 마스킹 여부는 같은 것이 아님

탐지기는 후보를 만들고 점수와 근거를 붙일 뿐입니다. 최종 마스킹 여부는
`detection_policy.py`가 아래 순서로 결정합니다.

```text
탐지 후보 PiiSpan
  ↓ 선택한 개인정보 유형인가?
  ↓ minimum_confidence 이상인가?
  ↓ minimum_risk_level 이상인가?
  ↓ 체크섬 정책을 통과했는가?
  ↓ 필요한 필드명·키워드·문맥 근거가 있는가?
  ↓ review 후보 처리 방식은 무엇인가?
최종 mask / review / exclude
```

현재 기본 `policy.json`의 `minimum_confidence`는 `0.0`입니다. 즉, 기본 설정에서는
ko-pii 규칙 점수나 AEGIS 모델 confidence를 하나의 공통 숫자 경계로 잘라내지
않습니다. 두 값의 의미가 다르기 때문입니다.

관리자가 `minimum_confidence`를 `0.8`로 올리면 현재 구현상 ko-pii, 자체 규칙,
AEGIS 결과에 모두 `0.8`이 적용됩니다. 하지만 이것은 문맥형 주소 `0.6`과 AEGIS
확률 `0.6`을 같은 기준으로 비교하는 것이므로, 충분한 평가 데이터로 유형별 영향을
확인하기 전에는 공통 threshold 사용에 주의해야 합니다.

#### 6. 실제 판단 예시

예시 A — 정상 주민등록번호:

```text
OCR 신뢰도: 0.93
ko-pii 규칙 점수: 1.0
근거: 날짜 유효 + 세기/성별 코드 유효 + 체크섬 정상
정책: 주민등록번호 선택, 체크섬 사용
결과: 마스킹 후보에 포함
```

예시 B — 체크섬이 틀린 주민등록번호 모양:

```text
OCR 신뢰도: 0.99
ko-pii 규칙 점수: 0.7
근거: 날짜와 세기/성별 코드는 유효, 체크섬 불일치
정책의 checksum_invalid_action이 mask   → 마스킹
정책의 checksum_invalid_action이 review → 검수 후보
정책의 checksum_invalid_action이 exclude → 제외
```

OCR을 아주 선명하게 읽어 `0.99`가 나와도 체크섬 불일치가 해결되는 것은 아닙니다.
반대로 OCR 신뢰도가 낮더라도 우연히 체크섬까지 통과할 수 있으므로 두 값을 따로
봐야 합니다.

예시 C — 체크섬 검사를 끈 사업자등록번호 후보:

```text
ko-pii 원래 결과: 체크섬 실패로 결과 없음
프로젝트 보완 결과: 강한 사업자번호 필드명 + 번호 형식 확인
자체 규칙 점수: 0.65
근거: checksum:skipped
최종 처리: 민감도와 검수 정책에 따라 마스킹 또는 검수
```

예시 D — AEGIS가 찾은 이름:

```text
AEGIS span confidence: 0.88
적용 threshold: 0.85
모델 단계: 통과
review_required: True
review_handling이 mask_all이면 마스킹
review_handling이 exclude_review이면 최종 결과에서 제외
```

따라서 화면을 볼 때는 신뢰도 숫자 하나만 보지 말고 다음 항목을 함께 확인해야
합니다.

1. `rule_id`: ko-pii, 자체 구조 규칙, AEGIS 중 누가 찾았는가
2. `evidence`: 형식, 체크섬, 필드명, 사전, 문맥 중 무엇을 확인했는가
3. `OCR 신뢰도`: 원본 글자 인식이 안정적인가
4. `detector_confidence`: 해당 탐지기의 내부 점수는 얼마인가
5. 정책 판단 사유: 최종적으로 왜 mask, review 또는 exclude가 되었는가

## 디렉터리

```text
ocr_pii_masking_pipeline/
├─ models/
│  ├─ onnx/model_quantized.onnx
│  ├─ config.json
│  ├─ vocab.txt
│  ├─ tokenizer.json
│  ├─ tokenizer_config.json
│  └─ special_tokens_map.json
├─ ko-pii/                    ko-pii 원본 저장소 복사본(.git 제외)
├─ src/pii_masking/           배포 대상 OCR·PII 마스킹 모듈
│  ├─ aegis_detector.py        AEGIS ONNX 실행·bbox 매핑
│  └─ detection_pipeline.py    규칙·AEGIS·정책 결합 공통 흐름
├─ apps/                      Streamlit 화면과 화면 표시용 코드
│  ├─ pipeline_viewer.py       전달·검수용 기본 화면
│  ├─ streamlit_app.py         관리자 정책 상세 test 화면
│  └─ dashboard_support.py
├─ tools/                     배포 모듈에서 제외한 개발 도구
│  ├─ cli.py                   터미널 중간 결과 확인
│  └─ debug.py                 OCR token 좌표 이미지 확인
├─ sample_data/
│  ├─ test_docs_ocr_results.json
│  └─ test_docs/               검증에 사용한 전체 이미지
├─ run_full_pipeline.py       전체 프로세스 실행기
├─ run_aegis.py               문자열 전용 NER 실행기
├─ config/
│  ├─ policy.json             기본 탐지·마스킹 정책
│  └─ POLICY_README.md        정책 항목 설명
├─ docs/
│  ├─ README.md               프로젝트 문서 목록
│  ├─ architecture/           엔진 구조와 전체 동작 설명
│  ├─ guides/                 코드리뷰·학습 가이드
│  ├─ models/                 사용 모델 정보
│  └─ specs/                  기능명세서
├─ requirements-cpu.txt
├─ requirements-full-cpu.txt
├─ requirements-gpu-cuda12.txt
└─ requirements-gpu-cuda13.txt
```

`src/pii_masking`에는 실제 처리에 필요한 데이터 모델, OCR 파싱, 전처리,
SearchView 생성, ko-pii·AEGIS 연결, 정책 적용과 이미지 마스킹 코드만 둡니다.
Streamlit 표·미리보기 생성 코드는 `apps`, 터미널 점검 코드는 `tools`에 두어
배포 모듈의 경계와 개발 도구를 구분했습니다.

`ko-pii`는 번들 안의 저장소에서 `ko-pii/src`를 직접 사용하므로 별도로
설치하지 않습니다. 원본 소스 외에도 문서, 테스트, 예제, 데이터 파일을
포함하며 Git 이력인 `.git` 폴더만 제외했습니다.

## 실제 OCR 결과 실행

OCR JSON의 최상위 객체는 `이미지 경로: OCR 응답` 형식입니다.

```json
{
  "images/document-001.jpg": {
    "resultCode": "0000",
    "formResult": {
      "fieldResults": []
    }
  }
}
```

원본 이미지 경로가 `D:\ocr_test\images\document-001.jpg`라면 다음처럼
실행합니다.

```powershell
.\.venv\Scripts\python.exe .\run_full_pipeline.py `
  --ocr-results D:\ocr_test\ocr_results.json `
  --image-root D:\ocr_test `
  --policy .\config\policy.json `
  --output-dir D:\ocr_test\output `
  --provider cpu
```

`--image-root`는 OCR JSON 안의 상대 이미지 경로 앞에 붙는 폴더입니다.
OCR JSON에 절대 이미지 경로가 들어 있다면 생략할 수 있습니다.

일부 OCR 실패 응답을 제외하고 나머지 문서를 계속 처리하려면
`--skip-failed`를 추가합니다.

```powershell
.\.venv\Scripts\python.exe .\run_full_pipeline.py `
  --ocr-results D:\ocr_test\ocr_results.json `
  --image-root D:\ocr_test `
  --policy .\config\policy.json `
  --output-dir D:\ocr_test\output `
  --provider cpu `
  --skip-failed
```

탐지만 실행하고 이미지를 만들지 않으려면 `--no-mask`를 추가합니다.

## 전체 프로세스 출력

`pipeline_results.json`에는 문서별로 다음 정보가 저장됩니다.

- 좌표 배율, OCR token 수, 재구성된 줄
- ko-pii·문맥 규칙 탐지 결과
- AEGIS NER 탐지 결과
- 중복·정책 적용 후 최종 결과
- 개인정보 유형과 문자열
- token ID와 bbox
- OCR 신뢰도와 탐지 신뢰도
- 적용 규칙과 검수 필요 여부
- 단계별 처리시간
- 최종 마스킹 이미지 경로

## 문자열 전용 AEGIS 실행

```powershell
uv venv .venv-ner --python 3.12
uv pip install --python .\.venv-ner\Scripts\python.exe `
  -r .\requirements-cpu.txt

.\.venv-ner\Scripts\python.exe .\run_aegis.py `
  --text "대표자 홍길동 주소 서울특별시 강남구 테헤란로 123" `
  --provider cpu `
  --scope all
```

## CUDA 실행

CPU용 `onnxruntime`과 GPU용 `onnxruntime-gpu`를 같은 가상환경에 함께
설치하지 않습니다. 별도 가상환경을 권장합니다.

CUDA 12 환경:

```powershell
uv venv .venv-gpu --python 3.12
uv pip install --python .\.venv-gpu\Scripts\python.exe `
  -r .\requirements-full-gpu-cuda12.txt

.\.venv-gpu\Scripts\python.exe .\run_full_pipeline.py `
  --ocr-results .\sample_data\test_docs_ocr_results.json `
  --image-root .\sample_data `
  --policy .\config\policy.json `
  --output-dir .\sample_full_output_cuda `
  --provider cuda `
  --limit 1
```

CUDA 13 환경은 `requirements-full-gpu-cuda13.txt`를 사용합니다.

## 문자열 전용 입력 형식

단일 객체, 객체 배열, 또는 `items` 배열을 받을 수 있습니다.

```json
{
  "items": [
    {
      "id": "document-001-line-03",
      "text": "대표자 홍길동 주소 서울특별시 강남구 테헤란로 123"
    }
  ]
}
```

긴 문서는 최대 입력 길이를 넘지 않도록 줄이나 문단 단위로 나누어
입력하는 것을 권장합니다.

## 문자열 전용 출력 형식

```json
{
  "category": "person_name",
  "entity_types": ["SURNAME", "GIVENNAME"],
  "start": 4,
  "end": 7,
  "text": "홍길동",
  "confidence": 0.97,
  "rule_id": "ner:aegis-v2:SURNAME+GIVENNAME"
}
```

- `start`: 입력 문자열에서 개인정보가 시작되는 문자 위치
- `end`: 개인정보 직후 문자 위치이며 해당 위치는 포함하지 않음
- `text`: `input_text[start:end]`
- `entity_types`: AEGIS 원본 entity 라벨
- `category`: AEGIS entity를 사용하기 쉽게 묶은 카테고리
- `confidence`: 병합된 WordPiece별 모델 확률의 평균

## AEGIS 주요 동작

- `BertWordPieceTokenizer(lowercase=False)`
- 최대 길이 512 WordPiece
- ONNX Runtime 최적화 `ORT_ENABLE_ALL`
- softmax 후 token별 최고 확률 라벨 선택
- 유형별 threshold와 `threshold_adjustment`
- 이름의 `SURNAME + GIVENNAME` 병합
- 주소의 `CITY + STREET + BUILDINGNUM` 인접 span 병합
- 한 글자 이름, 지나치게 짧은 주소 후보 제거
- CPU 또는 CUDA provider 선택과 CUDA의 CPU fallback
