# `config/policy.json` 설정 안내

`config/policy.json`은 OCR 이후 개인정보 탐지, 탐지기 결합, 후보 필터링,
이미지 마스킹 범위를 결정하는 실행 설정 파일입니다.

표준 JSON에는 `//`, `#` 주석을 넣을 수 없습니다. 설명은 이 문서에서
확인하고 `config/policy.json`에는 실제 설정값만 입력합니다.

## 1. 최상위 설정

| 옵션 | 의미 | 허용값 |
|---|---|---|
| `schema_version` | 정책 파일 형식 버전 | 현재 `1` |
| `detection` | 개인정보 탐지·결합·필터 설정 | JSON 객체 |
| `masking` | 이미지 마스킹 표시 설정 | JSON 객체 |

## 2. 탐지기 실행 방식

### `execution_mode`

ko-pii·자체 규칙과 AEGIS NER를 어떤 순서로 실행할지 정합니다.

| 값 | 동작 |
|---|---|
| `rule_only` | ko-pii·자체 규칙만 실행 |
| `ner_only` | AEGIS NER만 실행 |
| `always_both` | 두 탐지기를 항상 실행 |
| `rule_first_fallback` | 규칙 탐지 후 선택한 NER 담당 유형이 부족할 때만 AEGIS 실행 |
| `category_routing` | 개인정보 유형별로 규칙 탐지기와 AEGIS의 담당 범위를 나누어 실행 |

### `merge_mode`

두 탐지기가 반환한 결과를 최종 후보로 결합하는 방법입니다.

| 값 | 동작 |
|---|---|
| `any` | 둘 중 하나만 탐지해도 유지 |
| `agreement` | 같은 위치를 두 탐지기가 함께 찾은 경우만 유지 |
| `category_priority` | 이름·주소 등 NER 담당 유형은 AEGIS를 우선하고 나머지는 규칙 결과 유지 |

## 3. 마스킹 대상 유형

### `selected_categories`

최종 결과와 마스킹에 포함할 개인정보 유형입니다. 목록에서 제외한 유형은
탐지되더라도 최종 마스킹에서 제외됩니다.

| 값 | 의미 |
|---|---|
| `account_card_number` | 계좌번호·카드번호 |
| `academic_major` | 전공 |
| `address` | 주소 |
| `age` | 나이 |
| `business_registration_number` | 사업자등록번호 |
| `company_name` | 회사명 |
| `corporate_registration_number` | 법인등록번호 |
| `court_case_number` | 법원 사건번호 |
| `credential_secret` | API 키·토큰·비밀번호 등 인증 비밀정보 |
| `date` | 날짜·생년월일 |
| `document_number` | 문서번호 |
| `driver_license_number` | 운전면허번호 |
| `drug_code` | 의약품 코드 |
| `education_history` | 학력 |
| `email_address` | 이메일 주소 |
| `employee_number` | 사번 |
| `foreigner_registration_number` | 외국인등록번호 |
| `health_insurance_number` | 건강보험증번호 |
| `height` | 키 |
| `id_card_number` | 기타 신분증 번호 |
| `ip_address` | IP 주소 |
| `job_position` | 직책·직급 |
| `land_lot_number` | 지번·필지번호 |
| `nationality` | 국적 |
| `passport_number` | 여권번호 |
| `person_name` | 사람 이름 |
| `petition_number` | 민원·청원 번호 |
| `phone_number` | 전화번호 |
| `postal_code` | 우편번호 |
| `prescription_number` | 처방전 번호 |
| `resident_registration_number` | 주민등록번호 |
| `time` | 시간 |
| `url` | URL |
| `username` | 사용자 ID |
| `vehicle_plate_number` | 차량번호 |
| `weight` | 몸무게 |

## 4. 탐지 민감도와 패턴 복구

### `sensitivity_level`

관리자 화면에서 선택한 민감도 프리셋을 기록합니다. 실제 동작은
`pattern_strictness`, 체크섬, 문맥, AEGIS 임계값 등의 세부 설정으로
결정됩니다.

| 값 | 의미 |
|---|---|
| `1` | 정확 우선 |
| `2` | 기본 |
| `3` | 민감 |
| `4` | 최대 민감도 |

### `pattern_strictness`

숫자형 개인정보에서 OCR 공백·구분자 오류를 어디까지 복구할지 정합니다.

| 값 | 동작 |
|---|---|
| `exact` | OCR 문자열이 정확한 형식일 때만 탐지 |
| `normalized` | 토큰 사이 공백과 하이픈 분리를 정리 |
| `recovered` | 중복 하이픈과 일부 구분자 오류까지 복구 |
| `ocr_tolerant` | 숫자 구간의 `O/0`, `I/1` OCR 혼동까지 제한적으로 복구 |

## 5. AEGIS 설정

### `aegis_scope`

| 값 | 동작 |
|---|---|
| `semantic` | 이름·주소 관련 AEGIS 라벨만 사용 |
| `all` | `aegis_entity_types`에서 선택한 전체 라벨 사용 |

### `aegis_entity_types`

AEGIS가 반환한 세부 NER 라벨 중 사용할 항목입니다.

| 라벨 | 의미 |
|---|---|
| `SURNAME` | 성 |
| `GIVENNAME` | 이름 |
| `CITY` | 도시·행정구역 |
| `STREET` | 도로·거리 주소 |
| `ZIPCODE` | 우편번호 |
| `BUILDINGNUM` | 건물번호 |
| `USERNAME` | 사용자 ID |
| `EMAIL` | 이메일 |
| `TELEPHONENUM` | 전화번호 |
| `DATEOFBIRTH` | 생년월일 |
| `CREDITCARDNUMBER` | 카드번호 |
| `IDCARD` | 신분증 번호 |
| `IP_ADDRESS` | IP 주소 |
| `PASSWORD` | 비밀번호 |
| `ACCOUNTNUM` | 계좌번호 |
| `DRIVERLICENSENUM` | 운전면허번호 |
| `TIME` | 시간 |
| `COMPANY` | 회사명 |

### `aegis_threshold_adjustment`

- 허용 범위: `-0.5`~`0.5`
- `0.0`: 모델 기본 임계값 사용
- 음수: 임계값을 낮춰 더 많은 후보 탐지
- 양수: 임계값을 높여 더 확실한 후보만 탐지

## 6. 신뢰도·위험도 설정

| 옵션 | 의미 |
|---|---|
| `minimum_confidence` | 최종 후보로 허용할 최소 탐지 신뢰도. 범위 `0.0`~`1.0`; `0.0`은 신뢰도로 제외하지 않음 |
| `minimum_risk_level` | ko-pii 최소 위험도. `1`=INFO 이상, `2`=LOW 이상, `3`=MEDIUM 이상, `4`=HIGH 이상, `5`=CRITICAL만 허용 |

ko-pii 신뢰도는 규칙 충족 정도를 나타내는 정책값이고, AEGIS 신뢰도는
모델의 예측 확률에 가깝습니다. 두 값을 같은 의미의 점수로 단순 비교하면
안 됩니다.

## 7. 체크섬 Gate

### `checksum_validation_enabled`

- `true`: 지원 번호의 체크섬 결과를 정책 판단에 사용
- `false`: 체크섬 불일치만으로 후보를 제거하지 않음

### `checksum_invalid_action`

체크섬이 틀린 후보의 처리 방법입니다.

| 값 | 동작 |
|---|---|
| `mask` | 자동 마스킹 |
| `review` | 검수 필요 후보로 표시 |
| `exclude` | 자동 마스킹에서 제외 |

## 8. 문맥·Anchor Gate

| 옵션 | 의미 |
|---|---|
| `minimum_context_evidence_count` | 요구할 문맥 근거 Gate 수. `0`~`3`; 필드명·키워드·문맥 사전 근거 사용 |
| `missing_context_action` | 요구한 문맥 근거를 충족하지 못한 후보 처리: `mask`, `review`, `exclude` |
| `anchor_required_categories` | 필드명·접두어·주변 키워드 anchor를 반드시 요구할 개인정보 유형 목록 |
| `missing_anchor_action` | 필수 anchor가 없을 때 처리: `mask`, `review`, `exclude` |

`anchor_required_categories`가 빈 목록이면 어떤 유형에도 anchor를 강제하지
않습니다.

## 9. 검수 후보 처리

### `review_handling`

| 값 | 동작 |
|---|---|
| `mask_all` | 검수 표시와 관계없이 최종 마스킹 |
| `exclude_review` | 검수 필요 후보는 자동 마스킹에서 제외 |
| `review_only` | 검수 필요 후보만 최종 결과에 포함 |

## 10. 문서 구조와 사용자 필드명

| 옵션 | 의미 |
|---|---|
| `max_adjacent_lines` | 주소 등 구조 탐지에서 연결할 최대 인접 줄 수. 허용값 `1`, `2`, `3` |
| `custom_person_field_labels` | 추가 이름 필드명. 예: `["학생명", "수강생"]` |
| `custom_address_field_labels` | 추가 주소 필드명. 예: `["배송지", "설치장소"]` |

## 11. 이미지 마스킹

### `masking.padding`

탐지 bbox의 상하좌우에 추가할 검은색 마스크 여백입니다.

- 단위: pixel
- 허용 범위: `0`~`50`
- 현재값 `2`: bbox보다 상하좌우 2px 넓게 마스킹

## 12. 현재 기본 정책 요약

현재 `config/policy.json`은 다음과 같이 동작합니다.

- ko-pii·자체 규칙과 AEGIS를 모두 실행
- 한 탐지기만 찾아도 최종 후보로 유지
- 등록된 모든 개인정보 유형을 마스킹 대상으로 선택
- 기본 민감도와 `normalized` 패턴 복구 사용
- AEGIS 전체 18개 라벨 사용
- 체크섬 불일치 후보도 마스킹
- 최소 신뢰도와 문맥 근거 수로 후보를 제외하지 않음
- 검수 필요 후보도 모두 마스킹
- 주소 등 인접 구조를 최대 3줄까지 탐색
- 최종 bbox보다 상하좌우 2px 넓게 마스킹
