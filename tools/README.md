# 개발 도구

이 폴더의 코드는 OCR·PII 마스킹 배포 모듈에 포함하지 않습니다.

- `cli.py`: OCR JSON을 읽어 전처리 문자열, 탐지 span과 마스킹 결과를 터미널에 출력합니다.
- `debug.py`: 지정한 OCR token의 bbox를 원본 이미지에 표시합니다.

전체 파이프라인의 제품 실행 파일은 프로젝트 루트의 `run_full_pipeline.py`입니다.
이 폴더는 좌표나 중간 결과를 개발 중에 직접 확인할 때만 사용합니다.
# 텍스트에서 AEGIS·ko-pii 결과 비교

이미지나 OCR JSON 없이 짧은 문자열을 직접 입력해 AEGIS와 ko-pii의 결과를
비교할 수 있습니다. AEGIS `IDCARD`는 주민등록번호·외국인등록번호 형식으로
재분류하고, `DRIVERLICENSENUM`은 운전면허번호 형식과 지역코드를 추가로
검증합니다. 이 도구는 진단용이며 원본 이미지 마스킹은 수행하지 않습니다.

한 문장만 검사하려면 다음처럼 실행합니다.

```powershell
python .\tools\test_pii_text.py --text "주민등록번호: <합성 테스트 번호>"
```

여러 문장을 계속 입력하려면 `--text`를 생략합니다.

```powershell
python .\tools\test_pii_text.py
```

입력 프롬프트에서 다음처럼 시험할 수 있습니다.

```text
주민등록번호: <합성 테스트 번호>
외국인등록번호: <합성 테스트 번호>
여권번호: <합성 테스트 번호>
운전면허번호: <합성 테스트 번호>
이메일: test@example.com
전화번호: 010-1234-5678
```

식별번호는 실제 값을 사용하지 말고 테스트 전용 합성 값으로 바꿔 입력합니다.

CUDA 환경에서는 다음 옵션을 사용할 수 있습니다.

```powershell
python .\tools\test_pii_text.py --provider cuda
```
