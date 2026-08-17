"""프로젝트의 표준 데이터·산출물 경로와 구버전 경로 호환 처리."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"
IMAGE_DATA_DIR = DATA_DIR / "images"
OCR_RESULTS_DIR = DATA_DIR / "ocr_results"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = PROJECT_ROOT / "models"

_LEGACY_IMAGE_ROOTS = {
    "test_docs": SAMPLE_DATA_DIR / "test_docs",
    "sample_images": SAMPLE_DATA_DIR / "samples",
    "_llm_testset": SAMPLE_DATA_DIR / "llm_testset",
}


def resolve_image_path(
    image_path: str | Path,
    *,
    image_root: str | Path | None = None,
) -> Path:
    """OCR JSON의 이미지 경로를 현재 프로젝트 구조의 실제 경로로 바꾼다.

    기존 OCR JSON에는 ``test_docs/...`` 같은 이전 상대 경로가 저장되어 있다.
    결과 JSON을 다시 생성하지 않아도 사용할 수 있도록 첫 경로 조각을
    전달용 폴더의 ``sample_data`` 위치로 대응시킨다.
    """
    path = Path(str(image_path).replace("\\", "/"))
    configured_root = Path(image_root).expanduser() if image_root else None
    candidates: list[Path] = []

    if path.is_absolute():
        candidates.append(path)
        if configured_root is not None:
            candidates.append(configured_root / path.name)
    else:
        if configured_root is not None:
            candidates.append(configured_root / path)
            if len(path.parts) > 1:
                candidates.append(
                    configured_root.joinpath(*path.parts[1:])
                )
        candidates.append(PROJECT_ROOT / path)

    if not path.is_absolute() and path.parts:
        mapped_root = _LEGACY_IMAGE_ROOTS.get(path.parts[0].casefold())
        if mapped_root is not None:
            candidates.append(mapped_root.joinpath(*path.parts[1:]))

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved

    # 실제 파일 검증과 상세 오류 표시는 전처리 단계에서 담당한다.
    return candidates[0].resolve()
