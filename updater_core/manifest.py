from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import requests

from .paths import CURRENT_VERSION_FILE, read_json


UPDATER_PROTOCOL_VERSION = 2
DEFAULT_UPDATE_INFO_URL = (
    "https://raw.githubusercontent.com/GreenRiceCake/GTMate/main/update_manifest.json"
)


@dataclass(frozen=True)
class UpdatePackage:
    package_type: str
    url: str
    sha256: Optional[str] = None
    size: Optional[int] = None
    package_manifest: str = "installed_manifest.json"
    transactional: bool = False


@dataclass(frozen=True)
class UpdateInfo:
    version: str
    title: str
    changelog: str
    package: UpdatePackage
    raw: dict[str, Any]


def parse_version(version_text: str) -> tuple[int, int, int]:
    parts: list[int] = []
    for part in str(version_text).strip().split("."):
        digits = "".join(character for character in part if character.isdigit())
        parts.append(int(digits or 0))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def is_newer_version(latest_version: str, current_version: str) -> bool:
    return parse_version(latest_version) > parse_version(current_version)


def load_current_version(base_dir: Path) -> str:
    data = read_json(base_dir / CURRENT_VERSION_FILE, default={})
    if not isinstance(data, dict):
        return "0.0.0"
    return str(data.get("version", "0.0.0"))


def _optional_positive_int(value: Any, field_name: str) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 값이 정수가 아닙니다.") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} 값은 0보다 커야 합니다.")
    return parsed


def parse_update_info(data: dict[str, Any]) -> UpdateInfo:
    if not isinstance(data, dict):
        raise ValueError("업데이트 매니페스트의 최상위 값은 객체여야 합니다.")

    version = str(data.get("version", "")).strip()
    if not version:
        raise ValueError("업데이트 매니페스트에 version이 없습니다.")

    modern = data.get("transactional_package")
    if modern is not None and not isinstance(modern, dict):
        raise ValueError("transactional_package는 객체여야 합니다.")

    if isinstance(modern, dict):
        minimum_protocol = int(data.get("min_updater_protocol", 2))
        if minimum_protocol > UPDATER_PROTOCOL_VERSION:
            raise ValueError(
                f"이 업데이트에는 Updater protocol {minimum_protocol} 이상이 필요합니다."
            )
        package_type = str(modern.get("type", "zip")).strip().lower()
        url = str(modern.get("url") or modern.get("download_url") or "").strip()
        sha256 = str(modern.get("sha256") or "").strip().lower() or None
        size = _optional_positive_int(modern.get("size"), "transactional_package.size")
        package_manifest = str(
            modern.get("package_manifest") or "installed_manifest.json"
        ).strip()
        transactional = True
    else:
        package_type = str(data.get("update_type", "exe")).strip().lower()
        url = str(data.get("download_url", "")).strip()
        sha256 = str(data.get("sha256") or "").strip().lower() or None
        size = _optional_positive_int(data.get("size"), "size")
        package_manifest = str(
            data.get("package_manifest") or "installed_manifest.json"
        ).strip()
        transactional = False

    if package_type not in {"zip", "exe", "installer"}:
        raise ValueError(f"지원하지 않는 업데이트 형식입니다: {package_type}")
    if not url:
        raise ValueError("업데이트 다운로드 URL이 없습니다.")
    if sha256 and (len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256)):
        raise ValueError("SHA-256 값의 형식이 올바르지 않습니다.")
    if transactional and not sha256:
        raise ValueError("transactional_package에는 ZIP SHA-256이 반드시 필요합니다.")

    package = UpdatePackage(
        package_type=package_type,
        url=url,
        sha256=sha256,
        size=size,
        package_manifest=package_manifest,
        transactional=transactional,
    )
    return UpdateInfo(
        version=version,
        title=str(data.get("title") or f"GTMate v{version} 업데이트"),
        changelog=str(data.get("changelog") or ""),
        package=package,
        raw=data,
    )


def fetch_update_info(
    url: str = DEFAULT_UPDATE_INFO_URL,
    timeout: int = 15,
    request_get: Callable[..., Any] = requests.get,
) -> UpdateInfo:
    response = request_get(url, timeout=timeout)
    response.raise_for_status()
    return parse_update_info(response.json())
