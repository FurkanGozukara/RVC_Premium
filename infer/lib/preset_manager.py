import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _sanitize_name(name: str) -> str:
    safe = "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(name)
    )
    return safe.strip("._") or "default"


class PresetManager:
    """Small file-based preset manager aligned with the SECourses app pattern."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _tab_dir(self, tab: str) -> Path:
        folder = self.base_dir / _sanitize_name(tab)
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _preset_path(self, tab: str, preset_name: str) -> Path:
        return self._tab_dir(tab) / f"{_sanitize_name(preset_name)}.json"

    def _last_used_path(self, tab: str, model: Optional[str]) -> Path:
        last_used_dir = self._tab_dir(tab) / ".last_used"
        last_used_dir.mkdir(parents=True, exist_ok=True)
        model_name = _sanitize_name(model) if model else "default"
        return last_used_dir / f"{model_name}.txt"

    def list_presets(self, tab: str, model: Optional[str] = None) -> List[str]:
        _ = model
        folder = self._tab_dir(tab)
        return sorted(path.stem for path in folder.glob("*.json"))

    def save_preset(self, tab: str, model: Optional[str], preset_name: str, data: Dict[str, Any]) -> str:
        preset_path = self._preset_path(tab, preset_name)
        tmp_path = preset_path.with_suffix(".json.tmp")
        payload = self.validate_and_clean_preset(data)

        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        tmp_path.replace(preset_path)

        self.set_last_used(tab, model, preset_name)
        return preset_path.stem

    def load_preset(self, tab: str, model: Optional[str], preset_name: str) -> Optional[Dict[str, Any]]:
        _ = model
        preset_path = self._preset_path(tab, preset_name)
        if not preset_path.exists():
            return None
        try:
            with preset_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    def delete_preset(self, tab: str, model: Optional[str], preset_name: str) -> bool:
        _ = model
        preset_path = self._preset_path(tab, preset_name)
        if not preset_path.exists():
            return False
        preset_path.unlink()
        return True

    def set_last_used(self, tab: str, model: Optional[str], preset_name: str) -> None:
        path = self._last_used_path(tab, model)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(_sanitize_name(preset_name))

    def get_last_used_name(self, tab: str, model: Optional[str]) -> Optional[str]:
        path = self._last_used_path(tab, model)
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8").strip() or None
        except Exception:
            return None

    def load_last_used(self, tab: str, model: Optional[str]) -> Optional[Dict[str, Any]]:
        name = self.get_last_used_name(tab, model)
        if not name:
            return None
        return self.load_preset(tab, model, name)

    @staticmethod
    def merge_config(current: Dict[str, Any], preset: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not preset:
            return current.copy()

        merged = current.copy()
        for key, value in preset.items():
            if key in current:
                try:
                    expected_type = type(current[key])
                    if expected_type in (int, float, str, bool):
                        merged[key] = expected_type(value)
                    else:
                        merged[key] = value
                except (TypeError, ValueError):
                    merged[key] = current[key]
            else:
                merged[key] = value
        return merged

    @staticmethod
    def validate_and_clean_preset(preset: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(preset, dict):
            return {}

        cleaned: Dict[str, Any] = {}
        for key, value in preset.items():
            try:
                json.dumps({key: value})
                cleaned[key] = value
            except (TypeError, ValueError):
                continue
        return cleaned

    def save_preset_safe(self, tab: str, model: Optional[str], preset_name: str, data: Dict[str, Any]) -> str:
        payload = self.validate_and_clean_preset(data)
        if not payload:
            raise ValueError("No valid preset data to save")
        return self.save_preset(tab, model, preset_name, payload)

    def load_preset_safe(self, tab: str, model: Optional[str], preset_name: str) -> Optional[Dict[str, Any]]:
        preset = self.load_preset(tab, model, preset_name)
        if preset is None:
            return None
        return self.validate_and_clean_preset(preset)
