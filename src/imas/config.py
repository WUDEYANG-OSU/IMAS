"""Configuration and portable path helpers for IMAS.

The public repository never assumes a fixed drive letter or mount point.
Set ``IMAS_PROJECT_ROOT`` to the workspace containing ``LLM-WorldWord``,
``networks`` and ``runs``. Individual roots can be overridden with the
more specific environment variables documented in ``docs/configuration.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any, Mapping

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: str) -> str:
    def repl(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name, default or "")
    return os.path.expanduser(_ENV_PATTERN.sub(repl, value))


def _repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__).resolve()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "imas").exists():
            return candidate
    return Path.cwd().resolve()


def _yaml_load(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load YAML configuration files. "
            "Install the project with `pip install -e .`."
        ) from exc
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML configuration and expand ``${ENV:-default}`` strings."""
    config_path = Path(path or os.environ.get("IMAS_CONFIG", "configs/project.yaml"))
    if not config_path.is_absolute():
        config_path = _repo_root() / config_path
    if not config_path.exists():
        fallback = _repo_root() / "configs" / "project.default.yaml"
        if not fallback.exists():
            return {}
        config_path = fallback
    payload = _yaml_load(config_path)

    def expand(value: Any) -> Any:
        if isinstance(value, str):
            return _expand_env(value)
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        return value

    return expand(payload)


def _configured_path(key: str, env_name: str, default: Path) -> Path:
    if env_name in os.environ:
        return Path(_expand_env(os.environ[env_name])).resolve()
    try:
        cfg = load_config()
    except Exception:
        cfg = {}
    value = (cfg.get("paths", {}) or {}).get(key)
    if value:
        path = Path(str(value))
        if not path.is_absolute():
            path = get_project_root() / path
        return path.resolve()
    return default.resolve()


def get_project_root() -> Path:
    """Return the external IMAS workspace root.

    Priority: ``IMAS_PROJECT_ROOT`` > project YAML > repository root.
    """
    env = os.environ.get("IMAS_PROJECT_ROOT")
    if env:
        return Path(_expand_env(env)).resolve()
    # Avoid recursion through load_config by reading only the default file here.
    default_file = _repo_root() / "configs" / "project.yaml"
    if not default_file.exists():
        default_file = _repo_root() / "configs" / "project.default.yaml"
    if default_file.exists():
        try:
            raw = _yaml_load(default_file)
            value = (raw.get("paths", {}) or {}).get("project_root")
            if value:
                expanded = Path(_expand_env(str(value)))
                if not expanded.is_absolute():
                    expanded = _repo_root() / expanded
                return expanded.resolve()
        except Exception:
            pass
    return _repo_root()


def get_world_root() -> Path:
    return _configured_path("world_root", "IMAS_WORLD_ROOT", get_project_root() / "LLM-WorldWord")


def get_runs_root() -> Path:
    return _configured_path("runs_root", "IMAS_RUNS_ROOT", get_project_root() / "runs")


def get_global_root() -> Path:
    return _configured_path("global_root", "IMAS_GLOBAL_ROOT", get_project_root() / "networks" / "qc_global_projection")


def get_maps_root() -> Path:
    return _configured_path("maps_root", "IMAS_MAPS_ROOT", get_global_root() / "maps")


def get_priors_root() -> Path:
    return _configured_path("priors_root", "IMAS_PRIORS_ROOT", get_global_root() / "priors_cache")


def get_checkpoint_root() -> Path:
    return _configured_path("checkpoint_root", "IMAS_CHECKPOINT_ROOT", get_project_root() / "checkpoints")


def get_resources_root() -> Path:
    return _configured_path("resources_root", "IMAS_RESOURCES_ROOT", get_project_root() / "resources")


def get_verify_pc_root() -> Path:
    return _configured_path("verify_pc_root", "IMAS_VERIFY_PC_ROOT", get_world_root() / "Verify-PC")


def get_external_root(name: str, default_subdir: str | None = None) -> Path:
    env_name = f"IMAS_{name.upper()}_ROOT"
    default = get_project_root() / (default_subdir or f"external/{name.lower()}")
    return _configured_path(f"external_{name.lower()}_root", env_name, default)


def get_dataset_checkpoint(dataset: str, filename: str = "ckpt_last_BASE_global_hashpeak.pt") -> Path:
    """Return the configurable checkpoint location for one dataset."""
    env_name = "IMAS_CKPT_" + re.sub(r"[^A-Za-z0-9]+", "_", dataset).upper()
    if env_name in os.environ:
        path = Path(_expand_env(os.environ[env_name]))
        return path if path.suffix else path / filename
    return get_checkpoint_root() / dataset / filename


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a relative path against ``IMAS_PROJECT_ROOT``."""
    path = Path(_expand_env(str(value)))
    return path.resolve() if path.is_absolute() else (get_project_root() / path).resolve()


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    world_root: Path
    runs_root: Path
    global_root: Path
    maps_root: Path
    priors_root: Path
    checkpoint_root: Path
    resources_root: Path

    @classmethod
    def discover(cls) -> "ProjectPaths":
        return cls(
            project_root=get_project_root(),
            world_root=get_world_root(),
            runs_root=get_runs_root(),
            global_root=get_global_root(),
            maps_root=get_maps_root(),
            priors_root=get_priors_root(),
            checkpoint_root=get_checkpoint_root(),
            resources_root=get_resources_root(),
        )
