from pathlib import Path

from imas.config import ProjectPaths, get_project_root, resolve_project_path


def test_project_paths_are_paths():
    paths = ProjectPaths.discover()
    assert isinstance(paths.project_root, Path)
    assert isinstance(paths.runs_root, Path)


def test_resolve_relative_path():
    resolved = resolve_project_path("runs/example")
    assert resolved == (get_project_root() / "runs/example").resolve()


def test_environment_override(monkeypatch, tmp_path):
    monkeypatch.setenv("IMAS_PROJECT_ROOT", str(tmp_path))
    assert get_project_root() == tmp_path.resolve()
