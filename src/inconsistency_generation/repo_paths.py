from pathlib import Path


def get_repo_root(start_path: Path | str | None = None) -> Path:
    current_path = Path.cwd().resolve() if start_path is None else Path(start_path).resolve()

    for candidate in (current_path, *current_path.parents):
        if (candidate / "src").exists() and (candidate / "notebooks").exists():
            return candidate

    raise RuntimeError(
        f"Could not infer the repository root from {current_path}. "
        "Expected a parent directory containing both 'src' and 'notebooks'."
    )


def get_src_dir(start_path: Path | str | None = None) -> Path:
    return get_repo_root(start_path) / "src"