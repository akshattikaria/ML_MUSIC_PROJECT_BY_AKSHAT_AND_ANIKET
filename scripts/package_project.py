"""Package deliverables without the environment, caches, or original large archive."""

from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = {
    "README.md", "LICENSE", "CITATION.bib", "pyproject.toml",
    "requirements.txt", "requirements-tested.txt", ".gitignore",
}
FOLDERS = {"src", "tests", "docs", "scripts", "notebooks", "reports", "models", "baseline"}


def eligible(path: Path, complete: bool) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"} for part in relative.parts):
        return False
    if path.suffix in {".pyc", ".zip"} or any(part.endswith(".egg-info") for part in relative.parts):
        return False
    if len(relative.parts) == 1:
        return relative.name in TOP_LEVEL
    if relative.parts[0] == "data":
        return complete and relative.parts[1] == "processed"
    if relative.parts[0] == "models":
        return complete
    if relative.parts[0] == "baseline":
        return complete or relative.parts[1] != "models"
    return relative.parts[0] in FOLDERS


def main():
    for complete, name in [
        (True, "MusicMap_Improved_Complete.zip"),
        (False, "MusicMap_Improved_Code_and_Results.zip"),
    ]:
        target = ROOT.parent / name
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for path in sorted(ROOT.rglob("*")):
                if path.is_file() and eligible(path, complete):
                    bundle.write(path, Path("musicmap") / path.relative_to(ROOT))
        with zipfile.ZipFile(target) as bundle:
            corrupt = bundle.testzip()
            if corrupt is not None:
                raise RuntimeError(f"Invalid ZIP member: {corrupt}")
        print(f"{target}: {target.stat().st_size / 1024**2:.1f} MiB")


if __name__ == "__main__":
    main()
