import argparse
from pathlib import Path

__all__ = [
    "cleanLogFile",
    "cleanLogText",
]


def isProgress(line: str) -> bool:
    options = (
        "training graph autoencoder:",
        "training:",
    )
    return line.startswith(options)


def isArtifact(line: str) -> bool:
    options = (
        "model saved to ",
        "configuration saved to ",
        "training log saved to ",
        "dataframe saved to ",
    )
    return line.startswith(options)


def cleanLogText(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    seen_artifacts = set()
    pending = None

    def flushPending() -> None:
        nonlocal pending
        if pending is None:
            return
        cleaned.append(pending)
        pending = None

    for raw in lines:
        line = raw.rstrip()

        if not line.strip():
            flushPending()
            if cleaned and (cleaned[-1] != "") and (not isProgress(cleaned[-1])):
                cleaned.append("")
            continue

        if isProgress(line):
            pending = line
            continue

        flushPending()

        if isArtifact(line):
            if line in seen_artifacts:
                continue
            seen_artifacts.add(line)

        cleaned.append(line)

    flushPending()

    while cleaned and cleaned[-1] == "":
        cleaned.pop()

    return "\n".join(cleaned) + "\n"


def cleanLogFile(src: Path | str, dst: Path | str) -> tuple[int, int]:
    src, dst = Path(src), Path(dst)
    text = src.read_text(encoding="utf-8", errors="replace")
    cleaned = cleanLogText(text)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(cleaned, encoding="utf-8")
    return len(text.splitlines()), len(cleaned.splitlines())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("-o", "--outdir", type=Path, default=Path.cwd())
    args = parser.parse_args()

    for src in args.files:
        assert src.exists(), "!!!"
        dst = args.outdir / src.name
        before, after = cleanLogFile(src, dst)
        print(f"{src.name}: {before} -> {after} lines > {dst}")


if __name__ == "__main__":
    main()
