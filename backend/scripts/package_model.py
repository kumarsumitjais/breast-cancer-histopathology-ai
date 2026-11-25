from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "app" / "model" / "densenet121_063A.keras"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Package the Experiment 06.3A Keras checkpoint.")
    parser.add_argument("--source", required=True, help="Path to the trained .keras checkpoint")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Source checkpoint does not exist: {source}")
    if source.suffix.lower() != ".keras":
        raise SystemExit("Expected a .keras model file.")

    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, DEST)

    source_hash = sha256(source)
    dest_hash = sha256(DEST)
    if source_hash != dest_hash:
        DEST.unlink(missing_ok=True)
        raise SystemExit("SHA-256 verification failed after copy.")

    checksum_path = DEST.with_suffix(DEST.suffix + ".sha256")
    checksum_path.write_text(f"{dest_hash}  {DEST.name}\n", encoding="utf-8")

    print(f"Packaged model: {DEST}")
    print(f"Size: {DEST.stat().st_size / (1024**2):.2f} MiB")
    print(f"SHA-256: {dest_hash}")
    print(f"Checksum file: {checksum_path}")


if __name__ == "__main__":
    main()
