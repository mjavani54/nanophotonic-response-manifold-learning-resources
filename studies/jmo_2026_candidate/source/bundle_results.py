#!/usr/bin/env python3
"""Make a small review ZIP of completed results, excluding large input data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ALLOWED = {".csv", ".json", ".txt", ".log", ".png", ".pdf"}
EXCLUDE_DIRS = {"extracted", "Datasets", "Datasets20", ".venv", "__pycache__"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if not args.runs_root.is_dir():
        p.error("Runs root does not exist")
    if args.out.exists():
        p.error("Refusing to overwrite an existing review ZIP")
    files = [f for f in sorted(args.runs_root.rglob("*")) if f.is_file()
             and f.suffix.lower() in ALLOWED
             and not set(f.relative_to(args.runs_root).parts[:-1]).intersection(EXCLUDE_DIRS)]
    if not files:
        p.error("No CSV/JSON/log/figure results found")
    manifest = {str(f.relative_to(args.runs_root)): {"bytes": f.stat().st_size,
                "sha256": digest(f)} for f in files}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.out, "w", compression=ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            z.write(f, f.relative_to(args.runs_root))
        z.writestr("REVIEW_FILE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.out} with {len(files)} result files; source data ZIPs excluded")


if __name__ == "__main__":
    main()
