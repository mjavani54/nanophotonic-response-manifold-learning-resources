#!/usr/bin/env python3
"""Restore ZIP-wrapped binary parts to the full JMO data asset archive."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
import json
import sys
def digest(p):
    h=sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()
def main():
    if len(sys.argv)!=3: raise SystemExit("Usage: python scripts/restore_data.py PARTS_DIRECTORY OUTPUT_DIRECTORY")
    parts_dir=Path(sys.argv[1]);out=Path(sys.argv[2])
    manifest=json.loads((Path(__file__).resolve().parents[1]/"docs/DATA_PARTS.json").read_text())
    if not out.exists():out.mkdir(parents=True)
    target=out/manifest["data_bundle_filename"]
    if target.exists():raise SystemExit(f"Refusing to overwrite {target}")
    try:
        with target.open("wb") as w:
            for i,entry in enumerate(manifest["parts"],1):
                p=parts_dir/entry["name"]
                if not p.is_file() or p.stat().st_size!=entry["bytes"] or digest(p)!=entry["sha256"]:
                    raise ValueError(f"Missing or damaged part {i}: {p}")
                with ZipFile(p) as z:
                    meta=json.loads(z.read("part.json"))
                    if meta["index"]!=i or meta["total"]!=len(manifest["parts"]):
                        raise ValueError(f"Wrong part index in {p}")
                    h=sha256();n=0
                    with z.open("payload.bin") as src:
                        for b in iter(lambda:src.read(8<<20),b""):
                            w.write(b);h.update(b);n+=len(b)
                    if n!=entry["payload_bytes"] or h.hexdigest()!=entry["payload_sha256"]:
                        raise ValueError(f"Damaged payload in {p}")
        if target.stat().st_size!=manifest["data_bundle_bytes"] or digest(target)!=manifest["data_bundle_sha256"]:
            raise ValueError("Restored data bundle checksum mismatch")
        with ZipFile(target) as z:
            for info in z.infolist():
                rel=Path(info.filename)
                if rel.is_absolute() or ".." in rel.parts or rel.parts[0]!="JMO_Data_Assets":
                    raise ValueError("Unsafe archive path")
            z.extractall(out)
        print(f"Restored {target} and extracted {out/'JMO_Data_Assets'}")
    except Exception:
        target.unlink(missing_ok=True)
        raise
if __name__=="__main__":main()
