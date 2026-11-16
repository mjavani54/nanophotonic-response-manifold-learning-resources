#!/usr/bin/env python3
"""Verify tracked-file integrity, recorded fit archives, and optional raw data."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
import argparse
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
    h=sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()
def check_manifest(root,path):
    listing=json.loads(path.read_text())
    for name,meta in listing.items():
        p=root/name
        if not p.is_file() or p.stat().st_size!=meta["bytes"] or digest(p)!=meta["sha256"]:
            raise ValueError(f"Missing/mismatched file {p}")
    return len(listing)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,help="Extracted JMO_Data_Assets directory")
    a=p.parse_args()
    lines=(ROOT/"CHECKSUMS.sha256").read_text().splitlines()
    for line in lines:
        expected,name=line.split("  ",1)
        path=ROOT/name
        if not path.is_file() or digest(path)!=expected:
            raise ValueError(f"Repository checksum mismatch: {name}")
    with ZipFile(ROOT/"records/run_records_review.zip") as z:
        listing=json.loads(z.read("REVIEW_FILE_MANIFEST.json"))
        for name,meta in listing.items():
            data=z.read(name)
            if len(data)!=meta["bytes"] or sha256(data).hexdigest()!=meta["sha256"]:
                raise ValueError(f"Recorded fit file mismatch: {name}")
    cmd=[sys.executable,str(ROOT/"verify_summary.py")]
    if a.data:cmd.append(str(a.data))
    subprocess.run(cmd,check=True,cwd=ROOT)
    if not a.data:
        print(f"Checked {len(lines)} repository files and {len(listing)} recorded-fit files. Raw assets NOT checked.")
        return
    count=check_manifest(a.data,ROOT/"docs/ASSET_MANIFEST.json")
    recorded=json.loads((ROOT/"records/extension_data/generation_manifest.json").read_text())
    for layer in (14,16,18):
        f=a.data/f"extension_data/extra_L{layer}_160000.npz"
        if digest(f)!=recorded["cases"][str(layer)]["file_sha256"]:
            raise ValueError(f"Augmented original run hash mismatch: {f}")
    with ZipFile(a.data/"primary/Datasets50k_eps21316.zip") as z:
        for layer in (14,16,18):
            for kind in ("DS","RS"):
                name=f"Datasets/SiO2_TiO2_{layer}_layer_50K_200_points_{kind}.csv"
                h=sha256()
                with z.open(name) as f:
                    for b in iter(lambda:f.read(8<<20),b""):h.update(b)
                if h.hexdigest()!=recorded["cases"][str(layer)]["source_member_sha256"][kind]:
                    raise ValueError(f"Recorded primary member differs: {name}")
    identity=json.loads((a.data/"archive_identity.json").read_text())
    assert all(identity["spectral_pca_diagnostics_match_recorded_values"].values())
    print(f"Checked {len(lines)} repository files, {len(listing)} recorded fits, {count} raw assets, six exact CSV members, and three exact augmented NPZs.")
    print("Original primary/extended ZIP-container identity and per-seed retraining remain unverified.")
if __name__=="__main__":main()
