# Published exact response archives

The repository and release v1.1.0 are public. The release contains the exact
archived design and response CSVs used in the paper. The completed checks were:

1. `python scripts/verify_repository.py` passes.
2. `sha256sum -c CHECKSUMS.sha256` passes for the Git-tracked files.
3. Both release assets match the SHA-256 values in
   `GITHUB_RELEASE_ASSETS.md`.
4. Tag `v1.1.0` resolves to the released repository commit.

The stored design matrices and Python reconstruction in the Git tree allow
convention checks. The release assets provide the exact archived inputs for
rerunning the reported analyses.
