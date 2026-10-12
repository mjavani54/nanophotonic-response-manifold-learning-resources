# Publishing the exact response archives

This repository is public. It does not yet have a tagged release or the exact
archived response CSVs used in the paper. Before creating a release:

1. Add the original Python data-generation source if it can be identified.
2. Run `python scripts/verify_repository.py` and `sha256sum -c CHECKSUMS.sha256`.
3. Check the two raw-data ZIPs and their SHA-256 hashes against
   `GITHUB_RELEASE_ASSETS.md`.
4. Create a GitHub release from a specific tagged commit and attach both ZIPs
   as release assets. Keep the large archives out of ordinary Git history.
5. Update the manuscript and `CITATION.cff` with the exact release URL or DOI
   and release date, then verify the public asset downloads.

The stored design matrices and Python reconstruction currently in the Git tree
allow convention checks. The planned release assets are needed to rerun the
reported analyses with the exact archived response values.
