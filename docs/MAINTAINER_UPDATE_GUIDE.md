# Updating this repository

The archived v1.1.0 tag and its release assets are a fixed historical record. Keep new response arrays and analysis outputs in a clearly versioned area of the current study, not under the v1.1.0 tag. The root `code/`, `results/`, and `figures/` remain visibly scoped to the archived study; new numerical results belong under `studies/jmo_2026_candidate/` until a separately checked public release exists.

After adding files, regenerate `CHECKSUMS.sha256` for all tracked files except itself, and check it from a clean checkout. Run `python scripts/verify_repository.py` and the validation commands for newly deposited input archives. A summary check alone does not verify full neural-network regeneration or raw-array identity.

Before citing a new complete deposit, confirm that the response archives and source designs listed in [the status inventory](CURRENT_STUDY_STATUS.md) are actually present, download correctly, match their recorded hashes, and load with the source scripts. Then record the new tag, asset names, hashes, and exact accessible URL in the data-availability statement. Keep the older v1.1.0 data separately labeled.

The complete repository ZIP distributed for this update contains an installer with a pinned base commit. It updates local checkouts only. Review the diff and push through normal Git workflow; a ZIP cannot alter an existing GitHub release or tag.
