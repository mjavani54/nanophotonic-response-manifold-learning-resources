# Uploading this package to GitHub using the website

1. Sign in to GitHub and create a **New repository**.
2. Suggested repository name: `nanophotonic-response-manifold-learning-resources`.
3. Choose **Public** only when the manuscript/data release is ready; otherwise start **Private**.
4. Do not initialize the repository with another README, license, or `.gitignore` because this package already contains them.
5. Extract the repository ZIP on your computer.
6. Open the new repository and choose **Add file → Upload files**.
7. Upload the repository contents while preserving the folder structure.
8. Commit with a message such as `Reproducibility release v1.1.0`.
9. Run or inspect `scripts/verify_repository.py` locally before creating the public release.
10. Create a tagged release, preferably `v1.1.0`.

## Exact raw CSVs without a separate data DOI

If you do not want a separate raw-data DOI, attach the two prepared raw-data ZIP files to the GitHub **Release**, not to normal Git history:

- `Paper1_50k_RawData.zip`
- `Paper1_20L_200k_RawData.zip`

Release assets are appropriate for these large files; the normal repository should remain code/document focused. After the release is public, use the release URL in the manuscript's Data and Code Availability statement.

The repository itself already contains the compressed exact design matrices and the historical response generator. The release assets provide the exact archived response CSVs for byte-level reproduction.
