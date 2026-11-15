# Open items for the candidate paper's complete public deposit

The archived v1.1.0 release is public and stays unchanged. The Journal of Modern Optics candidate study has its code, compact summaries, and recorded fit bundle under `studies/jmo_2026_candidate/`, but this repository update is not yet a complete reproduction deposit.

1. Supply the exact 50k and 20-layer 200k response archives identified in the material manifest. Verify both SHA-256 hashes and a clean input load.
2. Supply the 14/16/18-layer extra training matrices and the 20-layer A-ensemble source-design archive, plus the spectral response arrays. Verify the generation manifest hashes and the source files actually used for the fits.
3. Run the complete analysis and document any implementation choices not already captured by `source/`, particularly dependencies, compute backends, and a validation report for the numerical response model.
4. Create a separately versioned public release/deposit. Test downloads and checksums from a clean environment. Then replace the bracketed data links in the manuscript and Supporting Information.
5. Confirm the final disclosure of AI assistance against the work actually performed and remove author-review text from the submission documents.

Passing `python scripts/verify_repository.py` confirms integrity and summary consistency only; it does not close items 1–4.
