# NOTES

Dated log of data surprises, decisions, and anything I was not sure about.

## 2026-10-02 — Phase 0: scaffold and network check

Environment
- The cloud sandbox ships Python 3.14, where `numcodecs` (needed by `zarr<3`) fails to build. I used a Python 3.11 virtualenv (`uv venv -p 3.11 .venv`). Installed: zarr 2.18.7, xarray 2026.9.0, pandas 3.0.6, numpy 2.4.6. `.venv/` is in `.gitignore`.

Network check from the cloud sandbox (what worked and what did not)
- DNS works: `mastapp.site` -> 130.246.80.113, `s3.echo.stfc.ac.uk` -> four 130.246.x.x addresses.
- `https://example.com` returns 200, so general internet access is fine.
- `pd.read_parquet("https://mastapp.site/parquet/level2/shots")` FAILED: `SSL: UNEXPECTED_EOF_WHILE_READING`. `curl` to the same URL timed out after 60 s (HTTP code 000). Plain `http://mastapp.site/...` only returns a 301 redirect to https.
- `xr.open_zarr(".../11860.zarr", group="summary")` FAILED: `KeyError '.zmetadata'`. `curl` to `.../11860.zarr/summary/.zgroup` fails with "Connection reset by peer" during the TLS handshake. Plain http is reset too.
- Retried three times over several minutes with the same result.
- Conclusion: both data servers are NOT reachable from this sandbox (the TLS connection is dropped for these hosts only). The shot table was not loaded and shot 11860's `summary` group was not opened here. Nothing in SPEC Section 3 has been re-verified from this side.
- Everything has to be run from the Colab notebook (the project rules, Environment notes).

Pending decisions (carried from the SPEC)
- Whether to drop `shot_abort` / `shot_useful == False` shots: not decided, needs real data (SPEC 3.2).
