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

## 2026-10-02 — Phase 1: data layer (inspection NOT done)

Status
- `src/data.py`, `scripts/download.py`, `scripts/inspect_shots.py` and `tests/test_data.py` are written. The 5 data tests use fake in-memory shots and pass (cache hit on rerun, NaN columns for missing variables, retries, failures logged, empty group raises).
- Real run from the sandbox: `python scripts/download.py --ids 11860` fails after 3 retries with `KeyError: '.zmetadata'` and the failure is written to `data/skipped.csv`. This is the same network block as Phase 0. The failure path works on a real error; the success path against real servers is untested.
- NOT DONE: inspecting 20 random shots across campaigns, and downloading 50 shots. Per the project rules I am not guessing at the answers. The table below is deliberately empty.
- To finish, run in Colab: `python scripts/inspect_shots.py --n 20` then `python scripts/download.py --n 50`, run the download a second time to confirm everything is served from the cache, and paste the output here.

Inspection summary (to fill in from the Colab run)
| item | finding |
|---|---|
| time step per campaign | TBD - not yet run |
| time range | TBD - not yet run |
| sign of ip | TBD - not yet run |
| variables present / missing | TBD - not yet run |
| anything odd | TBD - not yet run |

Decisions
- A shot whose summary group has none of the four signals (or no `time`) is treated as a load failure, not as a shot of NaN columns. This catches the zarr 3 "empty group" problem loudly instead of silently producing empty data.
- A variable whose length differs from `time` is treated as missing (NaN column) rather than guessing how to align it. Revisit if the inspection shows this happens.
- Retries wait 2 s, 4 s, 8 s (`RETRY_BACKOFF_S` in config). Failures are appended to `data/skipped.csv` from the main thread only, so parallel workers cannot corrupt it.
