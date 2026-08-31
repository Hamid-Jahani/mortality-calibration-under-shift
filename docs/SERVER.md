# Running the sweeps on the CPU server

Why the server: on the laptop, 12 workers each importing torch exceeded the
Windows commit limit (`WinError 1455`) and the load was unacceptable in use;
the sweeps are embarrassingly parallel over populations and CPU-bound.

## Sizes to expect (solo per-cell timings, `results/timings_solo.json`)

| regime | triples | wall on N workers ≈ 2.17 h × triples / N |
|---|---|---|
| shift | 40 | 87 h / N  → 16 cores: 6 h, 32 cores: 3 h |
| placebo | 22 | 48 h / N  → 16: 3 h, 32: 1.5 h |
| stable | 401 | 870 h / N → 16: 54 h, 32: 27 h, 64: 14 h |

Plus a low-parallel GP pass per regime (~1.6 GB RAM per GP cell; solo
GP/native 45 s, conformal cells a few minutes each): `GP_JOBS ≈ RAM_GB / 3`.

## Steps

1. `git clone git@github.com:Hamid-Jahani/mortality-calibration-under-shift.git && cd mortality-calibration-under-shift`
2. Copy the HMD bulk files to `Dataset/` on the server. They are
   registration-restricted (not in the repo). From the laptop:
   `rsync -av --progress "/g/Mortality - Explainable AI/Dataset/" user@server:~/mortality-calibration-under-shift/Dataset/`
   (~900 MB; only `deaths/Deaths_1x1`, `exposures/Exposures_1x1`,
   `death_rates/Mx_1x1` and `lt_*/…per_1x1` are read by the sweeps and gates).
3. `bash scripts/server_setup.sh` — installs uv + a CPU-torch env, verifies the
   data vintage against `data/MANIFEST.sha256`, runs the suite, then launches
   `scripts/launch_sweeps.sh shift placebo` detached with `JOBS = cores − 2`.
   `SKIP_LAUNCH=1` prepares without launching; `REGIMES="stable"` for the
   multi-day run; `JOBS=…`/`GP_JOBS=…` override the defaults.
4. Follow: `tail -f results/logs/server_launch.out`. Parts land in
   `results/<regime>.parts/<POP>__<MODEL>.parquet`; a killed run resumes at
   that granularity by re-running the same command.
5. Bring results back: `rsync -av user@server:~/mortality-calibration-under-shift/results/*.parquet "/g/Mortality - Explainable AI/results/"`
   then `git add results/*.parquet` (parts and logs are gitignored).

## Rules that still apply on the server

- `MORTCAL_DEVICE=cpu` for the whole regime (one device per regime; the
  `device` column in every row records it).
- R/StMoMo is NOT needed on the server: gate 2 (oracle parity) is already
  closed and its artefacts are in `results/parity/`.
- Nothing in `PREREGISTRATION*.md` changes because the machine changed.
