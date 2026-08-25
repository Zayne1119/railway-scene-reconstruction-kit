# Contributing

1. Create a short-lived branch from the current main branch.
2. Put site-specific values in a project config, never in library code.
3. Add or update tests for every reusable behavior.
4. Run `pytest`, `ruff check .` and `railway-recon safety-check --root .`.
5. Do not commit generated runs or production data.
6. Describe evidence assumptions, confidence changes and compatibility impact in the pull request.

Core module filenames must not contain iteration suffixes such as `_v37` or `_v90`; use Git tags and the run manifest for versioning.

