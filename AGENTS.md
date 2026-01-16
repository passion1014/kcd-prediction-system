# AGENTS.md
Centralized instructions for agentic contributors working inside `/Users/passion1014/project/axlrator/ml`.
Follow every rule below when reading, editing, testing, or shipping code.

## Reference Overview
1. Primary domains: AutoGluon trainer (src/trainer), FastAPI service (src/api), custom NER/KCD stack (src/ner, src/kcd).
2. Tests live under `test/` plus several executable notebooks in the same folder.
3. Documentation and historical decisions live in `docs/code/history`; skim before making architectural shifts.
4. Cursor/Copilot rule files are absent in this repo; treat this guide as the single source of truth.
5. 모든 작업 대화와 코드 주석은 한글로 작성한다.

## 1. Environment Setup
- Python 3.10+ is assumed; prefer venvs inside `./venv` and keep them out of git.
- Install shared dependencies with `pip install -r requirements.api.txt` for API work or `pip install -r requirements.trainer.txt` for trainer-only tasks.
- Torch/Transformers are not pinned here; install matching CUDA wheels when training on GPU.
- Export MLflow-related env vars (`MLFLOW_TRACKING_URI`, `MLFLOW_MODEL_NAME`, `MODEL_STAGE`) before running trainer or API services.
- Keep large artifact directories (e.g., `ner_output`, `kcd_output`, `mlruns`) out of commits; re-create locally via the commands below.

## 2. Build & Runtime Commands
Use absolute module paths (`python -m ...`) so imports stay stable.

### 2.1 Core scripts
- FastAPI dev server: `uvicorn src.api.main:app --reload --port 58080` (ensure MLflow registry is reachable).
- AutoGluon trainer dry run: `python -m src.trainer.train` (registers a model and promotes it to Staging).
- NER sample training: `python -m src.ner.train --sample --epochs 2 --output_dir ./ner_output`.
- KCD sample training: `python -m src.kcd.train --sample --epochs 2 --output_dir ./kcd_output`.
- NER inference CLI: `python -m src.ner.inference --model_path ./ner_output --text "환자가 좌측 무릎에 통증"`.
- Pipeline quick check: reuse the snippet from README §4 "파이프라인 사용" after generating both checkpoints.

### 2.2 Docker services
- Compose MLflow + API: `docker compose up -d --build`.
- Trainer image run (writes to mlruns): `docker run --rm --network ml_default -v $(pwd)/mlruns:/mlflow/mlruns -e MLFLOW_TRACKING_URI=http://mlflow:5000 trainer:local`.
- Restart API after promoting a model: `docker compose restart api`.
- Tear down services: `docker compose down` (add `-v` only if you intend to drop volumes).

## 3. Lint, Format, and Static Analysis
No enforced tooling is wired into CI, but keep the following conventions:
1. Run `ruff check src test` (install with `pip install ruff`) before submitting Python patches; silence false positives via `# noqa: <rule>` only when documented.
2. Format touched Python files with `black src test --line-length 120`; match existing indentation (4 spaces) when manual tweaks suffice.
3. Type-check critical modules (API schemas, dataclasses, inference utilities) with `pyright src/api src/common src/kcd src/ner` when you add or refactor type-heavy code; keep Optional typing explicit.
4. Keep imports sorted: builtins, third-party, then absolute project imports (`from src....`). Avoid relative imports except within package `__init__` files.
5. Never mix tabs/spaces; stick to UTF-8 files with Unix newlines.

## 4. Test Strategy
### 4.1 Pytest
- Install extra deps: `pip install -r requirements.api.txt requests pytest` (pytest not pinned but required).
- Run entire suite: `pytest`.
- Fast API smoke-only: `pytest test/test_api_predict.py`.
- Single test focus (requested most often): `pytest test/test_api_predict.py::test_api_predict_smoke -vv`.
### 4.2 Module CLIs
- Sanity-check NER tagging: `python -m src.ner.tags`.
- Validate NER/KCD data formats: `python -m src.ner.data_format` and `python -m src.kcd.data_format`.
- Exercise KCD dictionary helpers: `python -m src.kcd.kcd_dictionary`.
### 4.3 End-to-end manual
1. Train or reuse NER/KCD checkpoints in `./ner_output` and `./kcd_output`.
2. Run the pipeline snippet from README section "4. 파이프라인 사용".
3. Start FastAPI (`uvicorn ...`) and curl `/health` plus `/predict` using the payload in `test/test_api_predict.py`.
4. Record MLflow run IDs when touching trainer logic; artifacts go to `mlruns/` and must not be committed.

## 5. Python Style Guide
1. **Typing**: keep function signatures annotated; favor `list[str]`/`dict[str, Any]` (Python 3.10) over `typing.List`.
2. **Dataclasses**: prefer `@dataclass` for configs and DTOs (see `src/ner/model.py`, `src/kcd/data_format.py`). Use `field(default_factory=list)` for mutables.
3. **Immutability**: treat configs as read-only; if mutation is needed, copy via `.copy()` or dataclass `replace`.
4. **Imports**: project modules must be imported absolutely (`from src.kcd...`). Avoid wildcard imports entirely.
5. **Constants**: define module-level defaults in ALL_CAPS (e.g., `DEFAULT_MODEL_NAME`), keep them near the top.
6. **Formatting**: wrap docstrings at ~100 chars, but allow longer code lines where HuggingFace call signatures demand it; maintain f-string readability over concatenation.
7. **Logging/prints**: CLI scripts may use `print` for status (matching existing training scripts). Inside services, prefer `logging` or FastAPI exceptions.
8. **Error handling**: raise explicit `ValueError`/`RuntimeError` when configuration is missing; convert user-facing issues into `HTTPException` with precise status codes.
9. **Device management**: always move tensors to `self.device` before computation; keep `.to(self.device)` calls close to data loader loops like in `NERModel.train`.
10. **Serialization**: when saving configs/labels, use UTF-8 with `ensure_ascii=False` to preserve Korean labels; mirror the JSON structure used in existing `save` methods.
11. **Randomness**: expose seeds through CLI arguments when writing new trainers.
12. **Docstrings**: start with a short imperative sentence, then add Args/Returns blocks as shown in `src/kcd/model.py`.

## 6. Naming & Module Boundaries
- Public classes follow PascalCase (`NERModel`, `KCDPredictionPipeline`).
- Private helpers should be prefixed with `_` and reside near their call sites.
- Keep dataset builders in their module (e.g., `create_sample_dataset` inside `data_format.py`).
- For FastAPI schemas, suffix request bodies with `Request` (`PredictRequest`).
- CLI entrypoints expose `main()` and guard with `if __name__ == "__main__":` like current scripts.

## 7. Error Handling Patterns
1. Validations should happen early (e.g., check for models in registry before inference).
2. Wrap external service calls (MLflow, HTTP) in try/except and produce actionable messages.
3. In training loops, fail fast on missing data files instead of swallowing exceptions.
4. When raising `HTTPException`, avoid leaking stack traces; convert caught exceptions to user-readable text like `Prediction failed: <reason>`.
5. Never log PHI or patient-identifying text; redact or summarize when printing.

## 8. Testing/Data Guidelines
- Dummy data lives in `data/ner` and `data/kcd`; extend them in-place for richer smoke tests but keep sample sizes tiny for version control.
- Large models should be regenerated locally; never upload checkpoints >10MB.
- When adding tests, prefer deterministic samples and avoid random shuffling without seeds.
- Keep notebooks out of automation; if you must reference them, export distilled scripts into `test/` or `docs/`.
- Document any manual verification steps in `docs/code/history` for future agents.

## 9. API Contracts
1. `/predict` accepts a JSON body with `rows: list[dict]`; keep parity with AutoGluon training columns.
2. Validate request payloads via Pydantic schemas; extend `PredictRequest` when new metadata columns appear.
3. Responses should stay simple: top-level `predictions` array containing JSON-serializable primitives.
4. Health endpoint must return `status`, `model_name`, `stage`, `loaded_version`; update tests if you add fields.

## 10. Workflow Expectations
- Keep changes minimal and scoped; do not refactor cross-module code without updating docs/tests.
- Before editing, skim git history (`git blame`) to understand author intent.
- When adding features, update README sections ("빠른 시작", "파이프라인 사용") plus this AGENTS file if rules change.
- Prefer feature flags or config toggles over hard-coded behavior, especially for MLflow URIs.
- Coordinate trainer/API schema changes so both components continue to interoperate.

## 11. Contribution Checklist
1. Create/activate virtualenv.
2. Install deps via appropriate requirements file.
3. Run targeted linters (`ruff`, `black`).
4. Execute relevant tests or CLIs (include single-test command if touching FastAPI or pipeline code).
5. Document manual validation in a short note (README snippet or docs/history entry).
6. Verify git status is clean except for intentional changes; never commit artifacts.
7. Communicate next steps or TODOs in pull request descriptions.

## 12. Module-Specific Notes
### 12.1 src/ner
- Maintain BIO tagging scheme defined in `tags.py`; update label maps consistently when adding entities.
- Keep `NERModelConfig` backwards-compatible; new fields require default values so older checkpoints load.
- Dataset preprocessing must continue to set ignored tokens to -100 for loss masking.
- Always group extracted entities by label name when returning from `extract_features`.
### 12.2 src/kcd
- Label mappings live in `labels.json`; regenerate them with helper utilities when class sets expand.
- Keep `NERFeatures.to_text()` human-readable (prefixed with Korean field tags) so the classifier sees consistent prompts.
- Any new metadata fields must flow through dataclasses, pipeline, and tokenizer input assembly.
### 12.3 src/api
- Avoid global state beyond `ModelHolder`; if you add caches, encapsulate them in classes.
- Network calls to MLflow should surface actionable RuntimeErrors when registry lacks models.
- Return JSON-safe types (coerce pandas/numpy scalars to Python builtins as shown).
### 12.4 src/trainer
- Keep MLflow logging minimal but complete: metrics, params, artifacts, and auto-registration.
- When changing AutoGluon presets, expose toggles via CLI args or env vars, not hard-coded strings.
- Clean up temp directories even on failure (use context managers as in the current script).
### 12.5 test/
- Tests may hit live services; gate external calls behind env vars so CI can skip them if needed.
- Notebook-based experiments should be mirrored by lightweight `.py` smoke tests where practical.

## 13. Documentation & Communication
- Capture non-obvious architectural decisions in `docs/code/history` with date-stamped markdown files.
- Keep README up to date when new CLI flags or endpoints ship.
- Note any data-sensitive assumptions (e.g., PHI handling) in doc comments or README warnings.
- When deviating from this guide, append a short addendum here so future agents inherit the new rule.

## 14. Safety & Secrets
- Do not commit `.env`, MLflow credentials, or patient data; add new patterns to `.gitignore` if necessary.
- Inspect diffs for accidental large files (>5MB) before pushing.
- Docker commands may write to bound volumes; verify paths before running containers.

Staying within these guardrails keeps the AutoGluon + FastAPI + MLflow stack reproducible and safe for every future agent.

— End of AGENTS instructions —
