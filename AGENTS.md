# AGENTS.md

## Project overview

Python data-science project applying CRISP-DM methodology to the Kaggle 50 Startups dataset. Simulates a 5-expert, 5-round discussion to identify factors driving Profit.

- **Entrypoint**: `main.py` — runs the full pipeline (EDA → ML → discussion → report)
- **Standalone**: `generate_workflow.py` — generates `output/workflow.png` (requires Chinese font)
- **Config**: `config.yaml` exists but `main.py` does **not** read it; paths are hardcoded
- **No tests, no lint, no typecheck, no CI, no .git**

## Commands

```bash
pip install -r requirements.txt   # install deps
python main.py                    # run full pipeline
python generate_workflow.py       # generate workflow diagram only
```

## Import quirk

`main.py` does `sys.path.insert(0, ...)` to add `src/` to the path, then imports via `from src.eda import ...`. Any new top-level script must follow this pattern or use `from eda import ...` _after_ adjusting `sys.path`.

## Key quirks

- `matplotlib.use("Agg")` is forced in `src/eda.py` and `generate_workflow.py` — no GUI backend needed/available.
- `generate_workflow.py` hardcodes font path `C:\Windows\Fonts\msyh.ttc` (Microsoft YaHei). It will fail on systems without this font.
- All output paths are relative to the project root (`output/`, `output/figures/`, `output/models/`). Directories are created automatically by `main.py`.
- Dataset is tiny: 50 rows, 5 columns. Everything runs in seconds.
- Discussion module (`src/discussion.py`) uses hardcoded Chinese text for expert dialog — not configurable.
