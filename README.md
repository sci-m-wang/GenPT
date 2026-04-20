# GenPT: Generative Projective Testing for LLM Psychological Assessment

GenPT evaluates persona-conditioned LLM agents with a three-stage projective-testing pipeline (Examinee / Interpreter / Diagnostician) grounded in standardised clinical scoring systems (SCORS-G, SRAS, SCT scoring). The framework is designed as a complementary alternative to self-report questionnaires when contamination resistance, bias asymmetry, and longitudinal-context sensitivity are the primary desiderata.

This repository contains the core pipeline code, the projective stimuli (Rorschach inkblots, TAT images, SCT sentence stems), the CharacterRAG persona set, and the scripts needed to reproduce the psychometric experiments (validity, social-desirability resistance, longitudinal-context responsiveness) on a CharacterRAG-only subset.

## Repository layout

```
.
├── genpt/                      Core pipeline package
│   ├── stimuli/                TAT / Rorschach / SCT loaders
│   ├── pipeline/               Examinee / Interpreter / Diagnostician
│   ├── scoring/                SCORS-G, SRAS, SCT quantitative scoring
│   ├── evaluation/             Metrics (accuracy, MAE, Hamming distance, …)
│   ├── experiments/            Reliability + validity evaluation entry points
│   ├── llm/                    vLLM / OpenAI-compatible clients
│   └── config.py               Model, stimuli, and experiment settings
├── gen_stimulis/               Generated projective stimuli (~45 MB)
│   ├── gen_Rorschach_figures/  10 Rorschach cards × diffusion variants
│   ├── gen_TAT_figures/        TAT images by scenario category
│   └── sct_final_filtered.json Sentence Completion Test stems
├── characters/CharacterRAG/    15 characters + PDB ground-truth labels
├── questionnaires/             Self-report instruments (BFI, 16Personalities, BDI, BSS, …)
└── scripts/                    Runnable CLIs (generation / evaluation / analysis)
```

Runtime artefacts (`data/`, `reports/`, `results/`, `outputs/`, `logs/`, `cache/`) are gitignored; each script writes its own outputs under these paths.

## Installation

```bash
pip install -e .
# or, with uv:
uv sync
```

Core dependencies: `torch`, `transformers`, `openai`, `qwen-vl-utils`, `numpy`, `scipy`, `scikit-learn`, `Pillow`.

## Running an assessment

The end-to-end pipeline on a single CharacterRAG persona:

```bash
python -m genpt assess \
    --source characterrag \
    --character-path characters/CharacterRAG/anya_forger \
    --use-api --api-base http://127.0.0.1:8000/v1 \
    --output-dir ./results
```

Inspect what stimuli are loaded:

```bash
python -m genpt stimuli --type all
```

## Reproducing the paper's experiments

The experiments in the paper are organised in three stages.

### Stage 1 — Generate examinee behaviors

For each persona, generate responses to the projective stimuli under each experimental condition (`baseline`, `sdb_job`, `sdb_clinical`, `longctx`):

```bash
# Start a vLLM server (one per GPU if you want throughput)
bash scripts/launch_vllm_plain.sh Qwen/Qwen3-8B 8000 0

# Generate behaviors
python scripts/generate_behaviors.py \
    --sources characterrag \
    --num-groups 1 \
    --output-dir data/behaviors \
    --condition baseline \
    --api-base http://127.0.0.1:8000/v1
```

Additional behavior sets under `--condition sdb_job`, `--condition sdb_clinical`, `--condition longctx` can be generated the same way; these conditions drive the contamination probes used in the paper's stability analysis.

### Stage 2 — Interpreter + Diagnostician (GenPT)

Run the full three-stage assessment pipeline (behavior elicitation → interpretation → diagnosis) end-to-end on all configured personas:

```bash
python scripts/run_experiments.py \
    --scope all \
    --annaagent-count 15
```

Per-persona raw responses, interpretations, and diagnostic labels are written under `results/`.

### Stage 2 (alternate) — Questionnaire baseline

For a direct head-to-head against traditional psychometric questionnaires on the same personas (BFI, 16Personalities, BDI, BSS), use the HuggingFace-backed single-shot evaluator:

```bash
python scripts/evaluate_questionnaire.py \
    --source all \
    --gpu 0
```

## Swapping the Interpreter/Diagnostician backbone

Point `--api-bases` / `--served-model-name` at a different vLLM server. The paper reports Qwen3-8B, Phi-4-mini-reasoning (3.84B), and Intern-S1-mini (~8B). Stage 1 behaviors are held fixed across backbones; only Stages 2–3 are rerun.

## Using your own persona data

The pipeline accepts two persona-source formats:

- **`characterrag`** — a directory per persona containing `{name}[_en].txt` with the persona description. Ground-truth personality labels (MBTI + Big Five) live in `characters/CharacterRAG/pdb_labels.json`.
- **`annaagent`** — a dialogue-grounded format with per-persona depression/suicide risk grades (0–3). The AnnaAgent data itself is not included in this repository because it contains real help-seeker conversations; the loader and evaluation paths (`--source annaagent`, `source_type == "annaagent"` branches in `genpt/` and `scripts/`) are preserved so you can plug in your own dialogue-based clinical personas following the same schema.

## Measurement metrics

- **Big Five / Depression / Suicide** — ordinal labels, evaluated with exact-match accuracy (↑) and mean absolute error (↓).
- **MBTI** — 4-letter type, evaluated with per-axis dimension accuracy (↑), 4-of-4 type accuracy (↑), and Hamming distance (↓, range 0–4).

All metric definitions are in `genpt/evaluation/metrics.py`.
