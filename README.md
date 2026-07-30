# Procedural Graph

Official implementation and interactive project page for **Knowing What to Do:
Self-Evolving Procedural Graphs for LLM Agents**.

Procedural Graphs externalize an agent's procedural knowledge as directed,
attributed triplets. At every decision step, the framework localizes the active
node, retrieves a connected local subgraph, and translates it into situational
guidance. An offline, validation-gated loop then refines the topology from
successful and failed trajectories.

## Highlights

- Evaluated on 6 main benchmarks plus a 132-month enterprise simulation
- Compared across 4 LLMs and 7 memory-based baselines with a shared ReAct solver
- First in 21 of 24 model-benchmark cells (19 outright, 2 tied)
- Supports expert-authored, scratch-built, and self-evolving graphs
- Includes an interactive, source-backed results explorer on the project page

## Project page

**https://yuxinglu613.github.io/Procedural-Graph/**

The page is a dependency-free static site. To preview it locally:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Repository layout

```text
.
├── index.html                    # GitHub Pages project page
├── styles.css                    # Responsive visual system
├── app.js                        # Interactive method and result explorers
├── assets/                       # Paper figures used by the project page
└── src/
    ├── procedural_graph/         # Graph, guidance, refinement, and solver core
    ├── experiments/run.py        # Unified experiment runner
    ├── analytics/                # Aggregation and visualization scripts
    └── utils/                    # Graph visualization utilities
```

## Installation

Python 3.10+ is recommended.

```bash
git clone https://github.com/YuxingLu613/Procedural-Graph.git
cd Procedural-Graph
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The provided LLM client uses Vertex AI application-default credentials. Keep
credentials outside this repository and configure access through environment
variables:

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/credential.json"
```

## Minimal experiment

```bash
PYTHONPATH=src python3 src/experiments/run.py \
  --dataset=hotpotqa \
  --num_samples=50 \
  --llm=gemini-3.5-flash \
  --llm_location=us-central1 \
  --method=react \
  --guided_only=True \
  --gcp_project="${GOOGLE_CLOUD_PROJECT}"
```

Use `python3 src/experiments/run.py --help` for the complete set of flags.
Benchmark datasets and model access must be configured separately.

## Reproducibility note

The website reports the point estimates and test-set confidence intervals in
the accompanying manuscript. Confidence intervals reflect finite test-set size
from a single greedy-decoding run, not variability across random seeds.

## Citation

A citation entry will be added when the paper is publicly available.

