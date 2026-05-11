# GRaSp

[![arXiv](https://img.shields.io/badge/arXiv-2605.07454-b31b1b.svg)](https://arxiv.org/abs/2605.07454)

Code for the GRaSp paper on automatic example optimization for in-context learning in low-data tasks (Bihaug-Frøyland & Brådland).

GRaSp selects in-context examples for few-shot prompting in three stages:

1. **Generate** — synthesize a candidate pool from a domain corpus with an LLM.
2. **Reduce** — embed, UMAP-project, and HDBSCAN-cluster the pool.
3. **Select** — run a (µ+λ) GA (DEAP) with diversity-adaptive inter/intra-cluster mutation.

## Setup

Python ≥3.10. The pipeline talks to two OpenAI-compatible HTTP endpoints (one generative LLM, one embedding model). Scripts to launch them locally with vLLM are in `scripts/`.

```sh
git clone https://github.com/sbfroy/GRaSp.git
cd GRaSp
pip install -e .
cp .env.example .env   # fill in endpoints / keys
```

Or with Docker:

```sh
docker build -t grasp:1.0 .
docker run -it --gpus all --shm-size 32GB \
  -e WANDB_ENTITY=<user> -e WANDB_API_KEY=<key> \
  -d grasp:1.0
```

## Running

The FiNER-139 experiments from the paper are in `tasks/financial_ner/`, configured via `config.yaml`. Run the stages in order:

```sh
python tasks/financial_ner/run_generate.py
python tasks/financial_ner/run_cluster.py
python tasks/financial_ner/run_evolve.py
```

## Citation

If you use this code or build on this work, please cite:

```bibtex
@article{bihaugfroyland2026grasp,
  title={{GRaSp}: Automatic Example Optimization for In-Context Learning in Low-Data Tasks},
  author={Bihaug-Fr{\o}yland, Simen and Br{\aa}dland, Henrik},
  journal={arXiv preprint arXiv:2605.07454},
  year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
