
# Behavior-Aware Item Modeling (BAIM)

This repository contains BAIM (Behavior-Aware Item Modeling) models and utilities for building **precomputed multi-stage item embeddings** and training KT models that consume them.

## What's inside

- BAIM KT models (PyKT toolkit fork): `akt_baim`, `qdkt_baim`, `qikt_baim`, `simplekt_baim`, `sparsekt_baim`

## Environment (uv)

This project is configured for Python `3.12.*` and uses `uv`.

```bash
uv sync
```

Run any command via:

```bash
uv run python -c "import torch; print(torch.__version__)"
```
