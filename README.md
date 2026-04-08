# LeWM — LeWorldModel on PushT

A complete implementation of **LeWorldModel (LeWM)**, the JEPA-based world model from [LeCun et al. (2026)](https://arxiv.org/abs/2603.19312), trained on the PushT manipulation task with a CEM planner for goal-conditioned control.

**LeWM** is a Joint-Embedding Predictive Architecture (JEPA) world model that learns latent dynamics from pixels. Its key innovation is replacing the traditional EMA target encoder with **SIGReg** (Stable Isotropic Gaussian Regularizer) — a regularizer based on the Cramér-Wold theorem that enforces isotropic Gaussian structure in the latent space. This eliminates the need for stop-gradient or exponential moving average, enabling stable end-to-end training with only **one hyperparameter** (λ_reg).

## Architecture

```
                      ┌──────────────┐
  obs (96×96 RGB) ──▶ │   Encoder    │ ──▶ latent z_t (192-d)
                      │  (CNN, 4 blk)│           │
                      └──────────────┘           │
                                                 ▼
                                         ┌───────────────┐
                           action a_t ──▶│   Predictor   │──▶ ẑ_{t+1} (192-d)
                                         │  (MLP, resid) │
                                         └───────────────┘
                                                 │
                      ┌──────────────┐           │    Loss = MSE(ẑ_{t+1}, z_{t+1})
  next_obs ─────────▶ │   Encoder    │ ──▶ z_{t+1}          + λ · SIGReg(z)
                      │  (SAME, full │     (target)
                      │   gradients) │     NO stop-grad!
                      └──────────────┘     NO EMA!
```

### Key Innovation: SIGReg
- **No EMA target encoder** — both encodings use full gradients  
- **No stop-gradient** — end-to-end backprop through everything  
- **SIGReg** enforces isotropic Gaussian structure via random projections + normality testing  
- **Only 1 hyperparameter**: λ_reg (default: 0.1)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the PushT dataset (~12 GB compressed)
python -m lewm_pusht.data.download

# 3. Train LeWM (dry-run first)
python train.py --config configs/pusht.yaml --epochs 2 --batch_size 32

# 4. Train LeWM (full training, ~3 hours on RTX 3090)
python train.py --config configs/pusht.yaml

# 5. Evaluate with CEM planning
python evaluate.py --checkpoint checkpoints/best.pt

# 6. Generate visualizations
python -c "
from lewm_pusht.visualization.visualize import run_all_visualizations
from omegaconf import OmegaConf
config = OmegaConf.load('configs/pusht.yaml')
run_all_visualizations('checkpoints/best.pt', '~/.lewm_data/pusht/pusht_expert_train.h5', config)
"
```

## Project Structure

```
lewm_pusht/
├── data/
│   └── download.py           # HuggingFace dataset downloader
├── models/
│   ├── encoder.py            # CNN encoder: image → latent vector
│   ├── predictor.py          # MLP predictor: latent + action → next latent
│   └── lewm.py               # LeWM wrapper (encoder + predictor + SIGReg)
├── training/
│   ├── dataset.py            # PushT HDF5 dataset loader
│   ├── sigreg.py             # SIGReg Gaussian regularizer
│   └── train.py              # Training loop
├── planning/
│   └── cem.py                # Cross-Entropy Method planner
├── evaluation/
│   └── eval.py               # Success rate evaluation
├── visualization/
│   └── visualize.py          # t-SNE, rollout GIFs, training curves
├── configs/
│   └── pusht.yaml            # All hyperparameters
├── tests/                    # Unit tests
├── train.py                  # Training entry point
├── evaluate.py               # Evaluation entry point
├── requirements.txt
└── README.md
```

## Results

| Model | PushT Success Rate | Mean Planning Time | λ_reg |
|-------|-------------------|--------------------|-------|
| LeWM (ours) | — | — | 0.1 |

*Fill in after training.*

## Running Tests

```bash
cd lewm_pusht
pytest tests/ -v
```

## Configuration

All hyperparameters are in `configs/pusht.yaml`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_dim` | 192 | Latent space dimension |
| `lambda_reg` | 0.1 | SIGReg weight (the ONE hyperparameter) |
| `lr` | 3e-4 | Learning rate (AdamW) |
| `batch_size` | 256 | Training batch size |
| `epochs` | 100 | Training epochs |
| `cem_n_samples` | 512 | CEM population size |
| `cem_horizon` | 10 | CEM planning horizon |

## Dataset

This project uses the official PushT dataset from:
**[quentinll/lewm-pusht](https://huggingface.co/datasets/quentinll/lewm-pusht)** on HuggingFace.

The dataset contains expert demonstrations for the PushT manipulation task (96×96 RGB images, 2D actions).

## Reference

**Paper:** [LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels](https://arxiv.org/abs/2603.19312)  
**Authors:** Maes, Le Lidec, Scieur, LeCun, Balestriero (2026)  
**Code:** [github.com/lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)

### Citation

```bibtex
@article{maes2026leworldmodel,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Justin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint arXiv:2603.19312},
  year={2026}
}
```

## License

This implementation is for research and educational purposes.
