# Take Note: Your Molecular Dataset Is Probably Aligned
Did you know: Molecular datasets are typically generated with computational chemistry codes which do not randomize pose. Hence, the resulting geometries are usually not randomly oriented.
For example, if you plot the orientations of all molecules in the QM9 dataset in 2D, the result looks far from uniform:

<img src="https://github.com/user-attachments/assets/886b0942-222b-4746-9627-433dd13b8756" 
     width="1000"  />

Even more interesting: Structurally similar molecules are oriented similarly. 

Such orientation bias is easily overlooked but can have serious consequences. Check out our [ICLR26 paper](https://openreview.net/forum?id=zrCGvLOrTL) for more.  

---

A complete walk-through how to detect and quantify orientation bias for the QM9 dataset is given in:

- `notebooks/demo_qm9.ipynb`

## Environment Setup

This project uses `uv` and Python 3.11-3.12. You can find instructions on how to setup uv [here](https://docs.astral.sh/uv/getting-started/installation).

1. Create and sync a CPU environment:

```bash
uv sync --extra cpu
```

2. Or create and sync a CUDA environment:

```bash
uv sync --extra cuda
```

3. (Optional) Activate the virtual environment:

```bash
source .venv/bin/activate
```

4. Launch Jupyter:

```bash
uv run jupyter notebook
```

## Data Path

The demo notebook expects a data directory path in a config cell:

- `DATA_DIR = "../data"`

Update this path if your data lives/should be stored elsewhere.

## What the Demo Offers

The tutorial notebook provides an end-to-end orientation-bias analysis pipeline:

1. Loads QM9 molecules and computes PCA-based molecular orientations.
2. Compares empirical rotation-angle distributions against the theoretical SO(3) reference.
3. Visualizes orientation structure with histograms, reference-axis renders, and a Mollweide projection.
4. Estimates KL divergence from the Haar-uniform SO(3) distribution.
5. Trains a small message passing classifier to distinguish canonical vs randomly rotated samples.
6. Trains an orientation-only regressor and compares it to an uninformative baseline.
