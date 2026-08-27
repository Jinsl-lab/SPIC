# SPIC: Spatial reconstruction of scRNA-seq data via cross-modal manifold alignment
SPIC is a deep learning framework designed for recovering spatial context of single cells within scRNA-seq datasets. SPIC leverages Riemannian geometry and algebraic topology to construct a shared nonlinear manifold for integrating scRNA-seq and ST data, followed by a position prediction neural network that directly infers precise spatial coordinates from the learned low-dimensional embeddings.
<p align="center">
  <img src="overview.png" width="850">
</p>

## Installation

The required environment can be created using:

```bash
conda env create -f environment.yml
conda activate SPIC
```
## Usage

SPIC consists of two main steps: **data integration** and **spatial prediction**. Detailed examples are available in the `tutorial` folder.

### 1. Data Integration

Integrate scRNA-seq and spatial transcriptomics data into a shared latent space:

```python
adata = Integration(
    [sc_adata, st_adata], ['X', 'X'],
    seed=2024,
    n_epochs=180,
    n_comps=20,
    n_neighbors=50,
    n_components=32
)
```

### 2. Spatial Prediction

Train the spatial prediction model using the spatial reference and predict the coordinates of scRNA-seq cells:

```python
trained_model = Fit_cord(
    data_train=st_embedding,
    out_features=2,
    location_data=st_location,
    hidden_dims=[512, 256, 128, 64, 32],
    num_epochs=500,
    batch_size=32,
    initial_learning_rate=0.001,
    seednum=2024,
    device="cuda:0"
)

pred_coord = Predict_cord(
    data_test=sc_embedding,
    model=trained_model,
    location_data=None
)
```

For complete examples and parameter settings, please refer to the notebooks in the `tutorial` folder.
