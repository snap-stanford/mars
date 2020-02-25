# MARS

PyTorch implementation of MARS, a meta-learning method for cell type discovery in heterogenous single-cell data.  MARS annotates known and new cell types by transferring latent cell representations across multiple datasets. It is able to discover cell types that have never been seen before and characterize experiments that are yet unannotated. For a detailed description of the algorithm, please see our preprint [Discovering Novel Cell Types across Heterogeneous Single-cell Experiments](TODO) (2020).


<p align="center">
<img src="https://github.com/mbrbic/mars/blob/master/images/MARS_overview.png" width="1100" align="center">
</p>


## Setup

MARS requires [anndata](https://icb-anndata.readthedocs-hosted.com/en/stable/anndata.AnnData.html) and [scanpy](https://icb-scanpy.readthedocs-hosted.com/en/stable/) libraries. Please check the requirements.txt file for more details on required Python packages. You can create new environment and install all required packages with:

```
pip install -r requirements.txt
```

## Using MARS

We implemented MARS model in a self-contained class. To make an instance and train MARS:

```
mars = MARS(n_clusters, params, labeled_exp, unlabeled_exp, pretrain_data)
adata, landmarks, scores = mars.train(evaluation_mode=True)
```
In the evaluation_mode labeles for unlabeled experiment are provided which MARS uses to compute clustering metrics and evaluate the model. Otherwise, MARS just provides predictions for the unlabeled experiment. MARS stores all embeddings (for labeled and unlabeled data), as well as predictions in anndata object.

Performance of MARS on the Tabula Muris data:

<p align="center">
<img src="https://github.com/mbrbic/mars/blob/master/images/MARS_performance.png" width="400" align="center">
</p>

MARS can also produce interpretable names for discovered clusters by calling:
```
mars.name_cell_types(adata, landmarks, cell_type_name_map)
```

Example of the MARS naming approach:

<p align="center">
<img src="https://github.com/mbrbic/mars/blob/master/images/MARS_naming.png" width="600" align="center">
</p>

Example of running MARS on Tabula Muris dataset in leave-one-tissue-out manner is provided in the main.py.


## Datasets

Tabula Muris Senis datasets used in our study can be downloaded from [https://figshare.com/projects/Tabula\_Muris\_Senis/64982](https://figshare.com/projects/Tabula\_Muris\_Senis/64982).




