# MARS

PyTorch implementation of MARS, a meta-learning approach for cell type discovery in heterogenous single-cell data.  MARS annotates known and new cell types by transferring latent cell representations across multiple datasets. It is able to discover cell types that have never been seen before and characterize experiments that are yet unannotated. For a detailed description of the algorithm, please see our preprint [Discovering Novel Cell Types across Heterogeneous Single-cell Experiments](https://www.biorxiv.org/content/10.1101/2020.02.25.960302v1) (2020).


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
MARS provides annotations for the unlabeled experimentm as well as embeddings for annotated and unannotated experiments, and stores them in anndata object. In the evaluation_mode annotations for unlabeled experiment need to be provided, and they are used to compute metrics and evaluate the performance of the model. 

MARS embeddings can be visualized in the 2D space using UMAP or tSNE. Example embeddings for diaphragm and liver tissues:

<p align="center">
<img src="https://github.com/snap-stanford/mars/blob/master/images/MARS_embeddings.png" width="400" align="center">
</p>

MARS can generate interpretable names for discovered clusters by calling:
```
mars.name_cell_types(adata, landmarks, cell_type_name_map)
```

Example of the MARS naming approach:

<p align="center">
<img src="https://github.com/mbrbic/mars/blob/master/images/MARS_naming.png" width="600" align="center">
</p>

Example of running MARS on Tabula Muris dataset in leave-one-tissue-out manner is provided in the [main_TM.py](https://github.com/snap-stanford/mars/blob/master/main_TM.py). We also provide two example notebooks that illustrate MARS on small-scale datasets:
- [cellbench.ipynb](https://github.com/snap-stanford/mars/blob/master/notebooks/cellbench.ipynb) demonstrates MARS on two CellBench dataset of five sorted lung cancer cell lines sequenced with 10Xand CEL-Seq2 protocols. 
- [kolod_pollen_bench.ipynb](https://github.com/snap-stanford/mars/blob/master/notebooks/kolod_pollen_bench.ipynb) demonstrates MARS on [Pollen](https://pubmed.ncbi.nlm.nih.gov/25086649/) dataset of diverse human cell types, and [Kolodziejczyk](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4595712/) dataset of mouse pluripotent cells.

## Cross-validation benchmark

We provide cross-validation benchmark [cross_tissue_generator.py](https://github.com/snap-stanford/mars/blob/master/cross_tissue_generator.py) for classifying cell types of Tabula Muris data. The iterator goes over cross-organ train/test splits with an auto-download of Tabula Muris data.

## Datasets

Tabula Muris Senis datasets is from [https://figshare.com/projects/Tabula\_Muris\_Senis/64982](https://figshare.com/projects/Tabula\_Muris\_Senis/64982).

Tabula Muris Senis dataset in h5ad format can be downladed at [http://snap.stanford.edu/mars/data/tms-facs-mars.tar.gz](http://snap.stanford.edu/mars/data/tms-facs-mars.tar.gz).
Small-scale example datasets CellBench and Kolodziejczyk/Pollen can be downloaded at [https://github.com/snap-stanford/mars/blob/master/benchmark_datasets/cellbench_kolod_pollen.tgz](https://github.com/snap-stanford/mars/blob/master/benchmark_datasets/cellbench_kolod_pollen.tgz).

Trained models for each tissue in Tabula Muris can be downladed from [http://snap.stanford.edu/mars/data/TM_trained_models.tar.gz](http://snap.stanford.edu/mars/data/TM_trained_models.tar.gz).

## Contact
Please contact Maria Brbic at mbrbic@cs.stanford.edu for questions.

