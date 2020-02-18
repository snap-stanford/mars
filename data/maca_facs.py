'''
Created on Jul 26, 2019

@author: maria
'''

from anndata import read_h5ad
import scanpy.api as sc
import pandas as pd
import numpy as np

class MacaData():
    
    def __init__(self, src_file, annotation_type='cell_ontology_class_reannotated'):
        """
        annotation type: cell_ontology_class, cell_ontology id or free_annotation
        """
        self.adata = read_h5ad(src_file)
        
        self.adata.obs[annotation_type] = self.adata.obs[annotation_type].astype(str)
        
        self.adata = self.adata[self.adata.obs[annotation_type]!='nan',:]
        self.adata = self.adata[self.adata.obs[annotation_type]!='NA',:]
        self.celltype_id_map = self.celltype_to_numeric(annotation_type)
        sc.pp.filter_genes(self.adata, min_cells=5)
        
        
    def preprocess_data(self, adata, scale=True):
        """Preprocessing dataset: filtering genes/cells, normalization and scaling."""
        sc.pp.filter_cells(adata, min_counts=5000)
        sc.pp.filter_cells(adata, min_genes=500)
        
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
        adata.raw = adata
        
        sc.pp.log1p(adata)
        if scale:
            sc.pp.scale(adata, max_value=10, zero_center=True)
            adata[np.isnan(adata.X)] = 0 

        adata = adata[adata.obs['tissue']!='Marrow',:]
        
        return adata
    
    
    def get_tissue_data(self, tissue):
        """Extract data for a given tissue."""
        tiss = self.adata[self.adata.obs['tissue'] == tissue,:]
        
        return tiss
    
    
    def celltype_to_numeric(self, annotation_type):
        """Adds ground truth clusters data."""
        annotations = list(self.adata.obs[annotation_type])
        annotations_set = sorted(set(annotations))
        
        mapping = {a:idx for idx,a in enumerate(annotations_set)}
        
        truth_labels = [mapping[a] for a in annotations]
        self.adata.obs['truth_labels'] = pd.Categorical(values=truth_labels)
         
        return mapping
    
#md = MacaData('/Users/maria/Documents/Stanford/Single-cell RNA/biohub data/tabula-muris-senis-facs-official-annotations.h5ad')
