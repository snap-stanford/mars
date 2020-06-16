'''
Created on Jul 26, 2019

@author: maria
'''

from anndata import read_h5ad
import scanpy.api as sc
import pandas as pd
import numpy as np

class BenchmarkData():
    
    def __init__(self, src_file, annotation_type='truth'):
        """
        annotation type: annotations field in anndata object
        """
        self.adata = read_h5ad(src_file)
        
        self.celltype_id_map = self.celltype_to_numeric(annotation_type)
        sc.pp.filter_genes(self.adata, min_cells=5)
        
        
    def preprocess_data(self, adata, scale=True):
        """Preprocessing dataset: filtering genes/cells, normalization and scaling."""
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
        adata.raw = adata

        #sc.pp.log1p(adata)
        if scale:
            sc.pp.scale(adata, max_value=10, zero_center=True)
            adata[np.isnan(adata.X)] = 0 

        return adata
    
    
    def get_experiment_data(self, experiment):
        """Extract data for a given tissue."""
        tiss = self.adata[self.adata.obs['experiment'] == experiment,:]
        
        return tiss
    
    
    def celltype_to_numeric(self, annotation_type):
        """Adds ground truth clusters data."""
        annotations = list(self.adata.obs[annotation_type])
        annotations_set = sorted(set(annotations))
        
        mapping = {a:idx for idx,a in enumerate(annotations_set)}
        
        truth_labels = [mapping[a] for a in annotations]
        self.adata.obs['truth_labels'] = pd.Categorical(values=truth_labels)
         
        return mapping
