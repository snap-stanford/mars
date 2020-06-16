'''

@author: maria
'''
  
import scanpy.api as sc
import numpy as np
import urllib.request
import gzip
import shutil
import os

    
class CrossTissueBenchmark():
    
    def __init__(self, download=True, dir_path='.', tabula_muris_senis=False):
        '''
        download: if True, data will be downloaded automatically and saved in dir_path, otherwise
                  data will be read from dir_path
        dir_path: path to directory where data is stored (if already downloaded), or where the data
                  should be saved 
        tabula_muris_senis: if False generator for Tabula Muris data  only will be created, otherwise
                            for Tabula Muris Senis
        ''' 
        if download:
            self.download_data(dir_path)
        self.adata = sc.read_h5ad(os.path.join(dir_path,'tms-facs-mars.h5ad'))
        self.preprocess()
        if not tabula_muris_senis:
            self.adata = self.adata[self.adata.obs['age']=='3m']
        
    def download_data(self, filepath):
        urllib.request.urlretrieve('http://snap.stanford.edu/mars/data/tms-facs-mars.tar.gz', os.path.join(filepath,'tms-facs-mars.tar.gz'))
        with gzip.open(os.path.join(filepath,'tms-facs-mars.tar.gz'), 'rb') as f_in:
            with open( os.path.join(filepath,'tms-facs-mars.h5ad'), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            
    def preprocess(self, annotation_type='cell_ontology_class_reannotated'):
        self.adata.obs[annotation_type] = self.adata.obs[annotation_type].astype(str)
        
        self.adata = self.adata[self.adata.obs[annotation_type]!='nan',:]
        self.adata = self.adata[self.adata.obs[annotation_type]!='NA',:]
        
        sc.pp.filter_genes(self.adata, min_cells=5)
        sc.pp.filter_cells(self.adata, min_counts=5000)
        sc.pp.filter_cells(self.adata, min_genes=500)
        sc.pp.normalize_per_cell(self.adata, counts_per_cell_after=1e4)
        sc.pp.log1p(self.adata)
        sc.pp.scale(self.adata, max_value=10, zero_center=True)
        self.adata[np.isnan(self.adata.X)] = 0 

       
    def cross_tissue_generator(self):
        
        tissues = list(set(self.adata.obs['tissue']))
        tissues = sorted(tissues)
        
        for tissue in tissues:
            test = self.adata[self.adata.obs['tissue'] == tissue,:]
            train = self.adata[self.adata.obs['tissue'] != tissue,:]
        
            yield (train, test)


