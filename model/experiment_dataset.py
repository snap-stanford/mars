# coding=utf-8
import torch.utils.data as data
import numpy as np
import torch

'''
Class representing dataset for an single-cell experiment.
'''

IMG_CACHE = {}


class ExperimentDataset(data.Dataset):
    
    
    def __init__(self, x, cells, genes, metadata, y=[]):
        '''
        x: numpy array of gene expressions of cells (rows are cells)
        cells: cell IDs in the order of appearance
        genes: gene IDs in the order of appearance
        metadata: experiment identifier
        y: numeric labels of cells (empty list if unknown)
        '''
        super(ExperimentDataset, self).__init__()
        
        self.nitems = x.shape[0]
        print("== Dataset: Found %d items " % x.shape[0])
        print("== Dataset: Found %d classes" % len(np.unique(y)))
                
        if type(x)==torch.Tensor:
            self.x = x
        else:
            shape = x.shape[1]
            self.x = [torch.from_numpy(inst).view(shape).float() for inst in x]
        if len(y)==0:
            y = np.zeros(len(self.x), dtype=np.int64)
        self.y = tuple(y.tolist())
        self.xIDs = cells
        self.yIDs = genes
        self.metadata = metadata
            
    def __getitem__(self, idx):
        return self.x[idx].squeeze(), self.y[idx], self.xIDs[idx]

    def __len__(self):
        return self.nitems
    
    def get_dim(self):
        return self.x[0].shape[0]
    
    
def load_data(tissue, src_dir='dataset/cell_data/', facs=True): # FIXME
    if facs:
        data_type = 'facs/'
    else: # droplet
        data_type = 'droplet/'
    x = np.loadtxt(src_dir+data_type+tissue+'/X', dtype=np.float32)
    y = np.loadtxt(src_dir+data_type+tissue+'/y', dtype=np.int32)

    with open(src_dir+data_type+tissue+'/cells.tsv') as f:
        data = f.readlines()
    cells = [d.strip() for d in data]
    
    return x,y,cells

