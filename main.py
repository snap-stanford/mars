'''

@author: maria
'''
  
import torch
import numpy as np
from args_parser import get_parser
from model.mars import MARS
from model.experiment_dataset import ExperimentDataset
from data.maca_facs import MacaData
import warnings
warnings.filterwarnings('ignore')

def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)
    
    
def init_dataset():
    """Init dataset"""

    test_maca = MacaData('../prototypical/dataset/cell_data/tabula-muris-senis-facs-official-annotations.h5ad', annotation_type='cell_ontology_class_reannotated')
    
    print(test_maca.adata)
    test_maca.adata = test_maca.preprocess_data(test_maca.adata)
    tissues = list(set(test_maca.adata.obs['tissue']))
    tissues = sorted(tissues)
    
    test_data = []
    pretrain_data = []
    
    for tissue in tissues:
        tiss = test_maca.get_tissue_data(tissue)
        tiss_test = tiss[tiss.obs['age']=='3m']
        y_test = np.array(tiss_test.obs['truth_labels'], dtype=np.int64)
        
        test_data.append(ExperimentDataset(tiss_test.X.toarray(), tiss_test.obs_names, 
                                           tiss_test.var_names, tissue, y_test))
        pretrain_data.append(ExperimentDataset(tiss.X.toarray(), tiss.obs_names, 
                                         tiss.var_names, tissue))
        
    IDs_to_celltypes = {v:k for k,v in test_maca.celltype_id_map.items()}
     
    return test_data, pretrain_data, IDs_to_celltypes


def main():
    '''
    Initialize everything and train
    '''
    params = get_parser().parse_args()
    print(params)

    if torch.cuda.is_available() and not params.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = 'cuda:0' if torch.cuda.is_available() and params.cuda else 'cpu'
    params.device = device

    init_seed(params)
    test_data, pretrain_data, cell_type_name_map = init_dataset()
    
    
    avg_score_direct = np.zeros((len(test_data), 5))
    
    for idx, unlabeled_data in enumerate(test_data):

        print(unlabeled_data.metadata)
   
        #leave one tissue out
        labeled_data = test_data[:idx]+test_data[idx+1:]
        
        n_clusters = len(np.unique(unlabeled_data.y))
        mars = MARS(n_clusters, params, labeled_data, unlabeled_data, pretrain_data[idx], hid_dim_1=100, hid_dim_2=100)
        adata, landmarks, scores = mars.train(evaluation_mode=True)
        mars.name_cell_types(adata, landmarks, cell_type_name_map)
        
        #adata.write(params.MODEL_DIR+tissue+'/'+tissue+'.h5ad')
        
        avg_score_direct[idx,0] = scores['accuracy']
        avg_score_direct[idx,1] = scores['f1_score']
        avg_score_direct[idx,2] = scores['nmi']
        avg_score_direct[idx,3] = scores['adj_rand']
        avg_score_direct[idx,4] = scores['adj_mi']
        
        print('{}: Acc {}, F1_score {}, NMI {}, Adj_Rand {}, Adj_MI {}'.format(unlabeled_data.metadata, 
                scores['accuracy'],scores['f1_score'],scores['nmi'],
                scores['adj_rand'],scores['adj_mi']))
        
        
    avg_score_direct = np.mean(avg_score_direct,axis=0) 
    print('\nAverage: Acc {}, F1_score {}, NMI {}, Adj_Rand {}, Adj_MI {}\n'.format(avg_score_direct[0],avg_score_direct[1],
            avg_score_direct[2],avg_score_direct[3],avg_score_direct[4]))
        
        
if __name__ == '__main__':
    main()
    
