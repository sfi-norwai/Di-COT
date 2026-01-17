import torch
import numpy as np
import argparse
import os
import time
import datetime
from baselines.catt import CaTT
from baselines.ts_tcc import TS_TCC
from baselines.tf_c import TF_C

from baselines.rand_init import Rand_Init
from baselines.supervised_b import Supervised_B
from baselines.supervisede2e import SupervisedE2E

from baselines.di_cot import Di_COT

from baselines.cost import CoST
from baselines.ts2vec import TS2Vec
from baselines.soft_ts2vec import Soft
from baselines.infots import InfoTS
from baselines.tnc import TNC
from baselines.simmtm import SimMTM
from utils import name_with_datetime, pkl_save
import wandb
import tasks
import argparse
import random
import src.data
from src.loader.dataloader import PAMAP2Dataset, ECGDataset2, ECGDataset3, ECGDatasetDeterministic
from src.loader.dataloader import WISDM2Dataset, SKODADataset, HARTHDataset, SleepmDataset
from torch.utils.data import Subset
from special_tasks.train_model import train_supervised_pretrain, train_loop, train_supervised
from special_tasks.evaluate_model import eval_supervised
from src.loader.dataloader import stratified_fixed_count

from mdl import InceptionTime


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Start Vanilla CL training.')
    parser.add_argument('model', help='The model name')
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('-p', '--params_path', required=False, type=str,
                        help='params path with config.yml file',
                        default='configs/sleepconfig.yml')
    parser.add_argument('-s', '--seed_value', required=False, type=int,
                        help='seed value.', default=42)
    parser.add_argument('-b', '--batch_size', required=False, type=int,
                        help='seed value.', default=8) 
    parser.add_argument('-v', '--verbose_bool', required=False, type=bool,
                        help='verbose bool.', default=False)
    parser.add_argument('-g', '--gpu', required=False, type=int,
                        help='int.', default=0)
    parser.add_argument('-th', '--max_threads', required=False, type=int,
                        help='number of threads.', default=8)
    parser.add_argument('-iter', '--iterations', required=False, type=int,
                        help='number of iterations.', default=None)
    parser.add_argument('--evaluate', required=False, type=str,
                        help='Task to evaluate on.', default=None)
    
    parser.add_argument('-sp', '--semi_percentage', required=False, type=int,
                        help='percentage of training data.', default=0.01)
    
    parser.add_argument('-trans', '--transfer_data', required=False, type=str,
                        help='The transfer dataset name', default='ecg')
    
   
    
    
    
    pargs = parser.parse_args()
    config_path = pargs.params_path
    # Read config
    config = src.config.Config(config_path)
    config.SEED = pargs.seed_value
    ds_path = pargs.dataset
    verbose = pargs.verbose_bool
    gpu_val = pargs.gpu
    max_threads = pargs.max_threads
    
    # Log in to Wandb
    if config.WANDB:
        wandb.login(key=config.WANDB_KEY)
    
    for ds_args in src.utils.grid_search(config.DATASET_ARGS):
        # Iterate over all model configs if given
        for args in src.utils.grid_search(config.ALGORITHM_ARGS):
            
            seed = config.SEED
            if pargs.iterations is not None:
                 args['iterations'] = pargs.iterations

           # args['feature_dim'] = 3

            device = torch.device(f"cuda:{pargs.gpu}" if torch.cuda.is_available() else "cpu")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

            # Set all seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # Multi-GPU
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            
            # Create the dataset
            if config.DATASET == 'HARTH':
                train_ds = HARTHDataset(
                        data_path=f'datasets/{ds_path}',
                        eval=False
                    )
                
                valid_ds = HARTHDataset(
                        data_path=f'datasets/{ds_path}',
                        eval=True
                    )
                
            elif config.DATASET == 'ECG2':
                train_ds = ECGDataset2(
                        data_path=f'datasets/{ds_path}',
                        eval=False
                    )
                
                valid_ds = ECGDatasetDeterministic(
                        data_path=f'datasets/{ds_path}',
                        eval=True
                    )
            
            elif config.DATASET == 'PAMAP2':
                train_ds = PAMAP2Dataset(
                        data_path=f'datasets/{ds_path}',
                        eval=False
                    )
                
                valid_ds = PAMAP2Dataset(
                        data_path=f'datasets/{ds_path}',
                        eval=True
                    )

            elif config.DATASET == 'SKODA':
                train_ds = SKODADataset(
                        data_path=f'datasets/{ds_path}',
                        eval=False
                    )
                
                valid_ds = SKODADataset(
                        data_path=f'datasets/{ds_path}',
                        eval=True
                    )
                   
            elif config.DATASET == 'WISDM2':
                train_ds = WISDM2Dataset(
                        data_path=f'datasets/{ds_path}',
                        eval=False
                    )
                
                valid_ds = WISDM2Dataset(
                        data_path=f'datasets/{ds_path}',
                        eval=True
                    )
            
            elif config.DATASET == 'SLEEPM':
                train_ds = SleepmDataset(
                        data_path=f'datasets/{ds_path}',
                        eval=False
                    )
                
                valid_ds = SleepmDataset(
                        data_path=f'datasets/{ds_path}',
                        eval=True
                    )

            else:
                raise ValueError(f"Unsupported DATASET: {config.DATASET}")

            
            
            t = time.time()

            if pargs.model == 'CaTT':
                model = CaTT(
                    args,
                    config,
                    device=device
                )

            elif pargs.model == 'Di_COT':
                model = Di_COT(
                    args,
                    config,
                    device=device
                )

            elif pargs.model == 'TS_TCC':
                model = TS_TCC(
                    args,
                    config,
                    device=device
                )
            
            elif pargs.model == 'TF_C':
                model = TF_C(
                    args,
                    config,
                    device=device
                )

            elif pargs.model == 'TS2Vec':
                model = TS2Vec(
                    args,
                    config,
                    device=device
                )

            elif pargs.model == 'CoST':
                model = CoST(
                    args,
                    config,
                    device=device
                )
            
            elif pargs.model == 'Soft':
                model = Soft(
                    args,
                    config,
                    device=device
                )

            elif pargs.model == 'InfoTS':
                model = InfoTS(
                    args,
                    config,
                    device=device
                )

            elif pargs.model == 'TNC':
                model = TNC(
                    args,
                    config,
                    device=device
                )

            elif pargs.model == 'SimMTM':
                model = SimMTM(
                    args,
                    config,
                    device=device
                )

            elif pargs.model == 'Rand_Init':
                model = Rand_Init(
                    args,
                    config,
                    device=device
                )

            elif pargs.model == 'Supervised_B':
                model = Supervised_B(
                    args,
                    config,
                    device=device
                )

            elif pargs.model == 'SupervisedE2E':
                model = SupervisedE2E(
                    args,
                    config,
                    ds_args['num_labels'],
                    device=device
                )

            else:
                raise ValueError(f"Unsupported BASELINE: {pargs.model}")

            #feature_idx = [3, 4, 5]  # select 3 out of 6
            #train_ds = FeatureSubsetDataset(train_ds, feature_idx)
            #valid_ds = FeatureSubsetDataset(valid_ds, feature_idx)

            if pargs.model in ['Supervised_B', 'SupervisedE2E']:
                loss_log = model.fit(train_ds, ds_path, ds_args['num_labels'], args, verbose=pargs.verbose_bool)
            else:
                loss_log = model.fit(train_ds,ds_path,verbose=pargs.verbose_bool)
            
            # model.save(f'{run_dir}/model.pkl')

            # t = time.time() - t
            # print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

            # Select fixed samples per class
            # print("Original size:", len(train_ds))
            # train_ds = stratified_fixed_count(train_ds, n_per_class=50)
            # print("Subset size:", len(train_ds))

            # trained_model = train_supervised_pretrain(model, train_ds, ds_args['num_labels'], args, config, device=device)

            if pargs.evaluate:
                if pargs.evaluate == 'supervised':
                    if pargs.model in ['SimMTM','TNC', 'Di_COT']:
                        eval_res = tasks.supervised_evaluation_ponly_block(model, train_ds, valid_ds, args['out_features'], args['linear_epochs'], args['batch_size'], config)
                    elif pargs.model == 'SupervisedE2E':
                        eval_res = eval_supervised(model, valid_ds, args['out_features'], args['linear_epochs'], args['batch_size'], config)
                    else:
                        eval_res = tasks.supervised_evaluation_ponly(model, train_ds, valid_ds, args['out_features'], args['linear_epochs'], args['batch_size'], config)
                
                elif pargs.evaluate == 'semi_supervised':
                    eval_res = tasks.semi_supervised_evaluation(model, train_ds, valid_ds, args['out_features'], args['linear_epochs'], args['batch_size'], pargs.semi_percentage/100, config)
                elif pargs.evaluate == 'clustering':
                    eval_res = tasks.clustering_evaluation(model, valid_ds, config)
                else:
                    assert False

                # Save evaluation results  
                run_dir = 'training/'
                os.makedirs(run_dir, exist_ok=True)

                time_file = os.path.join(run_dir, f'{pargs.dataset}_{str(config.SEED)}_{name_with_datetime(pargs.model)}.txt')
                with open(time_file, 'w') as f:
                    f.write(str(eval_res))

                print('Evaluation result:', eval_res)

                
    print("Finished.")
