import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import tasks
from src.src_utils.load_data import preload_data
from src.src_utils.load_model import initialize_model
import src
from special_tasks.train_model import train_supervised_pretrain, train_supervised_pretrain_pool
from special_tasks.evaluate_model import eval_supervised, eval_supervised2
from src.loader.dataloader import stratified_fixed_count
from sklearn.preprocessing import label_binarize
from src.loader.dataloader import UEADataset
import random
import time


datasets = [
   # "InsectWingbeat",
    #"EigenWorms",
    "ArticularyWordRecognition",
    "AtrialFibrillation",
    "BasicMotions",
    "CharacterTrajectories",
    "Cricket",
    "DuckDuckGeese",
    "Epilepsy",
    "EthanolConcentration",
    "ERing",
    "FaceDetection",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "JapaneseVowels",
    "Libras",
    "LSST",
    "MotorImagery",
    "NATOPS",
    "PenDigits",
    "PEMS-SF",
    "PhonemeSpectra",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "SpokenArabicDigits",
    "StandWalkJump",
    "UWaveGestureLibrary"
]

baselines = [
    'Raw_Data','MiniRocket', 'Random_Forest'
]

SUB_TASK = 'linear_probing'
# SUB_TASK = 'finetuning'
eval_dir = f'ml_uea_evaluate_csv/{SUB_TASK}'
os.makedirs(eval_dir, exist_ok=True)





def set_seed(seed):
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU & GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Extra (PyTorch 2+ safety)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

    # DataLoader workers
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return seed_worker

# ------------------------------
# MAIN LOOP
# ------------------------------

config = src.config.Config(f"configs/ueaconfig.yml")

# Get algorithm and dataset arguments
for args in src.utils.grid_search(config.ALGORITHM_ARGS):
    args = args
for ds_args in src.utils.grid_search(config.DATASET_ARGS):
    ds_args = ds_args

seed = 1
gpu = 1

device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

# ------------------------------
# AGGREGATE RESULTS FOR ALL DATASETS
# ------------------------------
all_results = []

for dataset in tqdm(datasets, desc="Datasets"):

    print(f"\n🔹 Processing dataset: {dataset}")
    
    # Load dataset
    train_ds = UEADataset(dataset, eval=False)
    valid_ds = UEADataset(dataset, eval=True)

    x, y = train_ds[0]  # get first sample
    args['feature_dim'] = x.shape[1]
    args['sequence_sample'] = x.shape[0]

    train_data_size = args['sequence_sample']*len(train_ds)
    
    args['iterations'] = 200 if train_data_size <= 100000 else 600  # default param for n_iters

    results = []  # store results for this dataset only
    for method in baselines:

        seed_worker = set_seed(seed)


        # --------------------
        # EVALUATE MODEL
        # --------------------
        # Load model
        print(f"\nEvaluating method: {method}, seed: {seed}")

        # --------------------
        # EVALUATE MODEL
        # --------------------
        if SUB_TASK == 'linear_probing':
            if method == 'Raw_Data':
                eval_res = tasks.raw_signal_evaluation(train_ds, valid_ds, eval_protocol='linear')
            elif method == 'MiniRocket':
                if dataset=='PenDigits':
                    eval_res = tasks.raw_signal_evaluation(train_ds, valid_ds, eval_protocol='linear')
                else:
                    eval_res = tasks.minirocket_signal_evaluation(train_ds, valid_ds, eval_protocol='linear')
            elif method == 'Random_Forest':
                eval_res = tasks.raw_signal_evaluation(train_ds, valid_ds, eval_protocol='random_forest')
            else:
                print(f"\nInvalid method: {method}")

        print("Eval result:", eval_res)

        # --------------------
        # STORE METRICS
        # --------------------
        metric_names = ['accuracy', 'f1', 'precision', 'recall']
        avg_metrics = {name: eval_res[i] for i, name in enumerate(metric_names)}
        avg_metrics_std = {name + "_std": 0.0 for name in metric_names}  # single run, std = 0

        results.append({
            "dataset": dataset,
            "method": method,
            "seed": seed,
            **avg_metrics,
            **avg_metrics_std,
            "train_time": 0
        })

    all_results.extend(results)  # collect all datasets

# ------------------------------
# CREATE DATAFRAMES AND SAVE CSVS
# ------------------------------
if all_results:
    df = pd.DataFrame(all_results)

    # Pivot for Accuracy
    acc_df = df.pivot_table(
        index='dataset', 
        columns='method', 
        values='accuracy', 
        aggfunc='mean'
    ).reset_index()
    acc_csv = os.path.join(eval_dir, "accuracy_summary.csv")
    acc_df.to_csv(acc_csv, index=False)
    print(f"✅ Saved accuracy CSV: {acc_csv}")

    # Pivot for Training Time
    time_df = df.pivot_table(
        index='dataset', 
        columns='method', 
        values='train_time', 
        aggfunc='mean'
    ).reset_index()
    time_csv = os.path.join(eval_dir, "train_time_summary.csv")
    time_df.to_csv(time_csv, index=False)
    print(f"✅ Saved training time CSV: {time_csv}")

else:
    print("⚠️ No results to summarize.")
