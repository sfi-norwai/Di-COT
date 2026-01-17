import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import tasks
from src.src_utils.load_data import preload_data
from src.src_utils.load_model import initialize_model
import src
from special_tasks.train_model import train_supervised_pretrain
from special_tasks.evaluate_model import eval_supervised
from src.loader.dataloader import stratified_fixed_count, stratified_percentage
from sklearn.preprocessing import label_binarize

SUB_TASK = 'linear_probing'
eval_dir = f'ml_evaluate_csv/'
os.makedirs(eval_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datasets = [
    'wisdm2','skoda','harth','ecg2','sleepm','pamap2'
]


baselines = [
    'Raw_Data','MiniRocket', 'Random_Forest'
]

seeds = [1,2,3,4,5]

# ------------------------------
# MAIN LOOP
# ------------------------------
for dataset in tqdm(datasets, desc="Datasets"):
    
    print(f"\n🔹 Processing dataset: {dataset}")
    config = src.config.Config(f"configs/{dataset}config.yml")

    # Get algorithm and dataset arguments
    for args in src.utils.grid_search(config.ALGORITHM_ARGS):
        args = args
    for ds_args in src.utils.grid_search(config.DATASET_ARGS):
        ds_args = ds_args

    # Load dataset
    train_ds, valid_ds = preload_data(dataset)

    # Subset train dataset to fixed samples per class
    print("Original size:", len(train_ds))
    # train_ds = stratified_fixed_count(train_ds, n_per_class=1)
    train_ds = stratified_percentage(train_ds, percentage=0.10)
    print("Subset size:", len(train_ds))

    # print("Original size:", len(valid_ds))
    # valid_ds = stratified_fixed_count(valid_ds, n_per_class=1)
    # print("Subset size:", len(valid_ds))

    results = []  # store results for this dataset only

    for seed in seeds:

        for method in baselines:
            
            # Load model
            print(f"\nEvaluating method: {method}, seed: {seed}")
    
            # --------------------
            # EVALUATE MODEL
            # --------------------
            if SUB_TASK == 'linear_probing':
                if method == 'Raw_Data':
                    eval_res = tasks.raw_signal_evaluation(train_ds, valid_ds, eval_protocol='linear')
                elif method == 'MiniRocket':
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

    # ------------------------------
    # AGGREGATE RESULTS FOR THIS DATASET
    # ------------------------------
    if results:
        df = pd.DataFrame(results)

        # Count number of runs per (dataset, method)
        df["n"] = df.groupby(["dataset", "method"])["accuracy"].transform("count")

        # --- Aggregate ---
        agg = df.groupby(["dataset", "method"]).agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            n=("n", "mean"),
            train_time=("train_time", lambda x: np.sum([float(t.split()[0]) if t else 0 for t in x])),
        ).reset_index()

        # --- Compute 95% Confidence Intervals ---
        for metric in ["accuracy", "f1", "precision", "recall"]:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"

            agg[f"{metric}_ci_lower"] = agg[mean_col] - 1.96 * (agg[std_col] / np.sqrt(agg["n"]))
            agg[f"{metric}_ci_upper"] = agg[mean_col] + 1.96 * (agg[std_col] / np.sqrt(agg["n"]))

        # Save results
        output_csv = f"{eval_dir}/{dataset}_results.csv"
        agg.to_csv(output_csv, index=False)

        print(agg)
        print(f"✅ Saved {output_csv}")

    else:
        print(f"⚠️ No valid results found for {dataset}")
