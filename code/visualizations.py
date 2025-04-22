import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Target file suffixes
TARGET_SUFFIXES = ["b0_p_cifar10", "b0_cifar10", "b0_p_fashion", "b0_fashion"]

# Target CSV file suffixes
FILE_SUFFIXES = [
    "Augmentor_combine", "Augmentor_combine_val", "Augmentor_different", "Augmentor_different_val", 
    "Augmentor_same", "Augmentor_same_val", "NoneType_None", "NoneType_None_val", 
    "STESAugmentor_None", "STESAugmentor_None_val"
]

# Metrics to visualize
METRICS = ["loss", "accuracy", "recall", "precision", "f1"]

# Find target folders based on suffix
def find_target_folders(root_path):
    return [
        os.path.join(root_path, d) for d in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, d)) and any(d.endswith(suffix) for suffix in TARGET_SUFFIXES)
    ]

# Generate plots for the given folder
def generate_plots(folder_path):
    # List CSV files matching the suffixes
    csv_files = [
        f for f in os.listdir(folder_path)
        if f.endswith('.csv') and any(f.endswith(suffix + '.csv') for suffix in FILE_SUFFIXES)
    ]
    
    data = {"train": {}, "val": {}}
    
    # Load CSV data for train and validation sets
    for file in csv_files:
        path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(path)
            key = "val" if file.endswith("_val.csv") else "train"
            if file.endswith("_val.csv"):
                #short_label = "_".join(file.removesuffix(".csv").split("_")[-3:])
                file_name = file.removesuffix("_val.csv")
                parts = file_name.split("_")
                short_label = "_".join(parts[-3:])
            else:
                #short_label = file.removesuffix(".csv").split("_")[-2:]
                file_name = file.removesuffix(".csv")
                parts = file_name.split("_")
                short_label = "_".join(parts[-2:]) 
            data[key][short_label] = df
        except Exception as e:
            print(f"Failed to load {file}: {e}")
    
    if not data["train"] and not data["val"]:
        print(f"No data for visualization in {folder_path}")
        return
    
    # Create the output directory for plots
    folder_suffix = Path(folder_path).name.split("_", 1)[-1]
    plots_dir = os.path.join(folder_path, f"plots_{folder_suffix}")
    os.makedirs(plots_dir, exist_ok=True)

    # Generate individual plots for training and validation datasets
    for dataset in ["train", "val"]:
        dataset_data = data[dataset]
        if not dataset_data:
            continue
        for metric in METRICS:
            plt.figure(figsize=(10, 6))
            for label, df in dataset_data.items():
                if metric in df.columns:
                    plt.plot(df["epoch"], df[metric], label=label)
            plt.xlabel("Epoch")
            plt.ylabel(metric.capitalize())
            plt.title(f"{metric.capitalize()} vs Epoch ({dataset})")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{metric}_{dataset}.png"))
            plt.close()

        # Bar plots for final epoch values for train/val
        final_metrics = {metric: [] for metric in METRICS}
        labels = []

        for label, df in dataset_data.items():
            last_row = df.iloc[-1]
            labels.append(label)
            for metric in METRICS:
                final_metrics[metric].append(last_row[metric])

        x = range(len(labels))
        for metric in METRICS:
            plt.figure(figsize=(10, 6))
            plt.bar(x, final_metrics[metric])
            plt.xticks(x, labels, rotation=45, ha="right")
            plt.title(f"Final {metric.capitalize()} ({dataset.capitalize()})")
            plt.ylabel(metric.capitalize())
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"bar_{metric}_{dataset}.png"))
            plt.close()

# Main function to run the script
def main():
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    target_folders = find_target_folders(parent_dir)
    
    for folder in target_folders:
        print(f"Processing: {folder}")
        generate_plots(folder)

# Run the main function if the script is executed
if __name__ == "__main__":
    main()
