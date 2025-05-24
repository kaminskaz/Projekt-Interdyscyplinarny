import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

def plot_metrics_per_augmentation(out_folder='out_20052025'):
    base_path = os.path.abspath(os.path.join(os.getcwd(), '..', '..', out_folder))
    metrics = ['loss', 'accuracy', 'recall', 'precision', 'f1']
    augment_keywords = ['blur', 'brightness', 'flip', 'noise', 'rotate']
    if out_folder == 'out_21052025':
        augment_keywords = ['combine', 'flip-brightness', 'flip-rotate', 'rotate-brightness', 'flip-rotate-brightness']
    palette = sns.color_palette("Set2", len(metrics))

    # Oblicz baseline
    baseline_means = {metric: [] for metric in metrics}
    baseline_path = os.path.join(base_path, 'baseline_00')
    if os.path.isdir(baseline_path):
        for fname in os.listdir(baseline_path):
            if fname.endswith('test.csv'):
                try:
                    df = pd.read_csv(os.path.join(baseline_path, fname))
                    for metric in metrics:
                        if metric in df.columns:
                            baseline_means[metric].append(df[metric].dropna().iloc[-1])
                except Exception as e:
                    print(f"Błąd przy przetwarzaniu pliku {fname}: {e}")
    baseline_means = {
        metric: sum(values) / len(values) if values else None
        for metric, values in baseline_means.items()
    }

    # Oblicz STES
    stes_means = {metric: [] for metric in metrics}
    stes_path = os.path.join(base_path, 'stes_00')
    if os.path.isdir(stes_path):
        for fname in os.listdir(stes_path):
            if fname.endswith('test.csv'):
                try:
                    df = pd.read_csv(os.path.join(stes_path, fname))
                    for metric in metrics:
                        if metric in df.columns:
                            stes_means[metric].append(df[metric].dropna().iloc[-1])
                except Exception as e:
                    print(f"Błąd przy przetwarzaniu pliku {fname}: {e}")
    stes_means = {
        metric: sum(values) / len(values) if values else None
        for metric, values in stes_means.items()
    }

    # Rysowanie wykresów dla każdej augmentacji
    for aug in augment_keywords:
        all_data = {metric: [] for metric in metrics}

        for folder in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder)
            if not os.path.isdir(folder_path) or aug not in folder.lower():
                continue

            split_name = folder[-2:]  # np. "33" z "brightness_33"
            for fname in os.listdir(folder_path):
                if fname.endswith('test.csv'):
                    try:
                        df = pd.read_csv(os.path.join(folder_path, fname))
                        for metric in metrics:
                            if metric in df.columns:
                                all_data[metric].append({
                                    'split': split_name,
                                    'value': df[metric].dropna().iloc[-1]
                                })
                    except Exception as e:
                        print(f"Błąd przy przetwarzaniu pliku {fname}: {e}")

        if any(all_data[metric] for metric in metrics):
            fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(15, 4), sharey=False)

            for i, metric in enumerate(metrics):
                df_metric = pd.DataFrame(all_data[metric])
                if not df_metric.empty:
                    sns.boxplot(x='split', y='value', data=df_metric, ax=axs[i], color=palette[i])
                    axs[i].set_title(metric.capitalize(), fontsize=12)
                    axs[i].set_xlabel('Splits')
                    axs[i].tick_params(axis='x')

                    # Dodaj linię baseline
                    baseline_val = baseline_means.get(metric)
                    if baseline_val is not None:
                        axs[i].axhline(y=baseline_val, color='red', linestyle='--', linewidth=1.5)
                        axs[i].annotate('Baseline', xy=(0.5, baseline_val),
                                        xycoords=('axes fraction', 'data'),
                                        ha='center', va='bottom',
                                        fontsize=9, color='red')

                    # Dodaj linię STES
                    stes_val = stes_means.get(metric)
                    if stes_val is not None:
                        axs[i].axhline(y=stes_val, color='blue', linestyle='--', linewidth=1.5)
                        axs[i].annotate('Stes', xy=(0.5, stes_val),
                                        xycoords=('axes fraction', 'data'),
                                        ha='center', va='top',
                                        fontsize=9, color='blue')
                else:
                    axs[i].set_visible(False)

            plt.tight_layout()
            plt.suptitle(f"Wyniki na zbiorze testowym ({aug.capitalize()})", fontsize=16, y=1.05)
            plt.show()

def get_best_avg_dfs_for_all_augmentations(out_folder='out_20052025'):
    base_path = os.path.abspath(os.path.join(os.getcwd(), '..', '..', out_folder))
    augment_keywords = ['blur', 'brightness', 'flip', 'rotate', 'noise']
    if out_folder == 'out_21052025':
        augment_keywords = ['combine', 'flip-brightness', 'flip-rotate', 'rotate-brightness', 'flip-rotate-brightness']
    
    best_dfs = {}

    for augmentation_keyword in augment_keywords:
        test_avg_per_folder = {}
        val_data_per_folder = {}

        for folder in os.listdir(base_path):
            if augmentation_keyword not in folder.lower():
                continue

            folder_path = os.path.join(base_path, folder)
            if not os.path.isdir(folder_path):
                continue

            test_dfs = []
            val_dfs = []

            for fname in os.listdir(folder_path):
                if fname.endswith('test.csv'):
                    base_name = fname.replace('test.csv', '')
                    val_name = base_name + 'val.csv'
                    test_path = os.path.join(folder_path, fname)
                    val_path = os.path.join(folder_path, val_name)

                    try:
                        df_test = pd.read_csv(test_path)
                        test_dfs.append(df_test)

                        if os.path.exists(val_path):
                            df_val = pd.read_csv(val_path)
                            val_dfs.append(df_val)
                    except Exception as e:
                        print(f"Błąd podczas wczytywania {fname} w {folder}: {e}")

            if test_dfs and val_dfs:
                avg_test = pd.concat(test_dfs).groupby(level=0).mean().reset_index(drop=True)
                avg_val = pd.concat(val_dfs).groupby(level=0).mean().reset_index(drop=True)
                test_avg_per_folder[folder] = avg_test
                val_data_per_folder[folder] = avg_val

        if test_avg_per_folder:
            best_folder = max(test_avg_per_folder.items(), key=lambda x: x[1]['accuracy'].iloc[-1])[0]
            best_val_df = val_data_per_folder[best_folder]
            print(f"Wybrano folder: {best_folder} dla augmentacji '{augmentation_keyword}' z accuracy (test) = {test_avg_per_folder[best_folder]['accuracy'].iloc[-1]:.4f}")
            best_dfs[augmentation_keyword] = best_val_df
        else:
            print(f"Brak danych dla augmentacji: {augmentation_keyword}")
            best_dfs[augmentation_keyword] = None

    # Obsługa baseline i stes – analogicznie

    def handle_special_case(keyword):
        dfs_test = []
        dfs_val = []
        for folder in os.listdir(base_path):
            if keyword not in folder.lower():
                continue

            folder_path = os.path.join(base_path, folder)
            if not os.path.isdir(folder_path):
                continue

            for fname in os.listdir(folder_path):
                if fname.endswith('test.csv'):
                    base_name = fname.replace('test.csv', '')
                    val_name = base_name + 'val.csv'
                    test_path = os.path.join(folder_path, fname)
                    val_path = os.path.join(folder_path, val_name)

                    try:
                        df_test = pd.read_csv(test_path)
                        dfs_test.append(df_test)

                        if os.path.exists(val_path):
                            df_val = pd.read_csv(val_path)
                            dfs_val.append(df_val)
                    except Exception as e:
                        print(f"Błąd podczas wczytywania {fname} w {folder}: {e}")

        if dfs_test and dfs_val:
            avg_test = pd.concat(dfs_test).groupby(level=0).mean().reset_index(drop=True)
            avg_val = pd.concat(dfs_val).groupby(level=0).mean().reset_index(drop=True)
            print(f"Obliczono średnią dla {keyword} z accuracy (test) = {avg_test['accuracy'].iloc[-1]:.4f}")
            return avg_val
        else:
            print(f"Brak danych dla {keyword}.")
            return None

    best_dfs['baseline'] = handle_special_case('baseline')
    best_dfs['stes'] = handle_special_case('stes')

    return best_dfs

def plot_all_metrics_subplots(best_dfs, out_folder='out_20052025'):
    metrics = ['loss', 'accuracy', 'recall', 'precision', 'f1']
    augmentations = ['baseline', 'stes', 'blur', 'brightness', 'flip', 'rotate', 'noise']
    if out_folder == 'out_21052025':
        augmentations = ['baseline', 'stes', 'combine', 'flip-brightness', 'flip-rotate', 'rotate-brightness', 'flip-rotate-brightness']
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'gray', 'olive']

    fig, axes = plt.subplots(1, 5, figsize=(15, 5), sharex=True)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for j, aug in enumerate(augmentations):
            df = best_dfs.get(aug)
            if df is not None and metric in df.columns:
                ax.plot(df[metric], label=aug, color=colors[j % len(colors)])
        
        ax.set_title(metric.capitalize())
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.grid(True)
        ax.legend()

    fig.suptitle('Wykresy metryk dla najlepszych podziałów zdjęć (na zbiorze walidacyjnym)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def build_summary_dataframe(out_folder='out_20052025'):
    base_path = os.path.abspath(os.path.join(os.getcwd(), '..', '..', out_folder))
    metrics = ['loss', 'accuracy', 'recall', 'precision', 'f1']
    rows = []

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path) or '_' not in folder or len(folder) < 3:
            continue

        try:
            mode, split_code = folder.rsplit('_', 1)
            split_x = split_code[-2] if len(split_code) == 2 else None
            split_y = split_code[-1] if len(split_code) == 2 else None
        except Exception as e:
            print(f"Nie udało się sparsować nazwy folderu: {folder} — {e}")
            continue

        metric_values = {metric: [] for metric in metrics}

        for fname in os.listdir(folder_path):
            if fname.endswith('test.csv'):
                file_path = os.path.join(folder_path, fname)
                try:
                    df = pd.read_csv(file_path)
                    for metric in metrics:
                        if metric in df.columns:
                            value = df[metric].dropna().iloc[-1]
                            metric_values[metric].append(value)
                except Exception as e:
                    print(f"Błąd przy przetwarzaniu pliku {file_path}: {e}")

        row = {
            'Tryb': mode,
            'liczba podziałów x': split_x,
            'liczba podziałów y': split_y
        }

        for metric in metrics:
            values = metric_values[metric]
            row[f'{metric}_mean'] = np.mean(values) if values else None
            row[f'{metric}_std'] = np.std(values, ddof=1) if len(values) > 1 else None

        rows.append(row)

    df_summary = pd.DataFrame(rows)
    return df_summary


def format_value(mean, std, decimals=3):
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"

def generate_latex_table(df: pd.DataFrame) -> str:
    grouped = defaultdict(list)
    
    # Grupowanie według trybu (np. blur, baseline, flip, itd.)
    for _, row in df.iterrows():
        mode = row["Tryb"]
        split = f"{int(row['liczba podziałów x'])}x{int(row['liczba podziałów y'])}"
        values = [
            split,
            format_value(row["loss_mean"], row["loss_std"]),
            format_value(row["accuracy_mean"], row["accuracy_std"]),
            format_value(row["recall_mean"], row["recall_std"]),
            format_value(row["precision_mean"], row["precision_std"]),
            format_value(row["f1_mean"], row["f1_std"]),
        ]
        grouped[mode].append(values)

    latex = []
    for mode, rows in grouped.items():
        latex.append(f"% === {mode} ===")
        if len(rows) == 1:
            # Tylko jeden wiersz – nie trzeba multirow
            line = rows[0]
            latex.append(f"\\multirow{{1}}{{*}}{{{mode}}} & " + " & ".join(line) + " \\\\")
        else:
            latex.append(f"\\multirow{{{len(rows)}}}{{*}}{{{mode}}} & " + " & ".join(rows[0]) + " \\\\")
            for row in rows[1:]:
                latex.append("  & " + " & ".join(row) + " \\\\")
        latex.append("\\hline\n")

    return "\n".join(latex)