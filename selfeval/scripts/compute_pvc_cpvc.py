import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import pearsonr
import os

_parser = argparse.ArgumentParser(
    description="Compute PVC / C-PVC / VUS / PM-VUS metrics from a released self-eval CSV."
)
_parser.add_argument(
    "--input",
    default="problem_evaluations_math500.csv",
    help="Path to a released self_eval CSV (see model_outputs/).",
)
_parser.add_argument(
    "--output",
    default=None,
    help="Optional path to write the per-model summary CSV. Default: alongside input.",
)
_args, _ = _parser.parse_known_args()

output_csv = _args.input
data = pd.read_csv(output_csv)
_basename = os.path.splitext(os.path.basename(output_csv))[0]
_cleaned = _basename.replace("problem_evaluations_", "").replace("_self_eval", "")
_name_map = {
    "mathbenchmark": "Math360",
    "math360": "Math360",
    "truthfulQA": "TruthfulQA",
    "truthfulqa": "TruthfulQA",
    "CSQA": "CSQA",
    "commonsenseqa": "CSQA",
    "math500": "Math500",
}
dataset_name = _name_map.get(_cleaned, _cleaned)

def combined_category_accuracy_plots(df, dataset_name, output_file=None, threshold=0.5, gamma_step=0.01):
    """
    Create a combined figure with two subplots:
    1. Bar plot showing self-reflection accuracy by category for each model
    2. Line plot showing how many categories have accuracy greater than gamma threshold
    """
    model_order = df['model_id'].unique()
    category_order = df['category'].unique()
    
    accuracy_df = df.groupby(['model_id', 'category'])['self_eval_correct'].mean().reset_index()
    
    accuracy_df['model_id'] = pd.Categorical(accuracy_df['model_id'], categories=model_order, ordered=True)
    accuracy_df['category'] = pd.Categorical(accuracy_df['category'], categories=category_order, ordered=True)
    
    base_pastel_colors = [
        '#8da0cb', '#fc8d62', '#66c2a5', '#e78ac3', '#ffb347', '#b19cd9', '#87cefa', '#90ee90',
        '#ff9999', '#d4a5ff', '#ffcc99', '#c2c2f0', '#ffb6c1', '#c3e6cb', '#ffd700', '#aec6cf',
        '#cb99c9', '#fdfd96', '#cccccc', '#ff6666'
    ]
    pastel_colors = base_pastel_colors[:len(model_order)]
    color_dict = {model: color for model, color in zip(model_order, pastel_colors)}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1.2, 0.8]})
    sns.set(style="whitegrid")
    
    # LEFT SUBPLOT: BAR PLOT
    bars = sns.barplot(
        x='category', y='self_eval_correct', hue='model_id', data=accuracy_df,
        palette=color_dict, hue_order=model_order, order=category_order,
        ax=ax1, alpha=0.8
    )
    
    ax1.set_title(f'Self-Reflection Accuracy by Category - {dataset_name}', fontsize=16)
    ax1.set_xlabel('Category', fontsize=14)
    ax1.set_ylabel('Self-Reflection Accuracy', fontsize=14)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha='right')
    
    ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.text(-0.5, threshold + 0.02, f'γ = {threshold}', color='red', fontsize=10, fontweight='bold')
    
    for i, container in enumerate(bars.containers):
        ax1.bar_label(container, fmt='%.2f', padding=3 + i, fontsize=7)
    
    min_val = max(0, accuracy_df['self_eval_correct'].min() - 0.05)
    max_val = min(1, accuracy_df['self_eval_correct'].max() + 0.1)
    ax1.set_ylim(min_val, max_val)
    ax1.get_legend().remove()
    
    # RIGHT SUBPLOT: LINE PLOT
    gamma_values = np.arange(0.3, 1.01, gamma_step)
    results = {model: [] for model in model_order}
    
    for gamma in gamma_values:
        for model in model_order:
            model_data = accuracy_df[accuracy_df['model_id'] == model]
            count = sum(model_data['self_eval_correct'] > gamma)
            results[model].append(count)
    
    total_categories = len(category_order)
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', 
                  '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':']
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 
              '+', 'x', 'd', '|', '_', '1', '2', '3', '4', 'P']
    
    for i, model in enumerate(model_order):
        ax2.plot(
            gamma_values, results[model], label=model, color=color_dict[model],
            linewidth=2.5, linestyle=line_styles[i % len(line_styles)],
            marker=markers[i % len(markers)], markersize=5,
            markevery=int(10/gamma_step)
        )
    
    ax2.axhline(y=total_categories, color='gray', linestyle=':', linewidth=1.5, 
                alpha=0.7, label=f'Total Categories ({total_categories})')
    
    ax2.set_title(f'PVC - γ ({dataset_name})', fontsize=16)
    ax2.set_xlabel('Threshold (γ)', fontsize=14)
    ax2.set_ylabel('Number of Categories with Accuracy > γ', fontsize=14)
    ax2.set_xticks(np.arange(0.3, 1.1, 0.1))
    ax2.set_yticks(range(0, total_categories + 1))
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(title='Model', loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    plt.show()

def plot_category_calibration_error(df, dataset_name, output_file=None):
    """
    Create a bar plot showing calibration error (self-confidence - actual accuracy)
    by category for each model.
    """
    model_order = df['model_id'].unique()
    category_order = df['category'].unique()
    
    accuracy_df = df.groupby(['model_id', 'category'])['self_eval_correct'].mean().reset_index()
    accuracy_df = accuracy_df.rename(columns={'self_eval_correct': 'actual_accuracy'})
    
    confidence_df = df.groupby(['model_id', 'category'])['self_eval_confidence'].mean().reset_index()
    calibration_df = pd.merge(accuracy_df, confidence_df, on=['model_id', 'category'])
    calibration_df['calibration_error'] = calibration_df['self_eval_confidence'] - calibration_df['actual_accuracy']
    
    calibration_df['model_id'] = pd.Categorical(calibration_df['model_id'], categories=model_order, ordered=True)
    calibration_df['category'] = pd.Categorical(calibration_df['category'], categories=category_order, ordered=True)
    calibration_df = calibration_df.sort_values(['model_id', 'category'])
    
    pastel_colors = [
        '#8da0cb', '#fc8d62', '#66c2a5', '#e78ac3', '#ffb347', '#b19cd9', '#87cefa', '#90ee90',
        '#ff9999', '#d4a5ff', '#ffcc99', '#c2c2f0', '#ffb6c1', '#c3e6cb', '#ffd700', '#aec6cf',
        '#cb99c9', '#fdfd96', '#cccccc', '#ff6666'
    ]
    colors = pastel_colors[:len(model_order)]
    color_dict = {model: color for model, color in zip(model_order, colors)}
    
    width = 0.8 / len(model_order)
    x = np.arange(len(category_order))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, model in enumerate(model_order):
        model_data = calibration_df[calibration_df['model_id'] == model]
        model_data = model_data.sort_values('category')
        
        model_x = x - (len(model_order) - 1) * width / 2 + i * width
        
        bars = ax.bar(
            model_x, model_data['calibration_error'], width, 
            label=model, color=color_dict[model], alpha=0.8
        )
        
        for j, bar in enumerate(bars):
            height = bar.get_height()
            value = model_data['calibration_error'].iloc[j]
            va = 'bottom' if value >= 0 else 'top'
            offset = 0.01 if value >= 0 else -0.01
            fontsize = 7 if len(model_order) <= 8 else 6
            rotation = 0 if len(model_order) <= 10 else 90
            
            ax.text(
                bar.get_x() + bar.get_width() / 2, height + offset,
                f'{value:.2f}', ha='center', va=va, fontsize=fontsize, rotation=rotation
            )
    
    ax.axhline(y=0, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(category_order, rotation=30, ha='right')
    ax.set_xlabel('Category', fontsize=14)
    ax.set_ylabel('Calibration Error (Confidence - Accuracy)', fontsize=14)
    ax.set_title(f'Calibration Error by Category and Model - {dataset_name}', fontsize=18)
    
    if len(model_order) > 10:
        ax.legend(title='Model', loc='upper right', frameon=True, fancybox=True, 
                 shadow=True, fontsize=8, ncol=2)
    else:
        ax.legend(title='Model', loc='upper right', frameon=True, fancybox=True, 
                 shadow=True, fontsize=10)
    
    plt.text(
        0.01, 0.02,
        "Positive values = Overconfident\nNegative values = Underconfident\nZero = Perfectly calibrated",
        transform=ax.transAxes, fontsize=9, va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
    )
    
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    plt.show()
    return calibration_df

def calculate_judge_correlations_by_model_category(df):
    """
    Calculate correlation between self-evaluation answers and judge answers,
    grouped by model_id and category.
    """
    df['self_eval_matched_correct'] = (df['self_eval_answer'] == df['correct_answer']).astype(int)
    df['judge_a_matched_correct'] = (df['judge_a_answer'] == df['correct_answer']).astype(int)
    df['judge_b_matched_correct'] = (df['judge_b_answer'] == df['correct_answer']).astype(int)
    df['judge_c_matched_correct'] = (df['judge_c_answer'] == df['correct_answer']).astype(int)
    
    df['self_eval_agrees_with_judge_a'] = (df['self_eval_answer'] == df['judge_a_answer']).astype(int)
    df['self_eval_agrees_with_judge_b'] = (df['self_eval_answer'] == df['judge_b_answer']).astype(int)
    df['self_eval_agrees_with_judge_c'] = (df['self_eval_answer'] == df['judge_c_answer']).astype(int)
    
    results = []
    
    for (model_id, category), group in df.groupby(['model_id', 'category']):
        if len(group) < 3:
            continue
        
        for judge in ['a', 'b', 'c']:
            if (group[f'self_eval_matched_correct'].var() > 0 and 
                group[f'judge_{judge}_matched_correct'].var() > 0):
                correlation = pearsonr(
                    group['self_eval_matched_correct'], 
                    group[f'judge_{judge}_matched_correct']
                )[0]
            else:
                correlation = np.nan
            
            agreement_rate = group[f'self_eval_agrees_with_judge_{judge}'].mean()
            
            results.append({
                'model_id': model_id,
                'category': category,
                'judge': f'judge_{judge}',
                'correlation': correlation,
                'agreement_rate': agreement_rate,
                'sample_size': len(group)
            })
    
    return pd.DataFrame(results)

def get_overall_summary(df):
    """
    Calculate overall correlation metrics across all data,
    without grouping by model or category.
    """
    df['self_eval_matched_correct'] = (df['self_eval_answer'] == df['correct_answer']).astype(int)
    df['judge_a_matched_correct'] = (df['judge_a_answer'] == df['correct_answer']).astype(int)
    df['judge_b_matched_correct'] = (df['judge_b_answer'] == df['correct_answer']).astype(int)
    df['judge_c_matched_correct'] = (df['judge_c_answer'] == df['correct_answer']).astype(int)
    
    df['self_eval_agrees_with_judge_a'] = (df['self_eval_answer'] == df['judge_a_answer']).astype(int)
    df['self_eval_agrees_with_judge_b'] = (df['self_eval_answer'] == df['judge_b_answer']).astype(int)
    df['self_eval_agrees_with_judge_c'] = (df['self_eval_answer'] == df['judge_c_answer']).astype(int)
    
    overall_results = {}
    
    for judge in ['a', 'b', 'c']:
        corr = pearsonr(
            df['self_eval_matched_correct'],
            df[f'judge_{judge}_matched_correct']
        )[0]
        
        agreement = df[f'self_eval_agrees_with_judge_{judge}'].mean()
        
        overall_results[f'judge_{judge}'] = {
            'correlation': corr,
            'agreement_rate': agreement
        }
    
    return overall_results

def calculate_pvc_dimension(df: pd.DataFrame, gamma: float = 0.7) -> Dict[str, int]:
    """Calculate the PVC dimension for each model at the given confidence threshold."""
    pvc_dimensions = {}
    
    for model in df['model_id'].unique():
        model_df = df[df['model_id'] == model]
        shattered_categories = 0
        
        for category in model_df['category'].unique():
            cat_df = model_df[model_df['category'] == category]
            
            high_conf_rows = cat_df[cat_df['self_eval_confidence'] >= gamma]
            if len(high_conf_rows) == 0:
                p_eval = 0
            else:
                p_eval = high_conf_rows['actual_accuracy'].mean()
            
            if p_eval >= gamma:
                shattered_categories += 1
        
        pvc_dimensions[model] = shattered_categories
    
    return pvc_dimensions

def calculate_calibration_metrics(df: pd.DataFrame, num_bins: int = 10) -> Dict[str, Dict[str, float]]:
    """Calculate calibration metrics for each model."""
    calibration_metrics = {}
    
    for model in df['model_id'].unique():
        model_df = df[df['model_id'] == model]
        
        conf_values = model_df['self_eval_confidence'].values
        acc_values = model_df['actual_accuracy'].values
        
        bin_size = 1.0 / num_bins
        bins = [i * bin_size for i in range(num_bins + 1)]
        
        bin_indices = np.digitize(conf_values, bins) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins-1)
        
        bin_counts = np.zeros(num_bins)
        bin_acc_sum = np.zeros(num_bins)
        bin_conf_sum = np.zeros(num_bins)
        
        for i in range(len(bin_indices)):
            bin_idx = bin_indices[i]
            bin_counts[bin_idx] += 1
            bin_acc_sum[bin_idx] += acc_values[i]
            bin_conf_sum[bin_idx] += conf_values[i]
        
        bin_acc = bin_acc_sum / np.maximum(bin_counts, 1)
        bin_conf = bin_conf_sum / np.maximum(bin_counts, 1)
        
        ece = np.sum(bin_counts * np.abs(bin_acc - bin_conf)) / np.sum(bin_counts)
        mce = np.max(np.abs(bin_acc - bin_conf))
        
        overconfidence = np.sum(bin_counts * np.maximum(bin_conf - bin_acc, 0)) / np.sum(bin_counts)
        underconfidence = np.sum(bin_counts * np.maximum(bin_acc - bin_conf, 0)) / np.sum(bin_counts)
        
        calibration_metrics[model] = {
            'ECE': ece,
            'MCE': mce,
            'Overconfidence': overconfidence,
            'Underconfidence': underconfidence,
            'Mean Accuracy': model_df['actual_accuracy'].mean(),
            'Mean Confidence': model_df['self_eval_confidence'].mean()
        }
    
    return calibration_metrics

def calculate_cpvc_dimension(df: pd.DataFrame, gamma: float = 0.7, tau: float = 0.1) -> Dict[str, int]:
    """Calculate the Calibration-aware PVC dimension (C-PVC) for each model."""
    cpvc_dimensions = {}
    
    for model in df['model_id'].unique():
        model_df = df[df['model_id'] == model]
        calibration_shattered_categories = 0
        
        for category in model_df['category'].unique():
            cat_df = model_df[model_df['category'] == category]
            
            high_conf_rows = cat_df[cat_df['self_eval_confidence'] >= gamma]
            if len(high_conf_rows) == 0:
                continue
            
            cal_error = np.mean(np.abs(high_conf_rows['self_eval_confidence'] - high_conf_rows['actual_accuracy']))
            p_eval = high_conf_rows['actual_accuracy'].mean()
            
            if p_eval >= gamma and cal_error <= tau:
                calibration_shattered_categories += 1
        
        cpvc_dimensions[model] = calibration_shattered_categories
    
    return cpvc_dimensions

def calculate_brier_score(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate Brier score for each model."""
    brier_scores = {}
    
    for model in df['model_id'].unique():
        model_df = df[df['model_id'] == model]
        
        # Brier score = mean((confidence - actual_accuracy)^2)
        brier_score = np.mean((model_df['self_eval_confidence'] - model_df['actual_accuracy']) ** 2)
        brier_scores[model] = brier_score
    
    return brier_scores

def category_performance_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance across different categories."""
    categories = df['category'].unique()
    models = df['model_id'].unique()
    
    results = []
    
    for category in categories:
        for model in models:
            cat_model_df = df[(df['category'] == category) & (df['model_id'] == model)]
            
            if not cat_model_df.empty:
                accuracy = cat_model_df['actual_accuracy'].mean()
                confidence = cat_model_df['self_eval_confidence'].mean()
                calibration_error = np.mean(np.abs(cat_model_df['self_eval_confidence'] - cat_model_df['actual_accuracy']))
                
                results.append({
                    'model_id': model,
                    'category': category,
                    'accuracy': accuracy,
                    'confidence': confidence,
                    'calibration_error': calibration_error
                })
    
    return pd.DataFrame(results)

def analyze_sample_complexity(pvc_dims: Dict[str, int], gamma: float = 0.7, 
                             tau: float = 0.1, epsilon:float = 0.1, delta: float = 0.05) -> pd.DataFrame:
    """Calculate the sample complexity requirements based on PVC dimensions."""
    C = 1
    results = []
    
    for model, d in pvc_dims.items():
        m = int(np.ceil((C / (epsilon**2)) * (d + np.log(1/delta))))
        
        results.append({
            'model_id': model,
            'PVC Dimension': d,
            'Sample Complexity': m
        })
    
    return pd.DataFrame(results)

def predict_generalization_error(df: pd.DataFrame, gamma: float = 0.7, 
                                tau: float = 0.1) -> pd.DataFrame:
    """Predict generalization error bounds based on calibration-aware PVC theory."""
    results = []
    
    for model in df['model_id'].unique():
        model_df = df[df['model_id'] == model]
        
        high_conf_rows = model_df[model_df['self_eval_confidence'] >= gamma]
        if len(high_conf_rows) == 0:
            delta = 1.0
        else:
            cal_violations = high_conf_rows[np.abs(high_conf_rows['self_eval_confidence'] - 
                                                high_conf_rows['actual_accuracy']) > tau]
            delta = len(cal_violations) / len(high_conf_rows)
        
        error_bound = (1 - gamma) + tau + delta
        actual_error = 1 - model_df['actual_accuracy'].mean()
        
        results.append({
            'model_id': model,
            'Predicted Error Bound': error_bound,
            'Actual Error': actual_error,
            'Gap': error_bound - actual_error
        })
    
    return pd.DataFrame(results)

def create_comprehensive_table(results: Dict, calibration_df: pd.DataFrame, gamma: float, tau: float) -> pd.DataFrame:
    """Create a comprehensive table with all key metrics."""
    table_data = []
    
    # Calculate Brier scores using the calibration DataFrame
    brier_scores = calculate_brier_score(calibration_df)
    
    for model in results['pvc_dimensions'].keys():
        # Get sample complexity for this model
        sample_complexity_row = results['sample_complexity'][
            results['sample_complexity']['model_id'] == model
        ]
        sample_complexity = sample_complexity_row['Sample Complexity'].iloc[0] if not sample_complexity_row.empty else 0
        
        # Get actual error for this model
        error_prediction_row = results['error_predictions'][
            results['error_predictions']['model_id'] == model
        ]
        actual_error = error_prediction_row['Actual Error'].iloc[0] if not error_prediction_row.empty else 0
        
        table_data.append({
            'Model': model,
            'PVC': results['pvc_dimensions'][model],
            'C-PVC': results['cpvc_dimensions'][model],
            'ECE': round(results['calibration_metrics'][model]['ECE'], 4),
            'Brier': round(brier_scores.get(model, 0), 4),
            'Sample_Complexity': sample_complexity,
            'Actual_Error': round(actual_error, 4)
        })
    
    return pd.DataFrame(table_data)

def plot_calibration_curve(df: pd.DataFrame, model_ids: List[str] = None, 
                          num_bins: int = 10, figsize: Tuple[int, int] = (12, 8)) -> None:
    """Plot calibration curves for the specified models."""
    if model_ids is None:
        model_ids = df['model_id'].unique()
    
    plt.figure(figsize=figsize)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    for model in model_ids:
        model_df = df[df['model_id'] == model]
        
        bin_size = 1.0 / num_bins
        bins = [i * bin_size for i in range(num_bins + 1)]
        bin_indices = np.digitize(model_df['self_eval_confidence'].values, bins) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins-1)
        
        bin_confidences = []
        bin_accuracies = []
        
        for i in range(num_bins):
            bin_mask = (bin_indices == i)
            if np.sum(bin_mask) > 0:
                bin_conf = np.mean(model_df['self_eval_confidence'].values[bin_mask])
                bin_acc = np.mean(model_df['actual_accuracy'].values[bin_mask])
                bin_confidences.append(bin_conf)
                bin_accuracies.append(bin_acc)
        
        plt.plot(bin_confidences, bin_accuracies, 'o-', label=f"{model}")
    
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Calibration Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_pvc_vs_cpvc(pvc_dims: Dict[str, int], cpvc_dims: Dict[str, int],
                     figsize: Tuple[int, int] = (10, 6)) -> None:
    """Plot PVC dimension vs. Calibration-aware PVC dimension."""
    plt.figure(figsize=figsize)
    
    models = list(pvc_dims.keys())
    x_vals = [pvc_dims[model] for model in models]
    y_vals = [cpvc_dims[model] for model in models]
    
    plt.scatter(x_vals, y_vals, s=100)
    
    max_val = max(max(x_vals), max(y_vals))
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.7, label='PVC = C-PVC')
    
    for i, model in enumerate(models):
        plt.annotate(model, (x_vals[i], y_vals[i]), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.xlabel('PVC Dimension')
    plt.ylabel('Calibration-aware PVC Dimension')
    plt.title('Comparison of PVC and C-PVC Dimensions')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

def plot_category_heatmap(df: pd.DataFrame, metric: str = 'accuracy',
                         figsize: Tuple[int, int] = (12, 8)) -> None:
    """Plot a heatmap of model performance across categories."""
    pivot_df = df.pivot(index='model_id', columns='category', values=metric)
    
    plt.figure(figsize=figsize)
    sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=.5)
    
    plt.title(f'Model {metric.capitalize()} by Category')
    plt.tight_layout()
    plt.show()

def plot_confidence_distributions(df: pd.DataFrame, figsize: Tuple[int, int] = (14, 8)) -> None:
    """Plot distributions of confidence scores for each model."""
    plt.figure(figsize=figsize)
    
    models = df['model_id'].unique()
    num_models = len(models)
    
    for i, model in enumerate(models, 1):
        model_df = df[df['model_id'] == model]
        
        plt.subplot(1, num_models, i)
        sns.histplot(model_df['self_eval_confidence'], bins=20, kde=True)
        plt.axvline(model_df['self_eval_confidence'].mean(), color='r', linestyle='--', 
                   label=f'Mean = {model_df["self_eval_confidence"].mean():.2f}')
        
        plt.title(f'{model}')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def run_comprehensive_analysis(df: pd.DataFrame, gamma: float = 0.7, tau: float = 0.1) -> Dict:
    """Run a comprehensive analysis of PVC dimensions and related metrics."""
    pvc_dims = calculate_pvc_dimension(df, gamma)
    cpvc_dims = calculate_cpvc_dimension(df, gamma, tau)
    cal_metrics = calculate_calibration_metrics(df)
    cat_analysis = category_performance_analysis(df)
    sample_complexity = analyze_sample_complexity(cpvc_dims, gamma, tau)
    error_predictions = predict_generalization_error(df, gamma, tau)
    
    print("Generating PVC vs C-PVC comparison plot...")
    plot_pvc_vs_cpvc(pvc_dims, cpvc_dims)
    
    print("Generating calibration curves...")
    plot_calibration_curve(df)
    
    print("Generating heatmap of accuracy by category...")
    plot_category_heatmap(cat_analysis, 'accuracy')
    
    print("Generating heatmap of calibration error by category...")
    plot_category_heatmap(cat_analysis, 'calibration_error')
    
    print("Generating confidence distributions...")
    plot_confidence_distributions(df)
    
    results = {
        'pvc_dimensions': pvc_dims,
        'cpvc_dimensions': cpvc_dims,
        'calibration_metrics': cal_metrics,
        'category_analysis': cat_analysis,
        'sample_complexity': sample_complexity,
        'error_predictions': error_predictions
    }
    
    return results

def combined_calibration_pvc_plot(df, pvc_dims, cpvc_dims, dataset_name, output_file=None, threshold=0):
    """
    Create a combined figure with two subplots:
    1. Bar plot showing calibration error by category for each model (left)
    2. Scatter plot comparing PVC and C-PVC dimensions (right)
    """
    model_order = df['model_id'].unique()
    category_order = df['category'].unique()
    
    df['model_id'] = pd.Categorical(df['model_id'], categories=model_order, ordered=True)
    df['category'] = pd.Categorical(df['category'], categories=category_order, ordered=True)
    
    pastel_colors = [
        '#8da0cb', '#fc8d62', '#66c2a5', '#e78ac3', '#ffb347', '#b19cd9', '#87cefa', '#90ee90',
        '#ff9999', '#d4a5ff', '#ffcc99', '#c2c2f0', '#ffb6c1', '#c3e6cb', '#ffd700', '#aec6cf',
        '#cb99c9', '#fdfd96', '#cccccc', '#ff6666'
    ]
    colors = pastel_colors[:len(model_order)]
    color_dict = {model: color for model, color in zip(model_order, colors)}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1.2, 0.8]})
    sns.set(style="whitegrid")
    
    # LEFT SUBPLOT: CALIBRATION ERROR BAR CHART
    x = np.arange(len(category_order))
    width = 0.8 / len(model_order)
    
    for i, model in enumerate(model_order):
        model_data = df[df['model_id'] == model]
        model_data = model_data.sort_values('category')
        
        model_x = x - (len(model_order) - 1) * width / 2 + i * width
        
        bars = ax1.bar(
            model_x, model_data['calibration_error'], width, 
            label=model, color=color_dict[model], alpha=0.8
        )
        
        for j, bar in enumerate(bars):
            height = bar.get_height()
            value = model_data['calibration_error'].iloc[j]
            va = 'bottom' if value >= 0 else 'top'
            offset = 0.01 if value >= 0 else -0.01
            ax1.text(
                bar.get_x() + bar.get_width() / 2, height + offset,
                f'{value:.2f}', ha='center', va=va, fontsize=7, rotation=0
            )
    
    ax1.axhline(y=threshold, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(category_order, rotation=30, ha='right')
    ax1.set_xlabel('Category', fontsize=14)
    ax1.set_ylabel('Calibration Error (Confidence - Accuracy)', fontsize=14)
    ax1.set_title(f'Calibration Error by Category and Model - {dataset_name}', fontsize=16)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    ax1.text(
        0.01, 0.02,
        "Positive values = Overconfident\nNegative values = Underconfident\nZero = Perfectly calibrated",
        transform=ax1.transAxes, fontsize=9, va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
    )
    
    # RIGHT SUBPLOT: PVC VS C-PVC SCATTER PLOT
    scatter_points = []
    for i, model in enumerate(model_order):
        if model in pvc_dims and model in cpvc_dims:
            point = ax2.scatter(
                pvc_dims[model], cpvc_dims[model], s=150, 
                color=color_dict[model], label=model,
                edgecolors='black', alpha=0.8
            )
            scatter_points.append((model, pvc_dims[model], cpvc_dims[model], point))
    
    x_vals = [pvc_dims[model] for model in model_order if model in pvc_dims]
    y_vals = [cpvc_dims[model] for model in model_order if model in cpvc_dims]
    
    max_val = max(max(x_vals), max(y_vals)) if x_vals and y_vals else 0
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.7, label='PVC = C-PVC')
    
    for model, x_val, y_val, _ in scatter_points:
        y_offset = 0
        if "Stra" in model or 'Deep' in model or 's1' in model:
            y_offset = -30
        ax2.annotate(
            model, (x_val, y_val), textcoords="offset points", 
            xytext=(0, 10 + y_offset), ha='center', fontsize=10, fontweight='bold'
        )
    
    ax2.set_xlabel('PVC Dimension (γ-shattered categories)', fontsize=14)
    ax2.set_ylabel('Calibration-aware PVC Dimension', fontsize=14)
    ax2.set_title(f'Comparison of PVC and C-PVC Dimensions - {dataset_name}', fontsize=16)
    
    max_dim = max_val
    ax2.set_xlim(-0.5, max_dim + 0.5)
    ax2.set_ylim(-0.5, max_dim + 0.5)
    ax2.set_xticks(range(int(max_dim) + 2))
    ax2.set_yticks(range(int(max_dim) + 2))
    ax2.grid(True, alpha=0.3)
    
    ax2.text(
        0.6, 0.02,
        "Points below line:\nCalibration reduces reliable categories",
        transform=ax2.transAxes, fontsize=9, va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
    )
    
    handles = []
    for model in model_order:
        handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color_dict[model], markersize=10, label=model))
    
    ax2.legend(
        handles=handles, labels=model_order, title='Model',
        loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=10
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    plt.show()

def generate_parameter_sweep_table_sequential(calibration_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Generate a comprehensive table with all gamma-tau combinations.
    Gamma: 0 to 1 in 0.01 steps (101 values)
    Tau: 0 to 1 in 0.01 steps (101 values)
    Total: 101 x 101 = 10,201 combinations
    """
    print("Starting parameter sweep analysis...")
    
    # Generate parameter ranges
    gamma_values = np.arange(0, 1.01, 0.01)
    tau_values = np.arange(0, 1.01, 0.01)
    
    # Get unique models
    models = calibration_df['model_id'].unique()
    
    # Store results
    sweep_results = []
    
    total_combinations = len(gamma_values) * len(tau_values)
    current_combination = 0
    
    for gamma in gamma_values:
        print(f"current gamma = {gamma}")
        for tau in tau_values:
            current_combination += 1
            
            # Progress indicator
            if current_combination % 1000 == 0:
                print(f"Processing combination {current_combination}/{total_combinations} (γ={gamma:.2f}, τ={tau:.2f})")
            
            # Calculate PVC and C-PVC dimensions for this gamma-tau combination
            pvc_dims = calculate_pvc_dimension(calibration_df, gamma)
            cpvc_dims = calculate_cpvc_dimension(calibration_df, gamma, tau)
            
            # Calculate sample complexity for each model
            sample_complexity_df = analyze_sample_complexity(cpvc_dims, gamma, tau)
            
            # Store results for each model
            for model in models:
                sample_complexity_row = sample_complexity_df[
                    sample_complexity_df['model_id'] == model
                ]
                sample_complexity = sample_complexity_row['Sample Complexity'].iloc[0] if not sample_complexity_row.empty else 0
                
                sweep_results.append({
                    'gamma': round(gamma, 2),
                    'tau': round(tau, 2),
                    'model': model,
                    'pvc': pvc_dims.get(model, 0),
                    'c_pvc': cpvc_dims.get(model, 0),
                    'sample_complexity': sample_complexity
                })
    
    print(f"Parameter sweep completed. Generated {len(sweep_results)} rows.")
    return pd.DataFrame(sweep_results)

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import os

def process_gamma_batch(gamma, calibration_df, tau_values, models):
    """Process all tau values for a single gamma value."""
    batch_results = []
    
    # Calculate PVC dimensions once for this gamma (independent of tau)
    pvc_dims = calculate_pvc_dimension(calibration_df, gamma)
    
    for tau in tau_values:
        # Calculate C-PVC dimensions for this gamma-tau combination
        cpvc_dims = calculate_cpvc_dimension(calibration_df, gamma, tau)
        
        # Calculate sample complexity for each model
        sample_complexity_df = analyze_sample_complexity(cpvc_dims, gamma, tau)
        
        # Store results for each model
        for model in models:
            sample_complexity_row = sample_complexity_df[
                sample_complexity_df['model_id'] == model
            ]
            sample_complexity = sample_complexity_row['Sample Complexity'].iloc[0] if not sample_complexity_row.empty else 0
            
            batch_results.append({
                'gamma': round(gamma, 2),
                'tau': round(tau, 2),
                'model': model,
                'pvc': pvc_dims.get(model, 0),
                'c_pvc': cpvc_dims.get(model, 0),
                'sample_complexity': sample_complexity
            })
    
    return batch_results

def generate_parameter_sweep_table_futures(calibration_df: pd.DataFrame, dataset_name: str, max_workers: int = None) -> pd.DataFrame:
    """
    Generate a comprehensive table with all gamma-tau combinations using concurrent.futures.
    Gamma: 0 to 1 in 0.01 steps (101 values)
    Tau: 0 to 1 in 0.01 steps (101 values)
    Total: 101 x 101 = 10,201 combinations
    """
    print("Starting concurrent parameter sweep analysis...")
    
    # Generate parameter ranges
    gamma_values = np.arange(0, 1.01, 0.01)
    tau_values = np.arange(0, 1.01, 0.01)
    
    # Get unique models
    models = calibration_df['model_id'].unique()
    
    # Determine number of workers
    if max_workers is None:
        max_workers = min(os.cpu_count(), len(gamma_values))
    
    print(f"Using {max_workers} workers for {len(gamma_values)} gamma values")
    print(f"Total combinations: {len(gamma_values)} × {len(tau_values)} = {len(gamma_values) * len(tau_values)}")
    
    sweep_results = []
    completed_count = 0
    
    # Use ProcessPoolExecutor for CPU-intensive tasks
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all gamma batches
        future_to_gamma = {
            executor.submit(process_gamma_batch, gamma, calibration_df, tau_values, models): gamma 
            for gamma in gamma_values
        }
        
        # Process completed futures as they finish
        for future in as_completed(future_to_gamma):
            gamma = future_to_gamma[future]
            completed_count += 1
            
            try:
                batch_results = future.result()
                sweep_results.extend(batch_results)
                print(f"Completed gamma={gamma:.2f} ({completed_count}/{len(gamma_values)})")
            except Exception as exc:
                print(f"Gamma {gamma} generated an exception: {exc}")
    
    print(f"Parameter sweep completed. Generated {len(sweep_results)} rows.")
    return pd.DataFrame(sweep_results)

from sklearn.metrics import auc
import os

def calculate_auc_metrics(sweep_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate proper 2D AUC values for PVC, C-PVC, and sample complexity for each model.
    For PVC: AUC over gamma values (1D)
    For C-PVC: AUC over gamma-tau grid (2D using trapezoidal rule)
    """
    print("Calculating 2D AUC metrics from parameter sweep...")
    
    models = sweep_df['model'].unique()
    auc_results = []
    
    for model in models:
        model_data = sweep_df[sweep_df['model'] == model].copy()
        
        # Sort by gamma and tau
        model_data = model_data.sort_values(['gamma', 'tau'])
        
        # For PVC: Calculate AUC over gamma only (since PVC doesn't depend on tau)
        pvc_by_gamma = model_data.groupby('gamma')['pvc'].first().reset_index()
        pvc_auc = np.trapz(pvc_by_gamma['pvc'], pvc_by_gamma['gamma'])
        
        # For C-PVC: Calculate 2D AUC over gamma-tau grid
        gamma_unique = sorted(model_data['gamma'].unique())
        tau_unique = sorted(model_data['tau'].unique())
        
        # Create 2D grid for C-PVC
        cpvc_grid = np.zeros((len(tau_unique), len(gamma_unique)))
        sample_complexity_grid = np.zeros((len(tau_unique), len(gamma_unique)))
        
        for i, tau in enumerate(tau_unique):
            for j, gamma in enumerate(gamma_unique):
                row = model_data[(model_data['gamma'] == gamma) & (model_data['tau'] == tau)]
                if not row.empty:
                    cpvc_grid[i, j] = row['c_pvc'].iloc[0]
                    sample_complexity_grid[i, j] = row['sample_complexity'].iloc[0]
        
        # Calculate 2D AUC using trapezoidal rule
        cpvc_auc = np.trapz(np.trapz(cpvc_grid, tau_unique, axis=0), gamma_unique)
        sample_complexity_auc = np.trapz(np.trapz(sample_complexity_grid, tau_unique, axis=0), gamma_unique)
        
        # Normalize by grid area
        grid_area = (gamma_unique[-1] - gamma_unique[0]) * (tau_unique[-1] - tau_unique[0])
        gamma_range = gamma_unique[-1] - gamma_unique[0]
        
        pvc_auc_normalized = pvc_auc / gamma_range if gamma_range > 0 else 0
        cpvc_auc_normalized = cpvc_auc / grid_area if grid_area > 0 else 0
        sample_complexity_auc_normalized = sample_complexity_auc / grid_area if grid_area > 0 else 0
        
        auc_results.append({
            'model': model,
            'pvc_auc': round(pvc_auc_normalized, 4),
            'cpvc_auc': round(cpvc_auc_normalized, 4),
            'sample_complexity_auc': round(sample_complexity_auc_normalized, 4)
        })
    
    return pd.DataFrame(auc_results)

def create_final_comprehensive_table(comprehensive_table: pd.DataFrame, auc_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the comprehensive table with AUC metrics.
    """
    # Merge on model column
    final_table = pd.merge(
        comprehensive_table, 
        auc_metrics, 
        left_on='Model', 
        right_on='model', 
        how='left'
    )
    
    # Drop the duplicate model column
    final_table = final_table.drop('model', axis=1)
    
    # Reorder columns for better readability
    column_order = [
        'Model', 'PVC', 'C-PVC', 'ECE', 'Brier', 'Sample_Complexity', 'Actual_Error',
        'pvc_auc', 'cpvc_auc', 'sample_complexity_auc'
    ]
    
    final_table = final_table[column_order]
    
    return final_table

def abbreviate_model_name(model_name):
    """Abbreviate model names for better visualization."""
    abbreviations = {
        'Qwen2.5-7B': 'Qwen7',
        'Qwen2.5-7B-Instruct': 'Qwen70I',
        'Qwen2.5-Math-7B-Instruct': 'Q7M',
        'Llama-3.1-8B-Instruct': 'L8I',
        'OpenThinker2-7B': 'OT7',
        'DeepSeek-R1-Distill-Qwen-7B': 'DS7',
        'Bespoke-Stratos-7B': 'BS7',
        'JiuZhang3.0-7B': 'JZ7',
        'Ministral-8B-Instruct-2410': 'M8I',
        'Open-Reasoner-Zero-7B': 'OR7',
        's1.1-7B': 'S7'
    }
    return abbreviations.get(model_name, model_name[:6])  # Fallback to first 6 chars

def plot_auc_pvc_scatter(auc_metrics, dataset_name, output_file=None):
    """
    Create a scatter plot comparing AUC_PVC and AUC_CPVC values.
    """
    pastel_colors = [
        '#8da0cb', '#fc8d62', '#66c2a5', '#e78ac3', '#ffb347', '#b19cd9', '#87cefa', '#90ee90',
        '#ff9999', '#d4a5ff', '#ffcc99', '#c2c2f0', '#ffb6c1', '#c3e6cb', '#ffd700', '#aec6cf',
        '#cb99c9', '#fdfd96', '#cccccc', '#ff6666'
    ]
    
    models = auc_metrics['model'].tolist()
    colors = pastel_colors[:len(models)]
    color_dict = {model: color for model, color in zip(models, colors)}
    
    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")
    
    # Create scatter plot
    scatter_points = []
    for _, row in auc_metrics.iterrows():
        model = row['model']
        pvc_auc = row['pvc_auc']
        cpvc_auc = row['cpvc_auc']
        
        plt.scatter(
            pvc_auc, cpvc_auc, s=150, 
            color=color_dict[model],
            edgecolors='black', alpha=0.8
        )
        scatter_points.append((model, pvc_auc, cpvc_auc))
    
    # Get axis ranges
    x_vals = auc_metrics['pvc_auc'].values
    y_vals = auc_metrics['cpvc_auc'].values
    
    if len(x_vals) > 0 and len(y_vals) > 0:
        # X-axis: Round PVC values to nearest 0.5
        import math
        min_x = min(x_vals)
        max_x = max(x_vals)
        x_min = math.floor(min_x * 2) / 2  # Round down to nearest 0.5
        x_max = math.ceil(max_x * 2) / 2   # Round up to nearest 0.5
        
        # Y-axis: Use original method (min/max with padding)
        min_val = min(min(x_vals), min(y_vals))
        max_val = max(max(x_vals), max(y_vals))
        
        # Add some padding
        padding = (max_val - min_val) * 0.1
        min_val -= padding
        max_val += padding
        
        # Set axis limits
        plt.xlim(x_min, x_max)
        plt.ylim(min_val, max_val)
        
        # Identity line
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
    
    # Add model labels
    for model, x_val, y_val in scatter_points:
        y_offset = 0
        if "Stra" in model or 'Deep' in model or 's1' in model:
            y_offset = -0.02
        plt.annotate(
            model, (x_val, y_val), textcoords="offset points", 
            xytext=(0, 10 + y_offset*100), ha='center', fontsize=10, fontweight='bold'
        )
    
    plt.xlabel('PVC-AUC', fontsize=14)
    plt.ylabel('Calibration-aware PVC-AUC', fontsize=14)
    plt.title(f'PVC-AUC vs C-PVC-AUC - {dataset_name}', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    plt.text(
        0.6, 0.02,
        "Points below line:\nCalibration reduces AUC reliability",
        transform=plt.gca().transAxes, fontsize=9, va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
    )
    
    # Create legend manually
    legend_elements = []
    for model in models:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=color_dict[model], markersize=10, label=model))
    
    plt.legend(handles=legend_elements, title='Model', loc='upper left', 
              frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"AUC scatter plot saved to {output_file}")
    
    plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

from matplotlib.colors import LinearSegmentedColormap

def plot_all_models_cpvc_3d_grid(parameter_sweep_table, dataset_name, output_file=None):
    """
    Create a 4x3 grid of 3D surface plots for all models with legend in the last subplot.
    """
    # Use same pastel colors as in other plots
    pastel_colors = [
        '#8da0cb', '#fc8d62', '#66c2a5', '#e78ac3', '#ffb347', '#b19cd9', '#87cefa', '#90ee90',
        '#ff9999', '#d4a5ff', '#ffcc99', '#c2c2f0', '#ffb6c1', '#c3e6cb', '#ffd700', '#aec6cf',
        '#cb99c9', '#fdfd96', '#cccccc', '#ff6666'
    ]
    
    models = parameter_sweep_table['model'].unique()
    colors = pastel_colors[:len(models)]
    color_dict = {model: color for model, color in zip(models, colors)}
    
    # Create 4x3 subplot grid
    fig = plt.figure(figsize=(20, 12))
    
    for idx, model in enumerate(models):
        print(f"Processing model {idx+1}/{len(models)}: {model}")
        
        model_data = parameter_sweep_table[parameter_sweep_table['model'] == model].copy()
        
        if model_data.empty:
            continue
        
        # Create meshgrid for gamma and tau
        gamma_unique = sorted(model_data['gamma'].unique())
        tau_unique = sorted(model_data['tau'].unique())
        
        gamma_grid, tau_grid = np.meshgrid(gamma_unique, tau_unique)
        cpvc_grid = np.zeros_like(gamma_grid)
        
        # Fill the C-PVC grid
        for i, tau in enumerate(tau_unique):
            for j, gamma in enumerate(gamma_unique):
                row = model_data[(model_data['gamma'] == gamma) & (model_data['tau'] == tau)]
                if not row.empty:
                    cpvc_grid[i, j] = row['c_pvc'].iloc[0]
        
        # Create custom colormap based on model's assigned color
        model_color = color_dict[model]
        colors_list = ['white', model_color, '#404040']
        custom_cmap = LinearSegmentedColormap.from_list(f'{model}_cmap', colors_list, N=256)
        
        # Create subplot
        ax = fig.add_subplot(3, 4, idx + 1, projection='3d')
        
        # Create surface plot
        surf = ax.plot_surface(gamma_grid, tau_grid, cpvc_grid, 
                              cmap=custom_cmap, alpha=0.9,
                              linewidth=0, antialiased=True,
                              shade=True, rcount=40, ccount=40)
        
        # Customize each subplot
        ax.set_xlabel('γ', fontsize=10)
        ax.set_ylabel('τ', fontsize=10)
        ax.set_zlabel('C-PVC', fontsize=10)
        ax.set_title(model, fontsize=12, pad=10, fontweight='bold')
        
        # Set viewing angle
        ax.view_init(elev=30, azim=-45)
        
        # Smaller tick labels
        ax.tick_params(labelsize=8)
    
    # Create legend in the last subplot (position 12) - perfectly centered
    if len(models) < 12:
        ax_legend = fig.add_subplot(3, 4, 12)
        ax_legend.axis('off')
        
        # Create legend elements
        legend_elements = []
        for model in models:
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, 
                                            facecolor=color_dict[model], 
                                            edgecolor='black', 
                                            label=model))
        
        # Center the legend perfectly in the subplot
        legend = ax_legend.legend(handles=legend_elements, 
                                loc='center', 
                                fontsize=10, 
                                title='Models', 
                                title_fontsize=12,
                                frameon=True, 
                                fancybox=True, 
                                shadow=True,
                                bbox_to_anchor=(0.5, 0.5))
        
        # Set the subplot limits to ensure centering
        ax_legend.set_xlim(0, 1)
        ax_legend.set_ylim(0, 1)
    
    # plt.suptitle(f'C-PVC Surfaces for All Models - {dataset_name}', fontsize=16, y=0.95, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Grid plot saved to {output_file}")
    
    plt.show()

def plot_model_cpvc_3d_surfaces(parameter_sweep_table, dataset_name, output_dir="3d_cpvc_plots"):
    """
    Create individual 3D surface plots for all models showing C-PVC changes across gamma-tau grid.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Use same pastel colors as in other plots
    pastel_colors = [
        '#8da0cb', '#fc8d62', '#66c2a5', '#e78ac3', '#ffb347', '#b19cd9', '#87cefa', '#90ee90',
        '#ff9999', '#d4a5ff', '#ffcc99', '#c2c2f0', '#ffb6c1', '#c3e6cb', '#ffd700', '#aec6cf',
        '#cb99c9', '#fdfd96', '#cccccc', '#ff6666'
    ]
    
    models = parameter_sweep_table['model'].unique()
    colors = pastel_colors[:len(models)]
    color_dict = {model: color for model, color in zip(models, colors)}
    
    for model in models:
        print(f"Creating individual 3D C-PVC surface plot for {model}...")
        
        model_data = parameter_sweep_table[parameter_sweep_table['model'] == model].copy()
        
        if model_data.empty:
            print(f"No data found for model: {model}")
            continue
        
        # Create meshgrid for gamma and tau
        gamma_unique = sorted(model_data['gamma'].unique())
        tau_unique = sorted(model_data['tau'].unique())
        
        gamma_grid, tau_grid = np.meshgrid(gamma_unique, tau_unique)
        cpvc_grid = np.zeros_like(gamma_grid)
        
        # Fill the C-PVC grid
        for i, tau in enumerate(tau_unique):
            for j, gamma in enumerate(gamma_unique):
                row = model_data[(model_data['gamma'] == gamma) & (model_data['tau'] == tau)]
                if not row.empty:
                    cpvc_grid[i, j] = row['c_pvc'].iloc[0]
        
        # Create custom colormap based on model's assigned color
        model_color = color_dict[model]
        # Convert hex to RGB and create darker version of the same color
        import matplotlib.colors as mcolors
        rgb = mcolors.hex2color(model_color)
        # Create darker version by reducing brightness by 40%
        darker_rgb = tuple(c * 0.6 for c in rgb)
        darker_color = mcolors.rgb2hex(darker_rgb)

        # Create gradient from white to model color to darker version of same color
        colors_list = ['white', model_color, darker_color]
        custom_cmap = LinearSegmentedColormap.from_list(f'{model}_cmap', colors_list, N=256)
        
        # Create 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create surface plot with custom colormap and enhanced shading
        surf = ax.plot_surface(gamma_grid, tau_grid, cpvc_grid, 
                              cmap=custom_cmap, alpha=0.9,
                              linewidth=0, antialiased=True,
                              shade=True, lightsource=None,
                              rcount=80, ccount=80)
        
        # Add contour lines on the bottom with same color theme
        ax.contour(gamma_grid, tau_grid, cpvc_grid, zdir='z', 
                  offset=cpvc_grid.min()-0.5, cmap=custom_cmap, alpha=0.6)
        
        # Customize the plot with better spacing
        ax.set_xlabel('Gamma (γ)', fontsize=12, labelpad=10)
        ax.set_ylabel('Tau (τ)', fontsize=12, labelpad=10)
        ax.set_zlabel('C-PVC Dimension', fontsize=12, labelpad=10)
        ax.set_title(f'{model} - C-PVC Surface ({dataset_name})', fontsize=14, pad=20)
        
        # Add colorbar
        fig.colorbar(surf, shrink=0.5, aspect=20, pad=0.1)
        
        # Set viewing angle
        ax.view_init(elev=30, azim=-45)
        
        # Adjust layout to prevent label cutoff
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        # Save plot
        clean_model_name = model.replace('/', '_').replace(' ', '_').replace('.', '_')
        output_file = os.path.join(output_dir, f"{dataset_name}_{clean_model_name}_cpvc_3d.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.close()
        
        print(f"Saved: {output_file}")
    
    # Create grid plot
    print("Creating 4x3 grid plot...")
    grid_output_file = os.path.join(output_dir, f"{dataset_name}_all_models_cpvc_3d_grid.png")
    plot_all_models_cpvc_3d_grid(parameter_sweep_table, dataset_name, grid_output_file)
    
    print(f"All 3D C-PVC plots saved to: {output_dir}")


# Update the main execution code
if __name__ == "__main__":
    # Parameters for single analysis
    gamma = 0.6
    tau = 0.25
    
    # Create filename suffix with parameters
    param_suffix = f"gamma{gamma}_tau{tau}"
    
    # Basic analysis
    combined_category_accuracy_plots(
        data, dataset_name, 
        f"{dataset_name}_combined_model_performance_{param_suffix}.png", 
        threshold=0.6
    )
    
    calibration_df = plot_category_calibration_error(
        data, dataset_name,
        f"{dataset_name}_category_calibration_error_{param_suffix}.png"
    )
    calibration_df.to_csv(f"{dataset_name}_calibration_error_{param_suffix}.csv", index=False)
    
    # Judge correlation analysis
    by_group_df = calculate_judge_correlations_by_model_category(data)
    overall_summary = get_overall_summary(data)
    
    print("Results by Model and Category:")
    print(by_group_df)
    print("\nOverall Summary:")
    print(overall_summary)
    
    # PVC analysis for single gamma-tau combination
    results = run_comprehensive_analysis(calibration_df, gamma, tau)
    
    # Create comprehensive table for single combination
    comprehensive_table = create_comprehensive_table(results, calibration_df, gamma, tau)
    comprehensive_table.to_csv(f"{dataset_name}_comprehensive_metrics_table_{param_suffix}.csv", index=False)
    
    print("\n==== Comprehensive Metrics Table ====")
    print(comprehensive_table.round(4))
    
    # Final combined plot
    combined_calibration_pvc_plot(
        df=calibration_df,
        pvc_dims=results['pvc_dimensions'],
        cpvc_dims=results['cpvc_dimensions'],
        dataset_name=dataset_name,
        output_file=f"{dataset_name}_combined_calibration_pvc_plot_{param_suffix}.png"
    )
    
    print(f"\nComprehensive table saved to: {dataset_name}_comprehensive_metrics_table_{param_suffix}.csv")
    
    # Parameter sweep analysis
    print("\n" + "="*50)
    print("PARAMETER SWEEP ANALYSIS")
    print("="*50)
    
    sweep_filename = f"{dataset_name}_parameter_sweep_table.csv"
    
    # Check if sweep table already exists
    if os.path.exists(sweep_filename):
        print(f"Found existing sweep table: {sweep_filename}")
        print("Loading existing parameter sweep table...")
        parameter_sweep_table = pd.read_csv(sweep_filename)
        print(f"Loaded {len(parameter_sweep_table)} rows from existing file.")
    else:
        print("No existing sweep table found. Generating new parameter sweep...")
        parameter_sweep_table = generate_parameter_sweep_table_futures(calibration_df, dataset_name)
        parameter_sweep_table.to_csv(sweep_filename, index=False)
        print(f"Parameter sweep table saved to: {sweep_filename}")
    
    # Sort the sweep table by gamma and tau
    print("Sorting parameter sweep table by gamma and tau...")
    parameter_sweep_table = parameter_sweep_table.sort_values(['gamma', 'tau']).reset_index(drop=True)
    
    # Calculate AUC metrics
    auc_metrics = calculate_auc_metrics(parameter_sweep_table)
    
    print("\n==== AUC Metrics ====")
    print(auc_metrics)
    
    # Create final comprehensive table with AUC metrics
    final_comprehensive_table = create_final_comprehensive_table(comprehensive_table, auc_metrics)
    
    # Save final table
    final_filename = f"{dataset_name}_final_comprehensive_table_{param_suffix}.csv"
    final_comprehensive_table.to_csv(final_filename, index=False)
    
    print("\n==== Final Comprehensive Table with AUC Metrics ====")
    print(final_comprehensive_table)
    
    # Create AUC PVC vs AUC C-PVC scatter plot
    print("Generating AUC PVC vs AUC C-PVC scatter plot...")
    plot_auc_pvc_scatter(
        auc_metrics=auc_metrics,
        dataset_name=dataset_name,
        output_file=f"{dataset_name}_auc_pvc_scatter_{param_suffix}.png"
    )
    
    print(f"\nFinal comprehensive table saved to: {final_filename}")
    print(f"Parameter sweep summary:")
    print(f"  - Total combinations: {len(parameter_sweep_table)}")
    print(f"  - Unique gamma values: {parameter_sweep_table['gamma'].nunique()}")
    print(f"  - Unique tau values: {parameter_sweep_table['tau'].nunique()}")
    print(f"  - Models included: {parameter_sweep_table['model'].nunique()}")

    # After parameter sweep analysis
    if os.path.exists(sweep_filename):
        print("\n" + "="*50)
        print("GENERATING 3D C-PVC SURFACE PLOTS")
        print("="*50)
        
        # Create 3D C-PVC surface plots for all models
        plot_model_cpvc_3d_surfaces(parameter_sweep_table, dataset_name)
    
    print("\nAnalysis Complete!")