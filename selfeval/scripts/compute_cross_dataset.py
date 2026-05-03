# cross_dataset_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import os

# NumPy ≥2.0 renamed ``np.trapz`` → ``np.trapezoid``.
_trapz = getattr(np, "trapezoid", None) or np.trapz

# Define the desired legend order
LEGEND_ORDER = [
    'Qwen2.5-7B',
    'Qwen2.5-7B-Instruct', 
    'Qwen2.5-Math-7B-Instruct',
    'Llama-3.1-8B-Instruct',
    'OpenThinker2-7B',
    'DeepSeek-R1-Distill-Qwen-7B',
    'Bespoke-Stratos-7B',
    'JiuZhang3.0-7B',
    'Ministral-8B-Instruct-2410',
    'Open-Reasoner-Zero-7B',
    's1.1-7B'
]

def get_ordered_models(models):
    """Return models in the specified legend order."""
    available_models = set(models)
    ordered_models = [model for model in LEGEND_ORDER if model in available_models]
    # Add any remaining models not in the predefined order
    remaining_models = [model for model in models if model not in LEGEND_ORDER]
    return ordered_models + remaining_models

def load_sweep_files():
    """Load sweep files from all three datasets."""
    datasets = {
        'Math360': 'mathbenchmark_parameter_sweep_table.csv',
        'TruthfulQA': 'truthfulQA_parameter_sweep_table.csv', 
        'CSQA': 'CSQA_parameter_sweep_table.csv'
    }
    
    sweep_data = {}
    
    for dataset_name, filename in datasets.items():
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            sweep_data[dataset_name] = df
            print(f"Loaded {len(df)} rows from {filename}")
        else:
            print(f"Warning: {filename} not found")
    
    return sweep_data

def calculate_cross_dataset_averages_fast(sweep_data):
    """Fast calculation using pandas groupby to average across datasets."""
    print("Combining all datasets...")
    
    # Combine all dataframes
    all_dfs = []
    for dataset_name, df in sweep_data.items():
        df_copy = df.copy()
        df_copy['dataset'] = dataset_name
        all_dfs.append(df_copy)
        print(f"{dataset_name}: {len(df_copy)} rows, models: {df_copy['model'].unique()}")
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined dataframe has {len(combined_df)} rows")
    
    # Check a specific example before averaging
    print("\nExample before averaging (gamma=0.6, tau=0.25, first model):")
    example_model = combined_df['model'].iloc[0]
    example_data = combined_df[(combined_df['gamma'] == 0.6) & 
                              (combined_df['tau'] == 0.25) & 
                              (combined_df['model'] == example_model)]
    print(example_data[['dataset', 'model', 'gamma', 'tau', 'pvc', 'c_pvc']])
    
    # Group by model, gamma, tau and calculate averages
    print("\nCalculating averages...")
    averaged_df = combined_df.groupby(['model', 'gamma', 'tau']).agg({
        'pvc': 'mean',
        'c_pvc': 'mean', 
        'sample_complexity': 'mean',
        'dataset': 'count'  # Count how many datasets contributed
    }).reset_index()
    
    # Rename count column
    averaged_df = averaged_df.rename(columns={'dataset': 'num_datasets'})
    
    # Check the same example after averaging
    print("\nExample after averaging:")
    example_avg = averaged_df[(averaged_df['gamma'] == 0.6) & 
                             (averaged_df['tau'] == 0.25) & 
                             (averaged_df['model'] == example_model)]
    print(example_avg[['model', 'gamma', 'tau', 'pvc', 'c_pvc', 'num_datasets']])
    
    # Only keep combinations with data from at least 2 datasets
    before_filter = len(averaged_df)
    averaged_df = averaged_df[averaged_df['num_datasets'] >= 2]
    after_filter = len(averaged_df)
    
    print(f"\nFiltered from {before_filter} to {after_filter} rows (kept only >= 2 datasets)")
    print(f"Generated {len(averaged_df)} averaged data points")
    print(f"Models: {averaged_df['model'].nunique()}")
    print(f"Gamma-Tau combinations: {len(averaged_df.groupby(['gamma', 'tau']))}")
    
    # Show distribution of num_datasets
    print(f"Dataset count distribution:")
    print(averaged_df['num_datasets'].value_counts().sort_index())
    
    return averaged_df

def plot_averaged_pvc_line_plot(averaged_df, output_file=None):
    """Create PVC line plot similar to Math_combined_model_performance format."""
    # Use same pastel colors
    pastel_colors = [
        '#8da0cb', '#fc8d62', '#66c2a5', '#e78ac3', '#ffb347', '#b19cd9', '#87cefa', '#90ee90',
        '#ff9999', '#d4a5ff', '#ffcc99', '#c2c2f0', '#ffb6c1', '#c3e6cb', '#ffd700', '#aec6cf',
        '#cb99c9', '#fdfd96', '#cccccc', '#ff6666'
    ]
    
    # Get models in the specified order
    models = get_ordered_models(averaged_df['model'].unique())
    colors = pastel_colors[:len(models)]
    color_dict = {model: color for model, color in zip(models, colors)}
    
    # Get PVC values by gamma (PVC doesn't depend on tau)
    pvc_by_gamma = averaged_df.groupby(['model', 'gamma'])['pvc'].first().reset_index()
    
    # Calculate gamma threshold values
    gamma_values = np.arange(0.3, 0.9, 0.01)
    results = {model: [] for model in models}
    
    # Get maximum PVC value to determine scale
    max_pvc = averaged_df['pvc'].max()
    
    for gamma_threshold in gamma_values:
        for model in models:
            # Get PVC value at this exact gamma
            model_pvc_data = pvc_by_gamma[pvc_by_gamma['model'] == model]
            
            if not model_pvc_data.empty:
                # Find the PVC value at the closest gamma <= threshold
                valid_gammas = model_pvc_data[model_pvc_data['gamma'] <= gamma_threshold]
                
                if not valid_gammas.empty:
                    # Get PVC at the highest gamma <= threshold (keep as float)
                    pvc_value = valid_gammas.loc[valid_gammas['gamma'].idxmax(), 'pvc']
                else:
                    pvc_value = 0.0
            else:
                pvc_value = 0.0
            
            results[model].append(pvc_value)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', 
                  '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':']
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 
              '+', 'x', 'd', '|', '_', '1', '2', '3', '4', 'P']
    
    for i, model in enumerate(models):
        ax.plot(
            gamma_values, results[model], label=model, color=color_dict[model],
            linewidth=2.5, linestyle=line_styles[i % len(line_styles)],
            marker=markers[i % len(markers)], markersize=5,
            markevery=10
        )
    
    ax.axhline(y=max_pvc, color='gray', linestyle=':', linewidth=1.5, 
               alpha=0.7, label=f'Max PVC ({max_pvc:.1f})')
    
    # ax.set_title('Cross-Dataset Average PVC', fontsize=16, fontweight='bold')
    ax.set_xlabel('Threshold (γ)', fontsize=14)
    ax.set_ylabel('Cross-Dataset Average PVC', fontsize=14)
    ax.set_xticks(np.arange(0.3, 0.9, 0.1))
    
    # Set Y-axis to show decimal values with smart limits
    y_min = 0
    y_max = max_pvc
    y_max_rounded = np.ceil(y_max * 2) / 2  # Round up to nearest 0.5

    ax.set_ylim(y_min, y_max_rounded)
    ax.set_yticks(np.arange(0, y_max_rounded + 0.5, 0.5))
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title='Model', loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Cross-dataset PVC plot saved to {output_file}")
    
    plt.show()

# Add this dictionary after the LEGEND_ORDER definition
CPVC_AVERAGES = {
    'Qwen2.5-7B': 3.76,
    'Qwen2.5-7B-Instruct': 4.09,
    'Qwen2.5-Math-7B-Instruct': 3.20,
    'Llama-3.1-8B-Instruct': 2.92,
    'OpenThinker2-7B': 3.50,
    'DeepSeek-R1-Distill-Qwen-7B': 3.49,
    'Bespoke-Stratos-7B': 3.54,
    'JiuZhang3.0-7B': 3.84,
    'Ministral-8B-Instruct-2410': 3.00,
    'Open-Reasoner-Zero-7B': 3.50,
    's1.1-7B': 4.27
}

def get_model_label_with_cpvc(model):
    """Return model name with C-PVC average in parentheses."""
    cpvc_avg = CPVC_AVERAGES.get(model, 0.0)
    return f"{model} ({cpvc_avg:.2f})"

def plot_averaged_cpvc_3d_grid(averaged_df, output_file=None):
    """Create 3D grid plot similar to CSQA_all_models_cpvc_3d_grid format."""
    # Use same pastel colors
    pastel_colors = [
        '#8da0cb', '#fc8d62', '#66c2a5', '#e78ac3', '#ffb347', '#b19cd9', '#87cefa', '#90ee90',
        '#ff9999', '#d4a5ff', '#ffcc99', '#c2c2f0', '#ffb6c1', '#c3e6cb', '#ffd700', '#aec6cf',
        '#cb99c9', '#fdfd96', '#cccccc', '#ff6666'
    ]
    
    # Get models in the specified order
    models = get_ordered_models(averaged_df['model'].unique())
    colors = pastel_colors[:len(models)]
    color_dict = {model: color for model, color in zip(models, colors)}
    
    # Create 4x3 subplot grid
    fig = plt.figure(figsize=(20, 12))
    
    for idx, model in enumerate(models):
        print(f"Processing model {idx+1}/{len(models)}: {model}")
        
        model_data = averaged_df[averaged_df['model'] == model].copy()
        
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
        import matplotlib.colors as mcolors
        rgb = mcolors.hex2color(model_color)
        darker_rgb = tuple(c * 0.6 for c in rgb)
        darker_color = mcolors.rgb2hex(darker_rgb)
        
        colors_list = ['white', model_color, darker_color]
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
        ax.set_title(get_model_label_with_cpvc(model), fontsize=12, pad=10, fontweight='bold')
        
        # Set viewing angle
        ax.view_init(elev=30, azim=-45)
        
        # Smaller tick labels
        ax.tick_params(labelsize=8)
    
    # Create legend in the last subplot
    if len(models) < 12:
        ax_legend = fig.add_subplot(3, 4, 12)
        ax_legend.axis('off')
        
        # Create legend elements in the specified order
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
        
        legend.get_title().set_fontweight('bold')
        ax_legend.set_xlim(0, 1)
        ax_legend.set_ylim(0, 1)
    
    # plt.suptitle('Cross-Dataset Average C-PVC Surfaces for All Models', fontsize=18, y=0.95, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Cross-dataset 3D grid plot saved to {output_file}")
    
    plt.show()

def plot_combined_pvc_cpvc(averaged_df, output_file=None):
    """Create combined PVC line plot (left) and C-PVC 3D grid plot (right)."""
    import matplotlib.gridspec as gridspec
    
    # Use same pastel colors
    pastel_colors = [
        '#8da0cb', '#fc8d62', '#66c2a5', '#e78ac3', '#ffb347', '#b19cd9', '#87cefa', '#90ee90',
        '#ff9999', '#d4a5ff', '#ffcc99', '#c2c2f0', '#ffb6c1', '#c3e6cb', '#ffd700', '#aec6cf',
        '#cb99c9', '#fdfd96', '#cccccc', '#ff6666'
    ]
    
    # Get models in the specified order
    models = get_ordered_models(averaged_df['model'].unique())
    colors = pastel_colors[:len(models)]
    color_dict = {model: color for model, color in zip(models, colors)}
    
    # Create figure with custom grid layout: 4 rows, 4 columns
    fig = plt.figure(figsize=(28, 16))
    gs = gridspec.GridSpec(4, 4, width_ratios=[1.2, 1, 1, 1])
    
    # LEFT SUBPLOT: PVC LINE PLOT (spans 4 rows, 1 column)
    ax1 = fig.add_subplot(gs[:, 0])
    
    # Get PVC values by gamma
    pvc_by_gamma = averaged_df.groupby(['model', 'gamma'])['pvc'].first().reset_index()
    gamma_values = np.arange(0.3, 1.01, 0.01)
    results = {model: [] for model in models}
    max_pvc = averaged_df['pvc'].max()
    
    for gamma_threshold in gamma_values:
        for model in models:
            model_pvc_data = pvc_by_gamma[pvc_by_gamma['model'] == model]
            
            if not model_pvc_data.empty:
                valid_gammas = model_pvc_data[model_pvc_data['gamma'] <= gamma_threshold]
                if not valid_gammas.empty:
                    pvc_value = valid_gammas.loc[valid_gammas['gamma'].idxmax(), 'pvc']
                else:
                    pvc_value = 0.0
            else:
                pvc_value = 0.0
            
            results[model].append(pvc_value)
    
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
    
    for i, model in enumerate(models):
        ax1.plot(
            gamma_values, results[model], color=color_dict[model],
            linewidth=2.5, linestyle=line_styles[i % len(line_styles)],
            marker=markers[i % len(markers)], markersize=5, markevery=10
        )
    
    y_max_rounded = np.ceil(max_pvc * 2) / 2
    ax1.set_ylim(0, y_max_rounded)
    ax1.set_yticks(np.arange(0, y_max_rounded + 0.5, 0.5))
    
    ax1.set_title('Cross-Dataset Average PVC - γ', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Threshold (γ)', fontsize=14)
    ax1.set_ylabel('PVC', fontsize=14)
    ax1.set_xticks(np.arange(0.3, 1.1, 0.1))
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # RIGHT SUBPLOTS: 4x3 grid of C-PVC 3D surfaces
    for idx, model in enumerate(models):
        if idx >= 11:  # Only 11 models, save last spot for legend
            break
            
        row = idx // 3
        col = idx % 3 + 1  # +1 because first column is PVC plot
        
        ax = fig.add_subplot(gs[row, col], projection='3d')
        
        model_data = averaged_df[averaged_df['model'] == model].copy()
        
        if not model_data.empty:
            gamma_unique = sorted(model_data['gamma'].unique())
            tau_unique = sorted(model_data['tau'].unique())
            
            gamma_grid, tau_grid = np.meshgrid(gamma_unique, tau_unique)
            cpvc_grid = np.zeros_like(gamma_grid)
            
            for i, tau in enumerate(tau_unique):
                for j, gamma in enumerate(gamma_unique):
                    row_data = model_data[(model_data['gamma'] == gamma) & (model_data['tau'] == tau)]
                    if not row_data.empty:
                        cpvc_grid[i, j] = row_data['c_pvc'].iloc[0]
            
            # Create custom colormap
            model_color = color_dict[model]
            import matplotlib.colors as mcolors
            rgb = mcolors.hex2color(model_color)
            darker_rgb = tuple(c * 0.6 for c in rgb)
            darker_color = mcolors.rgb2hex(darker_rgb)
            
            colors_list = ['white', model_color, darker_color]
            custom_cmap = LinearSegmentedColormap.from_list(f'{model}_cmap', colors_list, N=256)
            
            surf = ax.plot_surface(gamma_grid, tau_grid, cpvc_grid, 
                                  cmap=custom_cmap, alpha=0.9,
                                  linewidth=0, antialiased=True,
                                  shade=True, rcount=30, ccount=30)
            
            ax.set_xlabel('γ', fontsize=8)
            ax.set_ylabel('τ', fontsize=8)
            ax.set_zlabel('C-PVC', fontsize=8)
            ax.set_title(model, fontsize=10, pad=5, fontweight='bold')
            ax.view_init(elev=30, azim=-45)
            ax.tick_params(labelsize=6)
    
    # Legend in the last position (4th row, 4th column)
    if len(models) < 12:
        ax_legend = fig.add_subplot(gs[3, 3])  # Bottom right position
        ax_legend.axis('off')
        
        legend_elements = []
        for model in models:
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, 
                                               facecolor=color_dict[model], 
                                               edgecolor='black', linewidth=1,
                                               label=model))
        
        legend = ax_legend.legend(handles=legend_elements, 
                                 loc='center', 
                                 fontsize=8, 
                                 title='Models', 
                                 title_fontsize=9,
                                 frameon=True, 
                                 fancybox=True, 
                                 shadow=True)
        
        legend.get_title().set_fontweight('bold')
    
    plt.suptitle('Cross-Dataset Analysis: PVC and C-PVC Comparison', fontsize=18, y=0.95, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to {output_file}")
    
    plt.show()

def calculate_auc_metrics(sweep_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate proper 2D AUC values for PVC, C-PVC, and sample complexity for each model."""
    print("Calculating 2D AUC metrics from parameter sweep...")
    
    models = sweep_df['model'].unique()
    auc_results = []
    
    for model in models:
        model_data = sweep_df[sweep_df['model'] == model].copy()
        
        # Sort by gamma and tau
        model_data = model_data.sort_values(['gamma', 'tau'])
        
        # For PVC: Calculate AUC over gamma only (since PVC doesn't depend on tau)
        pvc_by_gamma = model_data.groupby('gamma')['pvc'].first().reset_index()
        pvc_auc = _trapz(pvc_by_gamma['pvc'], pvc_by_gamma['gamma'])
        
        # For C-PVC: Calculate 2D AUC over gamma-tau grid
        gamma_unique = sorted(model_data['gamma'].unique())
        tau_unique = sorted(model_data['tau'].unique())
        
        # Create 2D grid for C-PVC
        cpvc_grid = np.zeros((len(tau_unique), len(gamma_unique)))
        
        for i, tau in enumerate(tau_unique):
            for j, gamma in enumerate(gamma_unique):
                row = model_data[(model_data['gamma'] == gamma) & (model_data['tau'] == tau)]
                if not row.empty:
                    cpvc_grid[i, j] = row['c_pvc'].iloc[0]
        
        # Calculate 2D AUC using trapezoidal rule
        cpvc_auc = _trapz(_trapz(cpvc_grid, tau_unique, axis=0), gamma_unique)
        
        # Normalize by grid area
        grid_area = (gamma_unique[-1] - gamma_unique[0]) * (tau_unique[-1] - tau_unique[0])
        gamma_range = gamma_unique[-1] - gamma_unique[0]
        
        pvc_auc_normalized = pvc_auc / gamma_range if gamma_range > 0 else 0
        cpvc_auc_normalized = cpvc_auc / grid_area if grid_area > 0 else 0
        
        auc_results.append({
            'model': model,
            'pvc_auc': round(pvc_auc_normalized, 4),
            'cpvc_auc': round(cpvc_auc_normalized, 4)
        })
    
    return pd.DataFrame(auc_results)

def plot_cross_domain_pvc_cpvc_comparison(averaged_df, output_file=None):
    """Create comparison PVC-VUS and C-PVC-VUS plot for cross-domain averages."""
    LEGEND_ORDER = [
        'Qwen2.5-7B', 'Qwen2.5-7B-Instruct', 'Qwen2.5-Math-7B-Instruct',
        'Llama-3.1-8B-Instruct', 'OpenThinker2-7B', 'DeepSeek-R1-Distill-Qwen-7B',
        'Bespoke-Stratos-7B', 'JiuZhang3.0-7B', 'Ministral-8B-Instruct-2410',
        'Open-Reasoner-Zero-7B', 's1.1-7B'
    ]
    
    available_models = set(averaged_df['model'].unique())
    models = [model for model in LEGEND_ORDER if model in available_models]
    
    pastel_colors = [
        '#8da0cb', '#fc8d62', '#66c2a5', '#e78ac3', '#ffb347', '#b19cd9', '#87cefa', '#90ee90',
        '#ff9999', '#d4a5ff', '#ffcc99', '#c2c2f0', '#ffb6c1', '#c3e6cb', '#ffd700', '#aec6cf',
        '#cb99c9', '#fdfd96', '#cccccc', '#ff6666'
    ]
    color_dict = {model: pastel_colors[i] for i, model in enumerate(models)}
    
    # Calculate VUS metrics
    auc_metrics = calculate_auc_metrics(averaged_df)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter_points = []
    for model in models:
        model_auc = auc_metrics[auc_metrics['model'] == model]
        if not model_auc.empty:
            pvc_auc = model_auc['pvc_auc'].iloc[0]
            cpvc_auc = model_auc['cpvc_auc'].iloc[0]
            
            ax.scatter(
                pvc_auc, cpvc_auc, s=150,
                color=color_dict[model],
                edgecolors='black', alpha=0.8,
                label=f"{model}"
            )
            scatter_points.append((model, pvc_auc, cpvc_auc))
    
    x_vals = [point[1] for point in scatter_points]
    y_vals = [point[2] for point in scatter_points]
    
    if x_vals and y_vals:
        min_val = min(min(x_vals), min(y_vals))
        max_val = max(max(x_vals), max(y_vals))
        
        # ax.set_xlim(min_val - 0.1, max_val + 0.1)
        ax.set_xlim(4.15, 6.1)#max_val + 0.1)
        # ax.set_ylim(min_val - 0.1, max_val + 0.1)
        ax.set_ylim(2.6, 4.6)
        
        # Identity line
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='PVC-VUS = C-PVC-VUS')
    
    for model, x_val, y_val in scatter_points:
        # y_offset = -30 if "Stratos" in model or 'DeepSeek' in model or 's1' in model else 0
        y_offset = 0
        y_offset = -30 if "Open-Reasoner" in model else y_offset
        y_offset = -10 if "OpenThinker" in model else y_offset
        x_offset = 0
        x_offset = -30 if "DeepSeek" in model else x_offset
        x_offset = 60 if "OpenThinker" in model else x_offset
        ax.annotate(
            model, (x_val, y_val), textcoords="offset points",
            xytext=(x_offset, 10 + y_offset), ha='center', fontsize=10, fontweight='bold'
        )
    
    ax.set_xlabel('PVC-VUS', fontsize=14)
    ax.set_ylabel('C-PVC-VUS', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # ax.text(
    #     0.6, 0.02,
    #     "Points below line:\nCalibration reduces AUC reliability",
    #     transform=ax.transAxes, fontsize=9, va='bottom',
    #     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
    # )
    
    ax.legend(title='Model (C-PVC-VUS)', loc='lower right', frameon=True, 
             fancybox=True, shadow=True, fontsize=9)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Cross-domain PVC-VUS vs C-PVC-VUS comparison plot saved to {output_file}")
    
    plt.show()

def main():
    """Main function to run cross-dataset analysis."""
    print("="*60)
    print("CROSS-DATASET ANALYSIS")
    print("="*60)
    
    # Load sweep files
    sweep_data = load_sweep_files()
    
    if len(sweep_data) < 2:
        print(f"Error: Need at least 2 datasets, found {len(sweep_data)}")
        return
    
    print(f"\nFound {len(sweep_data)} datasets: {list(sweep_data.keys())}")
    
    # Calculate cross-dataset averages (FAST VERSION)
    print("\nCalculating cross-dataset averages...")
    averaged_df = calculate_cross_dataset_averages_fast(sweep_data)
    
    # Save averaged data
    averaged_df.to_csv("cross_dataset_averaged_sweep_table.csv", index=False)
    print("Saved averaged data to: cross_dataset_averaged_sweep_table.csv")
    
    # Generate plots
    print("\nGenerating cross-dataset PVC line plot...")
    plot_averaged_pvc_line_plot(averaged_df, "cross_dataset_average_pvc_plot.png")
    
    print("\nGenerating cross-dataset C-PVC 3D grid plot...")
    # plot_averaged_cpvc_3d_grid(averaged_df, "cross_dataset_average_cpvc_3d_grid.png")
     
    print("\nGenerating combined PVC and C-PVC plot...")
    # plot_combined_pvc_cpvc(averaged_df, "cross_dataset_combined_pvc_cpvc_plot.png")
    
    print("\nGenerating cross-domain PVC vs C-PVC comparison plot...")
    plot_cross_domain_pvc_cpvc_comparison(averaged_df, "cross_domain_pvc_cpvc_comparison.png")
    
    print("\nCross-dataset analysis complete!")



if __name__ == "__main__":
    main()


# # cross_dataset_analysis.py
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.colors import LinearSegmentedColormap
# import os

# def load_sweep_files():
#     """Load sweep files from all three datasets."""
#     datasets = {
#         'Math360': 'mathbenchmark_parameter_sweep_table.csv',
#         'TruthfulQA': 'truthfulQA_parameter_sweep_table.csv', 
#         'CSQA': 'CSQA_parameter_sweep_table.csv'
#     }
    
#     sweep_data = {}
    
#     for dataset_name, filename in datasets.items():
#         if os.path.exists(filename):
#             df = pd.read_csv(filename)
#             sweep_data[dataset_name] = df
#             print(f"Loaded {len(df)} rows from {filename}")
#         else:
#             print(f"Warning: {filename} not found")
    
#     return sweep_data

# def calculate_cross_dataset_averages_fast(sweep_data):
#     """Fast calculation using pandas groupby to average across datasets."""
#     print("Combining all datasets...")
    
#     # Combine all dataframes
#     all_dfs = []
#     for dataset_name, df in sweep_data.items():
#         df_copy = df.copy()
#         df_copy['dataset'] = dataset_name
#         all_dfs.append(df_copy)
#         print(f"{dataset_name}: {len(df_copy)} rows, models: {df_copy['model'].unique()}")
    
#     combined_df = pd.concat(all_dfs, ignore_index=True)
#     print(f"Combined dataframe has {len(combined_df)} rows")
    
#     # Check a specific example before averaging
#     print("\nExample before averaging (gamma=0.6, tau=0.25, first model):")
#     example_model = combined_df['model'].iloc[0]
#     example_data = combined_df[(combined_df['gamma'] == 0.6) & 
#                               (combined_df['tau'] == 0.25) & 
#                               (combined_df['model'] == example_model)]
#     print(example_data[['dataset', 'model', 'gamma', 'tau', 'pvc', 'c_pvc']])
    
#     # Group by model, gamma, tau and calculate averages
#     print("\nCalculating averages...")
#     averaged_df = combined_df.groupby(['model', 'gamma', 'tau']).agg({
#         'pvc': 'mean',
#         'c_pvc': 'mean', 
#         'sample_complexity': 'mean',
#         'dataset': 'count'  # Count how many datasets contributed
#     }).reset_index()
    
#     # Rename count column
#     averaged_df = averaged_df.rename(columns={'dataset': 'num_datasets'})
    
#     # Check the same example after averaging
#     print("\nExample after averaging:")
#     example_avg = averaged_df[(averaged_df['gamma'] == 0.6) & 
#                              (averaged_df['tau'] == 0.25) & 
#                              (averaged_df['model'] == example_model)]
#     print(example_avg[['model', 'gamma', 'tau', 'pvc', 'c_pvc', 'num_datasets']])
    
#     # Only keep combinations with data from at least 2 datasets
#     before_filter = len(averaged_df)
#     averaged_df = averaged_df[averaged_df['num_datasets'] >= 2]
#     after_filter = len(averaged_df)
    
#     print(f"\nFiltered from {before_filter} to {after_filter} rows (kept only >= 2 datasets)")
#     print(f"Generated {len(averaged_df)} averaged data points")
#     print(f"Models: {averaged_df['model'].nunique()}")
#     print(f"Gamma-Tau combinations: {len(averaged_df.groupby(['gamma', 'tau']))}")
    
#     # Show distribution of num_datasets
#     print(f"Dataset count distribution:")
#     print(averaged_df['num_datasets'].value_counts().sort_index())
    
#     return averaged_df

# def plot_averaged_pvc_line_plot(averaged_df, output_file=None):
#     """Create PVC line plot similar to Math_combined_model_performance format."""
#     # Use same pastel colors
#     pastel_colors = [
#         '#8da0cb', '#fc8d62', '#66c2a5', '#e78ac3', '#ffb347', '#b19cd9', '#87cefa', '#90ee90',
#         '#ff9999', '#d4a5ff', '#ffcc99', '#c2c2f0', '#ffb6c1', '#c3e6cb', '#ffd700', '#aec6cf',
#         '#cb99c9', '#fdfd96', '#cccccc', '#ff6666'
#     ]
    
#     models = averaged_df['model'].unique()
#     colors = pastel_colors[:len(models)]
#     color_dict = {model: color for model, color in zip(models, colors)}
    
#     # Get PVC values by gamma (PVC doesn't depend on tau)
#     pvc_by_gamma = averaged_df.groupby(['model', 'gamma'])['pvc'].first().reset_index()
    
#     # Calculate gamma threshold values
#     gamma_values = np.arange(0.3, 1.01, 0.01)
#     results = {model: [] for model in models}
    
#     # Get maximum PVC value to determine scale
#     max_pvc = averaged_df['pvc'].max()
    
#     for gamma_threshold in gamma_values:
#         for model in models:
#             # Get PVC value at this exact gamma
#             model_pvc_data = pvc_by_gamma[pvc_by_gamma['model'] == model]
            
#             if not model_pvc_data.empty:
#                 # Find the PVC value at the closest gamma <= threshold
#                 valid_gammas = model_pvc_data[model_pvc_data['gamma'] <= gamma_threshold]
                
#                 if not valid_gammas.empty:
#                     # Get PVC at the highest gamma <= threshold (keep as float)
#                     pvc_value = valid_gammas.loc[valid_gammas['gamma'].idxmax(), 'pvc']
#                 else:
#                     pvc_value = 0.0
#             else:
#                 pvc_value = 0.0
            
#             results[model].append(pvc_value)
    
#     # Create plot
#     fig, ax = plt.subplots(figsize=(10, 8))
    
#     line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', 
#                   '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':']
#     markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 
#               '+', 'x', 'd', '|', '_', '1', '2', '3', '4', 'P']
    
#     for i, model in enumerate(models):
#         ax.plot(
#             gamma_values, results[model], label=model, color=color_dict[model],
#             linewidth=2.5, linestyle=line_styles[i % len(line_styles)],
#             marker=markers[i % len(markers)], markersize=5,
#             markevery=10
#         )
    
#     ax.axhline(y=max_pvc, color='gray', linestyle=':', linewidth=1.5, 
#                alpha=0.7, label=f'Max PVC ({max_pvc:.1f})')
    
#     # ax.set_title('Cross-Dataset Average PVC', fontsize=16, fontweight='bold')
#     ax.set_xlabel('Threshold (γ)', fontsize=14)
#     ax.set_ylabel('Cross-Dataset Average PVC', fontsize=14)
#     ax.set_xticks(np.arange(0.3, 1.1, 0.1))
    
#     # Set Y-axis to show decimal values with smart limits
#     y_min = 0
#     y_max = max_pvc
#     y_max_rounded = np.ceil(y_max * 2) / 2  # Round up to nearest 0.5

#     ax.set_ylim(y_min, y_max_rounded)
#     ax.set_yticks(np.arange(0, y_max_rounded + 0.5, 0.5))
    
#     ax.grid(True, linestyle='--', alpha=0.7)
#     ax.legend(title='Model', loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=10)
    
#     plt.tight_layout()
    
#     if output_file:
#         plt.savefig(output_file, dpi=300, bbox_inches='tight')
#         print(f"Cross-dataset PVC plot saved to {output_file}")
    
#     plt.show()


# def plot_averaged_cpvc_3d_grid(averaged_df, output_file=None):
#     """Create 3D grid plot similar to CSQA_all_models_cpvc_3d_grid format."""
#     # Use same pastel colors
#     pastel_colors = [
#         '#8da0cb', '#fc8d62', '#66c2a5', '#e78ac3', '#ffb347', '#b19cd9', '#87cefa', '#90ee90',
#         '#ff9999', '#d4a5ff', '#ffcc99', '#c2c2f0', '#ffb6c1', '#c3e6cb', '#ffd700', '#aec6cf',
#         '#cb99c9', '#fdfd96', '#cccccc', '#ff6666'
#     ]
    
#     models = averaged_df['model'].unique()
#     colors = pastel_colors[:len(models)]
#     color_dict = {model: color for model, color in zip(models, colors)}
    
#     # Create 4x3 subplot grid
#     fig = plt.figure(figsize=(20, 12))
    
#     for idx, model in enumerate(models):
#         print(f"Processing model {idx+1}/{len(models)}: {model}")
        
#         model_data = averaged_df[averaged_df['model'] == model].copy()
        
#         if model_data.empty:
#             continue
        
#         # Create meshgrid for gamma and tau
#         gamma_unique = sorted(model_data['gamma'].unique())
#         tau_unique = sorted(model_data['tau'].unique())
        
#         gamma_grid, tau_grid = np.meshgrid(gamma_unique, tau_unique)
#         cpvc_grid = np.zeros_like(gamma_grid)
        
#         # Fill the C-PVC grid
#         for i, tau in enumerate(tau_unique):
#             for j, gamma in enumerate(gamma_unique):
#                 row = model_data[(model_data['gamma'] == gamma) & (model_data['tau'] == tau)]
#                 if not row.empty:
#                     cpvc_grid[i, j] = row['c_pvc'].iloc[0]
        
#         # Create custom colormap based on model's assigned color
#         model_color = color_dict[model]
#         import matplotlib.colors as mcolors
#         rgb = mcolors.hex2color(model_color)
#         darker_rgb = tuple(c * 0.6 for c in rgb)
#         darker_color = mcolors.rgb2hex(darker_rgb)
        
#         colors_list = ['white', model_color, darker_color]
#         custom_cmap = LinearSegmentedColormap.from_list(f'{model}_cmap', colors_list, N=256)
        
#         # Create subplot
#         ax = fig.add_subplot(3, 4, idx + 1, projection='3d')
        
#         # Create surface plot
#         surf = ax.plot_surface(gamma_grid, tau_grid, cpvc_grid, 
#                               cmap=custom_cmap, alpha=0.9,
#                               linewidth=0, antialiased=True,
#                               shade=True, rcount=40, ccount=40)
        
#         # Customize each subplot
#         ax.set_xlabel('γ', fontsize=10)
#         ax.set_ylabel('τ', fontsize=10)
#         ax.set_zlabel('C-PVC', fontsize=10)
#         ax.set_title(model, fontsize=12, pad=10, fontweight='bold')
        
#         # Set viewing angle
#         ax.view_init(elev=30, azim=-45)
        
#         # Smaller tick labels
#         ax.tick_params(labelsize=8)
    
#     # Create legend in the last subplot
#     if len(models) < 12:
#         ax_legend = fig.add_subplot(3, 4, 12)
#         ax_legend.axis('off')
        
#         # Create legend elements
#         legend_elements = []
#         for model in models:
#             legend_elements.append(plt.Rectangle((0, 0), 1, 1, 
#                                             facecolor=color_dict[model], 
#                                             edgecolor='black', 
#                                             label=model))
        
#         # Center the legend perfectly in the subplot
#         legend = ax_legend.legend(handles=legend_elements, 
#                                 loc='center', 
#                                 fontsize=10, 
#                                 title='Models', 
#                                 title_fontsize=12,
#                                 frameon=True, 
#                                 fancybox=True, 
#                                 shadow=True,
#                                 bbox_to_anchor=(0.5, 0.5))
        
#         legend.get_title().set_fontweight('bold')
#         ax_legend.set_xlim(0, 1)
#         ax_legend.set_ylim(0, 1)
    
#     # plt.suptitle('Cross-Dataset Average C-PVC Surfaces for All Models', fontsize=18, y=0.95, fontweight='bold')
#     plt.tight_layout(rect=[0, 0, 1, 0.93])
    
#     if output_file:
#         plt.savefig(output_file, dpi=300, bbox_inches='tight')
#         print(f"Cross-dataset 3D grid plot saved to {output_file}")
    
#     plt.show()

# def plot_combined_pvc_cpvc(averaged_df, output_file=None):
#     """Create combined PVC line plot (left) and C-PVC 3D grid plot (right)."""
#     import matplotlib.gridspec as gridspec
    
#     # Use same pastel colors
#     pastel_colors = [
#         '#8da0cb', '#fc8d62', '#66c2a5', '#e78ac3', '#ffb347', '#b19cd9', '#87cefa', '#90ee90',
#         '#ff9999', '#d4a5ff', '#ffcc99', '#c2c2f0', '#ffb6c1', '#c3e6cb', '#ffd700', '#aec6cf',
#         '#cb99c9', '#fdfd96', '#cccccc', '#ff6666'
#     ]
    
#     models = averaged_df['model'].unique()
#     colors = pastel_colors[:len(models)]
#     color_dict = {model: color for model, color in zip(models, colors)}
    
#     # Create figure with custom grid layout: 4 rows, 4 columns
#     fig = plt.figure(figsize=(28, 16))
#     gs = gridspec.GridSpec(4, 4, width_ratios=[1.2, 1, 1, 1])
    
#     # LEFT SUBPLOT: PVC LINE PLOT (spans 4 rows, 1 column)
#     ax1 = fig.add_subplot(gs[:, 0])
    
#     # Get PVC values by gamma
#     pvc_by_gamma = averaged_df.groupby(['model', 'gamma'])['pvc'].first().reset_index()
#     gamma_values = np.arange(0.3, 1.01, 0.01)
#     results = {model: [] for model in models}
#     max_pvc = averaged_df['pvc'].max()
    
#     for gamma_threshold in gamma_values:
#         for model in models:
#             model_pvc_data = pvc_by_gamma[pvc_by_gamma['model'] == model]
            
#             if not model_pvc_data.empty:
#                 valid_gammas = model_pvc_data[model_pvc_data['gamma'] <= gamma_threshold]
#                 if not valid_gammas.empty:
#                     pvc_value = valid_gammas.loc[valid_gammas['gamma'].idxmax(), 'pvc']
#                 else:
#                     pvc_value = 0.0
#             else:
#                 pvc_value = 0.0
            
#             results[model].append(pvc_value)
    
#     line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
#     markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
    
#     for i, model in enumerate(models):
#         ax1.plot(
#             gamma_values, results[model], color=color_dict[model],
#             linewidth=2.5, linestyle=line_styles[i % len(line_styles)],
#             marker=markers[i % len(markers)], markersize=5, markevery=10
#         )
    
#     y_max_rounded = np.ceil(max_pvc * 2) / 2
#     ax1.set_ylim(0, y_max_rounded)
#     ax1.set_yticks(np.arange(0, y_max_rounded + 0.5, 0.5))
    
#     ax1.set_title('Cross-Dataset Average PVC - γ', fontsize=16, fontweight='bold')
#     ax1.set_xlabel('Threshold (γ)', fontsize=14)
#     ax1.set_ylabel('PVC', fontsize=14)
#     ax1.set_xticks(np.arange(0.3, 1.1, 0.1))
#     ax1.grid(True, linestyle='--', alpha=0.7)
    
#     # RIGHT SUBPLOTS: 4x3 grid of C-PVC 3D surfaces
#     for idx, model in enumerate(models):
#         if idx >= 11:  # Only 11 models, save last spot for legend
#             break
            
#         row = idx // 3
#         col = idx % 3 + 1  # +1 because first column is PVC plot
        
#         ax = fig.add_subplot(gs[row, col], projection='3d')
        
#         model_data = averaged_df[averaged_df['model'] == model].copy()
        
#         if not model_data.empty:
#             gamma_unique = sorted(model_data['gamma'].unique())
#             tau_unique = sorted(model_data['tau'].unique())
            
#             gamma_grid, tau_grid = np.meshgrid(gamma_unique, tau_unique)
#             cpvc_grid = np.zeros_like(gamma_grid)
            
#             for i, tau in enumerate(tau_unique):
#                 for j, gamma in enumerate(gamma_unique):
#                     row_data = model_data[(model_data['gamma'] == gamma) & (model_data['tau'] == tau)]
#                     if not row_data.empty:
#                         cpvc_grid[i, j] = row_data['c_pvc'].iloc[0]
            
#             # Create custom colormap
#             model_color = color_dict[model]
#             import matplotlib.colors as mcolors
#             rgb = mcolors.hex2color(model_color)
#             darker_rgb = tuple(c * 0.6 for c in rgb)
#             darker_color = mcolors.rgb2hex(darker_rgb)
            
#             colors_list = ['white', model_color, darker_color]
#             custom_cmap = LinearSegmentedColormap.from_list(f'{model}_cmap', colors_list, N=256)
            
#             surf = ax.plot_surface(gamma_grid, tau_grid, cpvc_grid, 
#                                   cmap=custom_cmap, alpha=0.9,
#                                   linewidth=0, antialiased=True,
#                                   shade=True, rcount=30, ccount=30)
            
#             ax.set_xlabel('γ', fontsize=8)
#             ax.set_ylabel('τ', fontsize=8)
#             ax.set_zlabel('C-PVC', fontsize=8)
#             ax.set_title(model, fontsize=10, pad=5, fontweight='bold')
#             ax.view_init(elev=30, azim=-45)
#             ax.tick_params(labelsize=6)
    
#     # Legend in the last position (4th row, 4th column)
#     if len(models) < 12:
#         ax_legend = fig.add_subplot(gs[3, 3])  # Bottom right position
#         ax_legend.axis('off')
        
#         legend_elements = []
#         for model in models:
#             legend_elements.append(plt.Rectangle((0, 0), 1, 1, 
#                                                facecolor=color_dict[model], 
#                                                edgecolor='black', linewidth=1,
#                                                label=model))
        
#         legend = ax_legend.legend(handles=legend_elements, 
#                                  loc='center', 
#                                  fontsize=8, 
#                                  title='Models', 
#                                  title_fontsize=9,
#                                  frameon=True, 
#                                  fancybox=True, 
#                                  shadow=True)
        
#         legend.get_title().set_fontweight('bold')
    
#     plt.suptitle('Cross-Dataset Analysis: PVC and C-PVC Comparison', fontsize=18, y=0.95, fontweight='bold')
#     plt.tight_layout(rect=[0, 0, 1, 0.93])
    
#     if output_file:
#         plt.savefig(output_file, dpi=300, bbox_inches='tight')
#         print(f"Combined plot saved to {output_file}")
    
#     plt.show()


# def main():
#     """Main function to run cross-dataset analysis."""
#     print("="*60)
#     print("CROSS-DATASET ANALYSIS")
#     print("="*60)
    
#     # Load sweep files
#     sweep_data = load_sweep_files()
    
#     if len(sweep_data) < 2:
#         print(f"Error: Need at least 2 datasets, found {len(sweep_data)}")
#         return
    
#     print(f"\nFound {len(sweep_data)} datasets: {list(sweep_data.keys())}")
    
#     # Calculate cross-dataset averages (FAST VERSION)
#     print("\nCalculating cross-dataset averages...")
#     averaged_df = calculate_cross_dataset_averages_fast(sweep_data)
    
#     # Save averaged data
#     averaged_df.to_csv("cross_dataset_averaged_sweep_table.csv", index=False)
#     print("Saved averaged data to: cross_dataset_averaged_sweep_table.csv")
    
#     # Generate plots
#     print("\nGenerating cross-dataset PVC line plot...")
#     plot_averaged_pvc_line_plot(averaged_df, "cross_dataset_average_pvc_plot.png")
    
#     print("\nGenerating cross-dataset C-PVC 3D grid plot...")
#     plot_averaged_cpvc_3d_grid(averaged_df, "cross_dataset_average_cpvc_3d_grid.png")
    
#     print("\nGenerating combined PVC and C-PVC plot...")
#     plot_combined_pvc_cpvc(averaged_df, "cross_dataset_combined_pvc_cpvc_plot.png")
    
#     print("\nCross-dataset analysis complete!")



# if __name__ == "__main__":
#     main()