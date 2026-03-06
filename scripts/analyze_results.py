"""Analyze and visualize results from wandb experiments."""

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def load_wandb_data(max_runs=30):
    """Load data from wandb finished runs."""
    api = wandb.Api()
    runs = api.runs('benjamin-steel-projects/evo-particle-life',
                    filters={'state': 'finished'},
                    order='-created_at')

    print(f'Found {len(runs)} finished runs')

    all_data = []
    time_series_data = {}

    for run in runs[:max_runs]:
        config = run.config
        history = run.history()

        if len(history) == 0:
            continue

        run_info = {
            'run_name': run.name,
            'run_id': run.id,
            'created_at': run.created_at,
            'max_copy_dist': config.get('mutation', {}).get('max_copy_dist', config.get('max_copy_dist', None)),
            'copy_prob': config.get('mutation', {}).get('copy_prob', config.get('copy_prob', None)),
            'mutation_prob': config.get('mutation', {}).get('mutation_prob', config.get('mutation_prob', None)),
            'species_dim': config.get('simulation', {}).get('species_dim', config.get('species_dim', None)),
            'num_particles': config.get('simulation', {}).get('num_particles', 4000),
            'num_steps': config.get('experiment', {}).get('num_steps', 4000),
            'num_datapoints': len(history),
        }

        # Add summary statistics for key metrics
        for metric in ['species_entropy', 'num_unique_species', 'compression_complexity',
                       'spatial_frequency', 'species_variance', 'species_pairwise_diversity',
                       'activity', 'clustering_coeff']:
            if metric in history.columns:
                vals = history[metric].dropna()
                if len(vals) > 0:
                    run_info[f'{metric}_mean'] = vals.mean()
                    run_info[f'{metric}_final'] = vals.iloc[-1]
                    run_info[f'{metric}_max'] = vals.max()
                    run_info[f'{metric}_std'] = vals.std()
                    run_info[f'{metric}_trend'] = (vals.iloc[-1] - vals.iloc[0]) if len(vals) > 1 else 0

        all_data.append(run_info)
        time_series_data[run.id] = history

    df = pd.DataFrame(all_data)
    print(f'\nCollected data from {len(df)} runs')

    return df, time_series_data

def plot_parameter_effects(df, output_dir='analysis_plots'):
    """Plot how parameters affect key metrics."""
    Path(output_dir).mkdir(exist_ok=True)

    # Metrics to analyze
    metrics = ['species_entropy_final', 'num_unique_species_final',
               'compression_complexity_final', 'activity_final']
    metric_labels = ['Species Entropy (Final)', 'Unique Species (Final)',
                     'Compression Complexity (Final)', 'Activity (Final)']

    # Parameters to test
    params = ['max_copy_dist', 'copy_prob', 'mutation_prob', 'species_dim']
    param_labels = ['Max Copy Distance', 'Copy Probability',
                    'Mutation Probability', 'Species Dimensionality']

    fig, axes = plt.subplots(len(metrics), len(params), figsize=(20, 16))
    fig.suptitle('Parameter Effects on Key Metrics', fontsize=16, y=0.995)

    for i, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        for j, (param, plabel) in enumerate(zip(params, param_labels)):
            ax = axes[i, j]

            # Filter valid data
            valid = df[[param, metric]].dropna()

            if len(valid) > 0:
                # Group by parameter and plot
                grouped = valid.groupby(param)[metric].agg(['mean', 'std', 'count'])

                if len(grouped) > 1:
                    ax.errorbar(grouped.index, grouped['mean'],
                               yerr=grouped['std'],
                               marker='o', capsize=5, capthick=2, markersize=8)
                    ax.set_xlabel(plabel)
                    ax.set_ylabel(mlabel if j == 0 else '')
                    ax.grid(True, alpha=0.3)

                    # Add sample counts
                    for x, count in zip(grouped.index, grouped['count']):
                        ax.text(x, ax.get_ylim()[0], f'n={int(count)}',
                               ha='center', va='top', fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'Insufficient data',
                           ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'No data',
                       ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/parameter_effects.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {output_dir}/parameter_effects.png')
    plt.close()

def plot_correlation_matrix(df, output_dir='analysis_plots'):
    """Plot correlation matrix of key variables."""
    # Select numerical columns
    cols = [col for col in df.columns if '_final' in col or '_mean' in col or '_trend' in col]
    cols = [col for col in cols if col in df.columns]

    if len(cols) < 2:
        print("Not enough metrics for correlation matrix")
        return

    corr_data = df[cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Matrix: Metrics and Trends', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {output_dir}/correlation_matrix.png')
    plt.close()

def plot_time_series_examples(time_series_data, df, output_dir='analysis_plots', n_examples=6):
    """Plot time series for interesting example runs."""

    # Select diverse runs based on parameters
    selected_runs = []

    # Try to get runs with different parameter combinations
    for species_dim in df['species_dim'].dropna().unique()[:2]:
        for mutation_prob in df['mutation_prob'].dropna().unique()[:2]:
            matching = df[(df['species_dim'] == species_dim) &
                         (df['mutation_prob'] == mutation_prob)]
            if len(matching) > 0:
                selected_runs.append(matching.iloc[0])
                if len(selected_runs) >= n_examples:
                    break
        if len(selected_runs) >= n_examples:
            break

    if len(selected_runs) == 0:
        selected_runs = df.head(n_examples).to_dict('records')

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Time Series Examples from Different Runs', fontsize=16)
    axes = axes.flatten()

    metrics = ['species_entropy', 'num_unique_species', 'compression_complexity',
               'species_variance', 'activity', 'clustering_coeff']
    labels = ['Species Entropy', 'Unique Species', 'Compression Complexity',
              'Species Variance', 'Activity', 'Clustering Coefficient']

    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[idx]

        for run_info in selected_runs:
            run_id = run_info['run_id']
            if run_id in time_series_data:
                history = time_series_data[run_id]
                if metric in history.columns:
                    data = history[['step', metric]].dropna()
                    if len(data) > 0:
                        params_str = f"dim={run_info.get('species_dim', '?')}, mut={run_info.get('mutation_prob', '?')}"
                        ax.plot(data['step'], data[metric], label=params_str, alpha=0.7)

        ax.set_xlabel('Step')
        ax.set_ylabel(label)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/time_series_examples.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {output_dir}/time_series_examples.png')
    plt.close()

def plot_complexity_vs_diversity(df, output_dir='analysis_plots'):
    """Plot relationship between visual complexity and species diversity."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Compression complexity vs species entropy
    ax = axes[0]
    valid = df[['compression_complexity_final', 'species_entropy_final', 'species_dim']].dropna()
    if len(valid) > 0:
        scatter = ax.scatter(valid['compression_complexity_final'],
                            valid['species_entropy_final'],
                            c=valid['species_dim'],
                            s=100, alpha=0.6, cmap='viridis')
        ax.set_xlabel('Compression Complexity (Final)')
        ax.set_ylabel('Species Entropy (Final)')
        ax.set_title('Complexity vs Diversity')
        plt.colorbar(scatter, ax=ax, label='Species Dim')
        ax.grid(True, alpha=0.3)

    # Compression complexity vs unique species
    ax = axes[1]
    valid = df[['compression_complexity_final', 'num_unique_species_final', 'mutation_prob']].dropna()
    if len(valid) > 0:
        scatter = ax.scatter(valid['compression_complexity_final'],
                            valid['num_unique_species_final'],
                            c=valid['mutation_prob'],
                            s=100, alpha=0.6, cmap='plasma')
        ax.set_xlabel('Compression Complexity (Final)')
        ax.set_ylabel('Unique Species (Final)')
        ax.set_title('Complexity vs Species Count')
        plt.colorbar(scatter, ax=ax, label='Mutation Prob')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/complexity_vs_diversity.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {output_dir}/complexity_vs_diversity.png')
    plt.close()

def print_key_findings(df):
    """Print summary statistics and key findings."""
    print("\n" + "="*70)
    print("KEY FINDINGS FROM EXPERIMENTS")
    print("="*70)

    print(f"\nTotal runs analyzed: {len(df)}")
    print(f"Parameter coverage:")
    for param in ['max_copy_dist', 'copy_prob', 'mutation_prob', 'species_dim']:
        if param in df.columns:
            unique_vals = df[param].dropna().unique()
            print(f"  {param}: {sorted(unique_vals)}")

    print(f"\n--- Species Diversity Metrics ---")
    for metric in ['species_entropy_final', 'num_unique_species_final', 'species_variance_final']:
        if metric in df.columns:
            vals = df[metric].dropna()
            if len(vals) > 0:
                print(f"{metric}:")
                print(f"  Range: [{vals.min():.3f}, {vals.max():.3f}]")
                print(f"  Mean: {vals.mean():.3f} ± {vals.std():.3f}")

    print(f"\n--- Visual Complexity Metrics ---")
    for metric in ['compression_complexity_final', 'spatial_frequency_final']:
        if metric in df.columns:
            vals = df[metric].dropna()
            if len(vals) > 0:
                print(f"{metric}:")
                print(f"  Range: [{vals.min():.3f}, {vals.max():.3f}]")
                print(f"  Mean: {vals.mean():.3f} ± {vals.std():.3f}")

    # Correlations
    print(f"\n--- Key Correlations ---")
    if 'compression_complexity_final' in df.columns and 'species_entropy_final' in df.columns:
        corr = df[['compression_complexity_final', 'species_entropy_final']].corr().iloc[0, 1]
        print(f"Compression Complexity vs Species Entropy: {corr:.3f}")

    if 'mutation_prob' in df.columns and 'num_unique_species_final' in df.columns:
        valid = df[['mutation_prob', 'num_unique_species_final']].dropna()
        if len(valid) > 0:
            corr = valid.corr().iloc[0, 1]
            print(f"Mutation Probability vs Unique Species: {corr:.3f}")

    # Best runs
    print(f"\n--- Most Interesting Runs ---")
    if 'species_entropy_final' in df.columns:
        best_entropy = df.loc[df['species_entropy_final'].idxmax()]
        print(f"\nHighest diversity (entropy={best_entropy['species_entropy_final']:.3f}):")
        print(f"  copy_dist={best_entropy.get('max_copy_dist')}, copy_prob={best_entropy.get('copy_prob')}")
        print(f"  mutation_prob={best_entropy.get('mutation_prob')}, species_dim={best_entropy.get('species_dim')}")

    if 'compression_complexity_final' in df.columns:
        best_complex = df.loc[df['compression_complexity_final'].idxmax()]
        print(f"\nHighest complexity (complexity={best_complex['compression_complexity_final']:.3f}):")
        print(f"  copy_dist={best_complex.get('max_copy_dist')}, copy_prob={best_complex.get('copy_prob')}")
        print(f"  mutation_prob={best_complex.get('mutation_prob')}, species_dim={best_complex.get('species_dim')}")

    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    print("Loading data from wandb...")
    df, time_series = load_wandb_data(max_runs=30)

    print("\nGenerating plots...")
    plot_parameter_effects(df)
    plot_correlation_matrix(df)
    plot_time_series_examples(time_series, df)
    plot_complexity_vs_diversity(df)

    print_key_findings(df)

    # Save summary data
    df.to_csv('analysis_plots/summary_data.csv', index=False)
    print("\nSaved summary data to analysis_plots/summary_data.csv")
    print("\nAnalysis complete! Check the analysis_plots/ directory.")
