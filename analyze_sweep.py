"""Analyze and visualize results from parameter sweeps."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def load_sweep_results(sweep_dir="outputs"):
    """Load results from a sweep experiment."""
    sweep_path = Path(sweep_dir)

    results = []
    for result_file in sweep_path.glob("**/results.json"):
        with open(result_file) as f:
            data = json.load(f)
            results.append(data)

    return results


def create_summary_plots(results, output_dir="outputs/analysis"):
    """Create summary visualizations of sweep results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Extract data into pandas DataFrame
    data = []
    for result in results:
        params = result.get('params', {})
        summary = result.get('summary', {})

        row = {**params, **summary}
        data.append(row)

    df = pd.DataFrame(data)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Parameter Sweep Analysis', fontsize=16)

    # 1. Species diversity vs rmax
    if 'rmax' in df.columns and 'species_entropy_mean' in df.columns:
        ax = axes[0, 0]
        for mass in df['mass'].unique() if 'mass' in df.columns else [None]:
            mask = df['mass'] == mass if mass is not None else [True] * len(df)
            subset = df[mask]
            ax.plot(subset['rmax'], subset['species_entropy_mean'],
                   marker='o', label=f'mass={mass}')
        ax.set_xlabel('Interaction Radius (rmax)')
        ax.set_ylabel('Mean Species Entropy')
        ax.set_title('Diversity vs Interaction Radius')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 2. Activity vs parameters
    if 'activity_mean' in df.columns:
        ax = axes[0, 1]
        if 'rmax' in df.columns and 'mass' in df.columns:
            pivot = df.pivot_table(values='activity_mean',
                                  index='rmax', columns='mass')
            sns.heatmap(pivot, annot=True, fmt='.2f', ax=ax, cmap='viridis')
            ax.set_title('Mean Activity (rmax vs mass)')

    # 3. Clustering coefficient
    if 'clustering_coeff_mean' in df.columns and 'rmax' in df.columns:
        ax = axes[0, 2]
        ax.scatter(df['rmax'], df['clustering_coeff_mean'],
                  c=df.get('mass', 0.02), cmap='coolwarm', s=100, alpha=0.6)
        ax.set_xlabel('Interaction Radius (rmax)')
        ax.set_ylabel('Mean Clustering Coefficient')
        ax.set_title('Spatial Clustering')
        ax.grid(True, alpha=0.3)
        plt.colorbar(ax.collections[0], ax=ax, label='Mass')

    # 4. Energy vs velocity
    if 'kinetic_energy_mean' in df.columns and 'mean_velocity_mean' in df.columns:
        ax = axes[1, 0]
        ax.scatter(df['mean_velocity_mean'], df['kinetic_energy_mean'],
                  c=df.get('rmax', 0.2), cmap='plasma', s=100, alpha=0.6)
        ax.set_xlabel('Mean Velocity')
        ax.set_ylabel('Mean Kinetic Energy')
        ax.set_title('Energy-Velocity Relationship')
        ax.grid(True, alpha=0.3)
        plt.colorbar(ax.collections[0], ax=ax, label='rmax')

    # 5. Number of species clusters
    if 'n_species_clusters_mean' in df.columns:
        ax = axes[1, 1]
        if 'mutation_prob' in df.columns:
            ax.boxplot([df[df['mutation_prob'] == mp]['n_species_clusters_mean'].values
                       for mp in sorted(df['mutation_prob'].unique())],
                      labels=[f'{mp:.3f}' for mp in sorted(df['mutation_prob'].unique())])
            ax.set_xlabel('Mutation Probability')
            ax.set_ylabel('Number of Species Clusters')
            ax.set_title('Diversity vs Mutation Rate')
            ax.grid(True, alpha=0.3, axis='y')

    # 6. Parameter correlation heatmap
    ax = axes[1, 2]
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Compute correlation
    corr = df[numeric_cols].corr()
    sns.heatmap(corr.iloc[:5, :5], annot=True, fmt='.2f', ax=ax,
               cmap='coolwarm', center=0, vmin=-1, vmax=1)
    ax.set_title('Parameter Correlations')

    plt.tight_layout()
    plt.savefig(output_dir / 'sweep_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved analysis to {output_dir / 'sweep_analysis.png'}")

    # Save summary statistics
    summary_file = output_dir / 'summary_stats.txt'
    with open(summary_file, 'w') as f:
        f.write("Parameter Sweep Summary Statistics\n")
        f.write("=" * 60 + "\n\n")

        f.write("Dataset Info:\n")
        f.write(f"  Total experiments: {len(df)}\n")
        f.write(f"  Parameters explored: {[c for c in ['mass', 'rmax', 'mutation_prob'] if c in df.columns]}\n\n")

        f.write("Key Metrics Summary:\n")
        for col in ['species_entropy_mean', 'activity_mean', 'kinetic_energy_mean',
                   'n_species_clusters_mean', 'clustering_coeff_mean']:
            if col in df.columns:
                f.write(f"\n{col}:\n")
                f.write(f"  Mean: {df[col].mean():.4f}\n")
                f.write(f"  Std:  {df[col].std():.4f}\n")
                f.write(f"  Min:  {df[col].min():.4f}\n")
                f.write(f"  Max:  {df[col].max():.4f}\n")

        # Best configurations
        f.write("\n" + "=" * 60 + "\n")
        f.write("Best Configurations:\n\n")

        if 'species_entropy_mean' in df.columns:
            best_diversity = df.loc[df['species_entropy_mean'].idxmax()]
            f.write("Highest Diversity:\n")
            for col in ['mass', 'rmax', 'mutation_prob', 'species_entropy_mean']:
                if col in best_diversity:
                    f.write(f"  {col}: {best_diversity[col]}\n")
            f.write("\n")

        if 'activity_mean' in df.columns:
            best_activity = df.loc[df['activity_mean'].idxmax()]
            f.write("Highest Activity:\n")
            for col in ['mass', 'rmax', 'mutation_prob', 'activity_mean']:
                if col in best_activity:
                    f.write(f"  {col}: {best_activity[col]}\n")
            f.write("\n")

        if 'clustering_coeff_mean' in df.columns:
            best_clustering = df.loc[df['clustering_coeff_mean'].idxmax()]
            f.write("Highest Clustering:\n")
            for col in ['mass', 'rmax', 'mutation_prob', 'clustering_coeff_mean']:
                if col in best_clustering:
                    f.write(f"  {col}: {best_clustering[col]}\n")

    print(f"Saved summary statistics to {summary_file}")

    return df


if __name__ == "__main__":
    import sys

    sweep_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs"

    print(f"Loading results from {sweep_dir}...")
    results = load_sweep_results(sweep_dir)

    if results:
        print(f"Found {len(results)} experiment results")
        df = create_summary_plots(results)
        print("\nDataFrame shape:", df.shape)
        print("\nColumns:", df.columns.tolist())
    else:
        print("No results found. Make sure to run a sweep first!")
