"""Analyze results from low_mutation_sweep experiment."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wandb.apis.public import Api

# Initialize wandb API
api = Api()
runs = api.runs("benjamin-steel-projects/evo-particle-life",
                filters={"config.experiment.name": "low_mutation_sweep"})

print(f"Found {len(runs)} runs from low_mutation_sweep")

# Collect run data
data = []
for run in runs:
    config = run.config
    summary = run.summary._json_dict

    # Extract config parameters
    row = {
        'run_id': run.id,
        'run_name': run.name,
        'state': run.state,
        'mutation_prob': config.get('mutation', {}).get('mutation_prob'),
        'max_copy_dist': config.get('mutation', {}).get('max_copy_dist'),
        'copy_prob': config.get('mutation', {}).get('copy_prob'),
        'species_dim': config.get('simulation', {}).get('species_dim'),
        'num_particles': config.get('simulation', {}).get('num_particles'),
    }

    # Extract summary metrics (using mean values)
    metrics_of_interest = [
        'species_entropy_mean', 'species_variance_mean', 'num_unique_species_mean',
        'species_pairwise_diversity_mean', 'mean_nn_distance_mean',
        'clustering_coeff_mean', 'activity_mean', 'momentum_mean',
        'compression_complexity_mean', 'normalized_complexity_mean',
        'oom_occurred', 'oom_step', 'completion_ratio'
    ]

    for metric in metrics_of_interest:
        row[metric] = summary.get(metric)

    data.append(row)

df = pd.DataFrame(data)

# Handle OOM data
if 'oom_occurred' not in df.columns or df['oom_occurred'].isna().all():
    df['oom_occurred'] = False
    df['completion_ratio'] = 1.0

df['oom_occurred'] = df['oom_occurred'].fillna(False)
df['completion_ratio'] = df['completion_ratio'].fillna(1.0)

# Filter to only include runs with actual metrics data
df = df[df['species_entropy_mean'].notna()].copy()

print(f"\nDataFrame shape: {df.shape}")
print(f"\nOOM runs: {df['oom_occurred'].sum()}")
print(f"Completed runs: {(~df['oom_occurred']).sum()}")
print(f"\nUnique parameter combinations:")
print(f"  mutation_prob: {sorted(df['mutation_prob'].unique())}")
print(f"  max_copy_dist: {sorted(df['max_copy_dist'].unique())}")
print(f"  copy_prob: {sorted(df['copy_prob'].unique())}")
print(f"  species_dim: {sorted(df['species_dim'].unique())}")

# Save raw data
df.to_csv('low_mutation_sweep_results.csv', index=False)
print(f"\n✓ Saved raw results to low_mutation_sweep_results.csv")

# Create analysis plots
fig = plt.figure(figsize=(20, 16))

# 1. Species Diversity vs Mutation Rate (by copy distance)
ax1 = plt.subplot(3, 3, 1)
for copy_dist in sorted(df['max_copy_dist'].unique()):
    subset = df[df['max_copy_dist'] == copy_dist]
    plt.plot(subset['mutation_prob'], subset['species_entropy_mean'],
             marker='o', label=f'copy_dist={copy_dist:.2f}')
plt.xlabel('Mutation Probability')
plt.ylabel('Mean Species Entropy')
plt.title('Species Entropy vs Mutation Rate')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Unique Species vs Mutation Rate
ax2 = plt.subplot(3, 3, 2)
for copy_dist in sorted(df['max_copy_dist'].unique()):
    subset = df[df['max_copy_dist'] == copy_dist]
    plt.plot(subset['mutation_prob'], subset['num_unique_species_mean'],
             marker='o', label=f'copy_dist={copy_dist:.2f}')
plt.xlabel('Mutation Probability')
plt.ylabel('Mean Unique Species')
plt.title('Unique Species vs Mutation Rate')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Pairwise Diversity vs Copy Distance
ax3 = plt.subplot(3, 3, 3)
for mut_prob in sorted(df['mutation_prob'].unique()):
    subset = df[df['mutation_prob'] == mut_prob]
    plt.plot(subset['max_copy_dist'], subset['species_pairwise_diversity_mean'],
             marker='o', label=f'mut={mut_prob:.3f}')
plt.xlabel('Max Copy Distance')
plt.ylabel('Mean Pairwise Diversity')
plt.title('Pairwise Diversity vs Copy Distance')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Heatmap: Species Entropy (mutation_prob × max_copy_dist)
ax4 = plt.subplot(3, 3, 4)
pivot = df.pivot_table(values='species_entropy_mean',
                       index='mutation_prob',
                       columns='max_copy_dist')
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis', ax=ax4)
plt.title('Species Entropy Heatmap')
plt.ylabel('Mutation Prob')
plt.xlabel('Max Copy Distance')

# 5. Heatmap: Unique Species
ax5 = plt.subplot(3, 3, 5)
pivot = df.pivot_table(values='num_unique_species_mean',
                       index='mutation_prob',
                       columns='max_copy_dist')
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='viridis', ax=ax5)
plt.title('Unique Species Heatmap')
plt.ylabel('Mutation Prob')
plt.xlabel('Max Copy Distance')

# 6. Spatial Clustering
ax6 = plt.subplot(3, 3, 6)
for copy_dist in sorted(df['max_copy_dist'].unique()):
    subset = df[df['max_copy_dist'] == copy_dist]
    plt.plot(subset['mutation_prob'], subset['clustering_coeff_mean'],
             marker='o', label=f'copy_dist={copy_dist:.2f}')
plt.xlabel('Mutation Probability')
plt.ylabel('Mean Clustering Coefficient')
plt.title('Spatial Clustering vs Mutation Rate')
plt.legend()
plt.grid(True, alpha=0.3)

# 7. Complexity vs Diversity
ax7 = plt.subplot(3, 3, 7)
plt.scatter(df['species_entropy_mean'], df['normalized_complexity_mean'],
            c=df['mutation_prob'], cmap='plasma', s=100, alpha=0.6)
plt.colorbar(label='Mutation Prob')
plt.xlabel('Mean Species Entropy')
plt.ylabel('Mean Normalized Complexity')
plt.title('Visual Complexity vs Species Diversity')
plt.grid(True, alpha=0.3)

# 8. Copy Probability Effect
ax8 = plt.subplot(3, 3, 8)
for copy_prob in sorted(df['copy_prob'].unique()):
    subset = df[df['copy_prob'] == copy_prob]
    means = subset.groupby('mutation_prob')['species_entropy_mean'].mean()
    plt.plot(means.index, means.values, marker='o', label=f'copy_prob={copy_prob:.4f}')
plt.xlabel('Mutation Probability')
plt.ylabel('Mean Species Entropy')
plt.title('Effect of Copy Probability')
plt.legend()
plt.grid(True, alpha=0.3)

# 9. Species Dimensionality Effect
ax9 = plt.subplot(3, 3, 9)
for species_dim in sorted(df['species_dim'].unique()):
    subset = df[df['species_dim'] == species_dim]
    means = subset.groupby('mutation_prob')['species_entropy_mean'].mean()
    plt.plot(means.index, means.values, marker='o', label=f'species_dim={int(species_dim)}')
plt.xlabel('Mutation Probability')
plt.ylabel('Mean Species Entropy')
plt.title('Effect of Species Dimensionality')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('low_mutation_sweep_analysis.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved analysis plots to low_mutation_sweep_analysis.png")

# Generate insights summary
print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

# Best diversity configuration
best_entropy = df.loc[df['species_entropy_mean'].idxmax()]
print(f"\n1. HIGHEST SPECIES ENTROPY:")
print(f"   Entropy: {best_entropy['species_entropy_mean']:.3f}")
print(f"   mutation_prob: {best_entropy['mutation_prob']:.4f}")
print(f"   max_copy_dist: {best_entropy['max_copy_dist']:.2f}")
print(f"   copy_prob: {best_entropy['copy_prob']:.4f}")
print(f"   species_dim: {int(best_entropy['species_dim'])}")

# Most unique species
best_unique = df.loc[df['num_unique_species_mean'].idxmax()]
print(f"\n2. MOST UNIQUE SPECIES:")
print(f"   Unique species: {best_unique['num_unique_species_mean']:.1f}")
print(f"   mutation_prob: {best_unique['mutation_prob']:.4f}")
print(f"   max_copy_dist: {best_unique['max_copy_dist']:.2f}")
print(f"   copy_prob: {best_unique['copy_prob']:.4f}")

# Copy distance effect
print(f"\n3. COPY DISTANCE TRADE-OFF:")
copy_dist_effect = df.groupby('max_copy_dist').agg({
    'species_entropy_mean': 'mean',
    'clustering_coeff_mean': 'mean'
})
print(copy_dist_effect.to_string())

# Mutation rate sweet spot
print(f"\n4. MUTATION RATE EFFECTS:")
mut_effect = df.groupby('mutation_prob').agg({
    'species_entropy_mean': 'mean',
    'num_unique_species_mean': 'mean',
    'species_pairwise_diversity_mean': 'mean'
})
print(mut_effect.to_string())

# Complexity correlation
complexity_corr = df[['species_entropy_mean', 'normalized_complexity_mean']].corr().iloc[0, 1]
print(f"\n5. COMPLEXITY-DIVERSITY CORRELATION:")
print(f"   Correlation: {complexity_corr:.3f}")
if not np.isnan(complexity_corr):
    if complexity_corr > 0.3:
        print(f"   → Higher diversity leads to more complex visual patterns")
    elif complexity_corr < -0.3:
        print(f"   → Higher diversity leads to simpler visual patterns")
    else:
        print(f"   → Weak correlation between diversity and visual complexity")

print("\n" + "="*70)
print("RECOMMENDATIONS FOR NEXT EXPERIMENTS")
print("="*70)

# Identify promising regions
high_diversity = df[df['species_entropy_mean'] > df['species_entropy_mean'].quantile(0.75)]
print(f"\nHigh diversity configurations (top 25%):")
print(f"  mutation_prob range: {high_diversity['mutation_prob'].min():.4f} - {high_diversity['mutation_prob'].max():.4f}")
print(f"  max_copy_dist range: {high_diversity['max_copy_dist'].min():.2f} - {high_diversity['max_copy_dist'].max():.2f}")
print(f"  Preferred species_dim: {high_diversity['species_dim'].mode().values[0] if len(high_diversity) > 0 else 'N/A'}")

# Suggest next sweep parameters
print(f"\nSuggested next sweep (refine around optimal region):")
if len(high_diversity) > 0:
    mut_center = high_diversity['mutation_prob'].median()
    copy_center = high_diversity['max_copy_dist'].median()
    print(f"  mutation_prob: [{mut_center*0.5:.4f}, {mut_center:.4f}, {mut_center*1.5:.4f}]")
    print(f"  max_copy_dist: [{copy_center*0.7:.2f}, {copy_center:.2f}, {copy_center*1.3:.2f}]")
    print(f"  Consider testing larger systems (2000-4000 particles) with these parameters")

print("\n" + "="*70)
