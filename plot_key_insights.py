"""Create focused plots for the key insights from the sweep."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('low_mutation_sweep_results.csv')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

fig = plt.figure(figsize=(16, 10))

# 1. Main Finding: Copy Distance Sweet Spot
ax1 = plt.subplot(2, 3, 1)
copy_dist_stats = df.groupby('max_copy_dist').agg({
    'species_entropy_mean': ['mean', 'std'],
    'num_unique_species_mean': ['mean', 'std']
})

x = copy_dist_stats.index
y1 = copy_dist_stats[('species_entropy_mean', 'mean')]
err1 = copy_dist_stats[('species_entropy_mean', 'std')]

ax1_twin = ax1.twinx()
y2 = copy_dist_stats[('num_unique_species_mean', 'mean')]
err2 = copy_dist_stats[('num_unique_species_mean', 'std')]

line1 = ax1.errorbar(x, y1, yerr=err1, marker='o', color='#2E86AB',
                      linewidth=2, markersize=8, capsize=5, label='Species Entropy')
line2 = ax1_twin.errorbar(x, y2, yerr=err2, marker='s', color='#A23B72',
                            linewidth=2, markersize=8, capsize=5, label='Unique Species')

ax1.set_xlabel('Max Copy Distance', fontsize=12)
ax1.set_ylabel('Species Entropy', color='#2E86AB', fontsize=12)
ax1_twin.set_ylabel('Unique Species', color='#A23B72', fontsize=12)
ax1.tick_params(axis='y', labelcolor='#2E86AB')
ax1_twin.tick_params(axis='y', labelcolor='#A23B72')
ax1.set_title('Copy Distance Trade-off:\nSweet Spot at 0.06', fontsize=13, fontweight='bold')
ax1.axvline(0.06, color='green', linestyle='--', alpha=0.5, label='Optimum')
ax1.grid(True, alpha=0.3)

# Add legend combining both axes
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower left', fontsize=10)

# 2. Mutation Rate Plateau
ax2 = plt.subplot(2, 3, 2)
mut_stats = df.groupby('mutation_prob').agg({
    'species_entropy_mean': ['mean', 'std']
})

x = mut_stats.index
y = mut_stats[('species_entropy_mean', 'mean')]
err = mut_stats[('species_entropy_mean', 'std')]

ax2.errorbar(x, y, yerr=err, marker='o', color='#F18F01',
             linewidth=2, markersize=10, capsize=5)
ax2.axhline(y.mean(), color='red', linestyle='--', alpha=0.5,
            label=f'Mean = {y.mean():.3f}')
ax2.set_xlabel('Mutation Probability', fontsize=12)
ax2.set_ylabel('Species Entropy', fontsize=12)
ax2.set_title('Mutation Rate Plateau:\nStable Diversity Across Range',
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# 3. Complexity Paradox
ax3 = plt.subplot(2, 3, 3)
# Color by mutation rate
scatter = ax3.scatter(df['species_entropy_mean'],
                     df['normalized_complexity_mean'],
                     c=df['mutation_prob'],
                     s=100, alpha=0.7, cmap='viridis',
                     edgecolors='black', linewidth=0.5)

# Add regression line
mask = ~(df['species_entropy_mean'].isna() | df['normalized_complexity_mean'].isna())
if mask.sum() > 0:
    z = np.polyfit(df.loc[mask, 'species_entropy_mean'],
                   df.loc[mask, 'normalized_complexity_mean'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['species_entropy_mean'].min(),
                         df['species_entropy_mean'].max(), 100)
    ax3.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    corr = df[['species_entropy_mean', 'normalized_complexity_mean']].corr().iloc[0, 1]
    ax3.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax3.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax3.set_xlabel('Species Entropy (Diversity)', fontsize=12)
ax3.set_ylabel('Normalized Visual Complexity', fontsize=12)
ax3.set_title('Complexity Paradox:\nHigh Diversity = Low Visual Complexity',
              fontsize=13, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Mutation Prob', fontsize=10)
ax3.grid(True, alpha=0.3)

# 4. Species Dimensionality Effect
ax4 = plt.subplot(2, 3, 4)
species_dim_stats = df.groupby('species_dim').agg({
    'species_entropy_mean': ['mean', 'std'],
    'num_unique_species_mean': ['mean', 'std']
})

x = species_dim_stats.index
y1 = species_dim_stats[('species_entropy_mean', 'mean')]
err1 = species_dim_stats[('species_entropy_mean', 'std')]
y2 = species_dim_stats[('num_unique_species_mean', 'mean')]
err2 = species_dim_stats[('num_unique_species_mean', 'std')]

x_pos = np.arange(len(x))
width = 0.35

bars1 = ax4.bar(x_pos - width/2, y1, width, yerr=err1,
                label='Entropy', color='#2E86AB', capsize=5)
ax4_twin = ax4.twinx()
bars2 = ax4_twin.bar(x_pos + width/2, y2, width, yerr=err2,
                     label='Unique Species', color='#A23B72', capsize=5)

ax4.set_xlabel('Species Dimensionality', fontsize=12)
ax4.set_ylabel('Species Entropy', color='#2E86AB', fontsize=12)
ax4_twin.set_ylabel('Unique Species', color='#A23B72', fontsize=12)
ax4.set_xticks(x_pos)
ax4.set_xticklabels([f'{int(d)}D' for d in x])
ax4.tick_params(axis='y', labelcolor='#2E86AB')
ax4_twin.tick_params(axis='y', labelcolor='#A23B72')
ax4.set_title('Species Dimensionality:\n4D Optimal for Entropy',
              fontsize=13, fontweight='bold')
ax4.legend(loc='upper left', fontsize=10)
ax4_twin.legend(loc='upper right', fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

# 5. Copy Probability Effect
ax5 = plt.subplot(2, 3, 5)
for copy_prob in sorted(df['copy_prob'].unique()):
    subset = df[df['copy_prob'] == copy_prob]
    stats = subset.groupby('mutation_prob')['species_entropy_mean'].agg(['mean', 'std'])
    ax5.errorbar(stats.index, stats['mean'], yerr=stats['std'],
                 marker='o', label=f'copy_prob={copy_prob:.4f}',
                 linewidth=2, markersize=8, capsize=5)

ax5.set_xlabel('Mutation Probability', fontsize=12)
ax5.set_ylabel('Species Entropy', fontsize=12)
ax5.set_title('Copy Probability Effect:\nRare Copying (0.0005) Slightly Better',
              fontsize=13, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

# 6. Optimal Configuration Highlight
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Find optimal configuration
best_idx = df['species_entropy_mean'].idxmax()
best = df.loc[best_idx]

info_text = f"""
OPTIMAL CONFIGURATION

🏆 Highest Species Entropy: {best['species_entropy_mean']:.3f}

Parameters:
  • mutation_prob: {best['mutation_prob']:.4f}
  • max_copy_dist: {best['max_copy_dist']:.2f}
  • copy_prob: {best['copy_prob']:.4f}
  • species_dim: {int(best['species_dim'])}

Metrics:
  • Unique Species: {best['num_unique_species_mean']:.1f}
  • Pairwise Diversity: {best['species_pairwise_diversity_mean']:.3f}
  • Clustering Coeff: {best['clustering_coeff_mean']:.2f}
  • Normalized Complexity: {best['normalized_complexity_mean']:.3f}

Key Insight:
Very low mutation (0.001) + local copying (0.06)
+ rare copy events (0.0005) creates the most
diverse and stable ecosystems.

Next Steps:
1. Test larger systems (2000-4000 particles)
2. Refine around mutation=0.001, copy_dist=0.06
3. Investigate complexity-diversity inversion
"""

ax6.text(0.1, 0.95, info_text, transform=ax6.transAxes,
         fontsize=11, verticalalignment='top',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.suptitle('Low Mutation Sweep: Key Insights & Optimal Configuration',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('key_insights.png', dpi=150, bbox_inches='tight')
print("✓ Saved key insights plot to key_insights.png")
