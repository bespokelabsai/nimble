"""Render the README's fixed 324-example comparison from aggregate results."""
import argparse
import json
import os
from pathlib import Path

from nimble.paths import PROJECT_ROOT

os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / '.cache/matplotlib'))
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter

# Bespoke Labs brand colors; MUTED is BLACK at 70% opacity.
TEAL, TAN, RED, BLUE, BLACK = '#0F484D', '#F8F3E6', '#F44029', '#B9C8DD', '#231F20'
MUTED = '#231F20B3'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path,
                        default=PROJECT_ROOT / 'assets/evidence-324-results.json')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'assets')
    args = parser.parse_args()
    result = json.loads(args.results.read_text())
    models = sorted(result['models'], key=lambda r: r['correct'], reverse=True)
    assert all(r['count'] == 324 and 0 <= r['correct'] <= r['count'] for r in models)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=TAN)
    ax.set_facecolor(TAN)
    fig.subplots_adjust(left=.085, right=.98, top=.76, bottom=.13)
    colors = [RED if r['name'] == 'Bespoke-Nimble-9B' else BLUE for r in models]
    values = [100 * r['correct'] / r['count'] for r in models]
    ax.bar(range(len(models)), values, width=.62, color=colors, zorder=3)
    labels = {'Bespoke-Nimble-9B': 'Bespoke-\nNimble-9B', 'Gemma 3 270M IT': 'Gemma 3\n270M IT'}
    ax.set_xticks(range(len(models)), [labels.get(r['name'], r['name'].replace('-', '\n')) for r in models])
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_axisbelow(True)
    ax.grid(axis='y', color=BLUE, alpha=.5, linewidth=.8)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color(BLUE)
    ax.tick_params(axis='both', length=0, pad=10, labelcolor=BLACK)
    ax.set_ylabel('Reference-label agreement', labelpad=12, color=MUTED)
    for i, (r, value) in enumerate(zip(models, values)):
        weight = 'bold' if r['name'] == 'Bespoke-Nimble-9B' else 'normal'
        ax.text(i, value + 2, f"{value:.2f}%\n{r['correct']}/{r['count']}",
                ha='center', va='bottom', fontsize=10, color=BLACK, weight=weight)
        ax.get_xticklabels()[i].set_weight(weight)
    # The README heading already names the 324 held-out examples.
    fig.text(.045, .94, 'Model comparison', fontsize=23, weight='bold', color=TEAL)
    fig.text(.045, .885, 'Same examples and labels for every model · higher is better',
             fontsize=12, color=MUTED)
    # Red is too light for small text, so a swatch ties this note to the red bar.
    fig.add_artist(Rectangle((.045, .823), .11 / 12, .11 / 7, transform=fig.transFigure, color=RED))
    fig.text(.06, .823, 'Bespoke-Nimble-9B: fine-tuned on 2,676 examples',
             fontsize=11, color=BLACK, weight='bold')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(args.output_dir / f'evidence-324-comparison.{ext}', dpi=180, facecolor=TAN)
    plt.close(fig)


if __name__ == '__main__':
    main()
