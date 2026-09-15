#!/usr/bin/env python3
"""Export scalar benchmark results as PNG/PDF; no image inference is performed."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_li2026 import NAMES, ORDER


def plot(root):
    report = json.loads((root / 'comparison.json').read_text())
    settings = [('uhd240', 'cosited_point', 'UHD240 · point sampling', 240),
                ('kodak', 'cosited_point', 'Kodak · point sampling', 24),
                ('kodak', 'center_box', 'Kodak · centered box', 24)]
    panels = []
    for dataset, sampling, title, expected in settings:
        values = {row['method']: row['cpsnr_rgb'] for row in report['results']
                  if row['dataset'] == dataset and row['sampling'] == sampling and row['images'] == expected}
        if all(name in values for name in ORDER):
            panels.append((title, values))
    if not panels:
        raise ValueError('No complete dataset/sampling group to plot')
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels) + 1.4, 5.6),
                             sharey=True, squeeze=False)
    lower, upper = 0., 0.
    for _, values in panels:
        baseline = values['conventional/bilinear']
        lower = min(lower, *(values[name] - baseline for name in ORDER))
        upper = max(upper, *(score - baseline for name, score in values.items() if name.startswith('scaled_')),
                    *(values[name] - baseline for name in ORDER))
    for ax, (title, values) in zip(axes[0], panels):
        baseline = values['conventional/bilinear']
        best = max((name for name in values if name.startswith('scaled_')), key=values.get)
        methods = ORDER + [best]
        gains = [values[name] - baseline for name in methods]
        colors = ['#17689a' if gain >= 0 else '#b66a28' for gain in gains[:-1]] + ['#555555']
        ax.barh(range(len(methods)), gains, color=colors, height=.65)
        ax.axvline(0, color='#333333', linewidth=.8)
        for i, gain in enumerate(gains):
            inside = gain < -1.5
            ax.text(gain + (.12 if gain >= 0 or inside else -.12), i, '%+.2f' % gain,
                    ha='left' if gain >= 0 or inside else 'right', va='center', fontsize=9,
                    color='white' if inside else '#222222')
        comparator = best.replace('scaled_decode_first/', 'decoded sites + ').replace('scaled_matrix/', 'matrix + ')
        ax.set_title(title + '\nBilinear: %.2f dB\nScaled: ' % baseline + comparator, fontsize=10, pad=12)
        ax.set_yticks(range(len(methods)), [NAMES[n] for n in ORDER] + ['Best independent\nscaled comparator'])
        ax.tick_params(axis='y', length=0)
        ax.set_xlim(lower - 1.3, upper + 1.3)
        ax.set_xlabel('RGB CPSNR gain over bilinear (dB)')
        ax.grid(axis='x', color='#dddddd', linewidth=.6)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right', 'left']].set_visible(False)
    axes[0, 0].invert_yaxis()
    fig.suptitle('Frozen models on the Li et al. benchmark datasets', fontsize=14, y=.98)
    fig.text(.02, .025, 'Native images; per-image RGB CPSNR averaged. Centered box is a sensitivity check.\n'
             'Independent implementation: exact reproduction of the published image protocol remains unverified.',
             fontsize=9, color='#444444')
    fig.tight_layout(rect=(0, .12, 1, .91))
    for extension in ('png', 'pdf'):
        fig.savefig(root / ('comparison_plot.' + extension), dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    plot(parser.parse_args().root)
