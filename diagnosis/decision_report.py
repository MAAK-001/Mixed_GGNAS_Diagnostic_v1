from __future__ import annotations

import argparse
from pathlib import Path
import json
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run_dir', type=Path)
    a=p.parse_args()
    run=a.run_dir
    corr_d=pd.read_csv(run/'correlations_val_dice.csv')
    corr_m=pd.read_csv(run/'correlations_val_miou.csv')
    scale=pd.read_csv(run/'scale_ablation.csv') if (run/'scale_ablation.csv').exists() else None
    sanity=json.loads((run/'sanity_checks.json').read_text())
    lines=[]
    lines.append('# Diagnostic decision report')
    lines.append('')
    lines.append('This report is generated from measured results only. It does not assume that a proxy is useful because the literature reported it useful elsewhere.')
    lines.append('')
    over=sanity['tiny_real_data_overfit']
    lines.append(f"## 1. Optimization sanity: best tiny-real-data Dice = **{over['dice']:.4f}** (mIoU {over['miou']:.4f}) at epoch {over['epoch']}.")
    if over['dice'] < .90:
        lines.append('**Action: investigate training/model construction before trusting NAS results.** The model cannot easily fit eight real training examples, so a proxy change is premature.')
    else:
        lines.append('Optimization sanity is healthy enough to proceed to proxy diagnosis.')
    lines.append('')
    lines.append('## 2. Proxy ranking')
    lines.append('')
    lines.append('### Validation Dice')
    lines.append(corr_d[['proxy','spearman','spearman_ci_low','spearman_ci_high','kendall','pearson']].to_markdown(index=False))
    lines.append('')
    lines.append('### Validation mIoU')
    lines.append(corr_m[['proxy','spearman','spearman_ci_low','spearman_ci_high','kendall','pearson']].to_markdown(index=False))
    lines.append('')
    best_d=corr_d.iloc[0] if len(corr_d) else None
    best_m=corr_m.iloc[0] if len(corr_m) else None
    if best_d is not None:
        lines.append(f"Best measured Dice rank correlation: **{best_d['proxy']}**, Spearman {best_d['spearman']:.3f}, Kendall {best_d['kendall']:.3f}.")
    if best_m is not None:
        lines.append(f"Best measured mIoU rank correlation: **{best_m['proxy']}**, Spearman {best_m['spearman']:.3f}, Kendall {best_m['kendall']:.3f}.")
    lines.append('')
    lines.append('## 3. Scale decision')
    if scale is not None:
        g=scale.groupby('mode')[['val_dice','val_miou']].mean().sort_values('val_dice', ascending=False)
        lines.append(g.to_markdown())
        all_mean=float(scale.loc[scale.mode=='all_three_scales','val_dice'].mean())
        cur=scale.loc[scale.mode=='current_10epoch_argmax_collapse','val_dice']
        cur_mean=float(cur.mean()) if len(cur) else float('nan')
        if cur_mean < all_mean:
            lines.append(f"**Recommendation: retain all three scales during final training** based on the matched ablation (mean Dice {all_mean:.4f} vs current-collapse {cur_mean:.4f}).")
        else:
            lines.append(f"Hard collapse is not worse in this sample (mean Dice {cur_mean:.4f} vs all-scale {all_mean:.4f}); inspect per-architecture variance before changing the pipeline.")
    lines.append('')
    lines.append('## 4. Research decision rule')
    lines.append('- If fixed baselines and tiny-real-data overfit are weak: fix training/data/model first.')
    lines.append('- If training is healthy but all proxies have weak or unstable rank correlation: the search is proxy-limited; do not run NSGA-II with the same proxy.')
    lines.append('- If a paper-faithful proxy beats the current heuristic consistently: replace the heuristic with that proxy and validate again before a large search.')
    lines.append('- If a segmentation-specific proxy wins: use it as the third objective, but keep the measured correlation report as evidence.')
    lines.append('- If parameters/FLOPs beat every sophisticated proxy: the current zero-cost objective is not useful enough for this search space.')
    (run/'RESEARCH_DECISION.md').write_text('\n'.join(lines), encoding='utf-8')
    print(run/'RESEARCH_DECISION.md')

if __name__=='__main__': main()
