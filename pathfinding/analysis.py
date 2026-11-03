"""Statistics and charts for the benchmark results.

Handling of unsuccessful runs
-----------------------------
* ``success_rate`` = successful runs / all runs in the group.
* Mean and standard deviation of runtime, nodes expanded AND path cost are
  taken over SUCCESSFUL runs only, so all three metrics describe the same set
  of trials and the algorithms are compared on identical instances.
* A failed run has cost=None, never infinity, so it cannot enter a cost mean.
* Standard deviation is the SAMPLE standard deviation (n - 1); it is NaN when
  fewer than two successful runs exist.
"""
import csv
import math
import os
import statistics

METRICS = ('runtime_ms', 'nodes_expanded', 'cost')
GROUP_KEYS = ('rows', 'cols', 'density', 'terrain_fraction', 'movement',
              'algorithm', 'heuristic')
SUMMARY_FIELDS = (list(GROUP_KEYS) + ['n_trials', 'n_found', 'success_rate']
                  + [f'{m}_{stat}' for m in METRICS for stat in ('mean', 'std')]
                  + ['nodes_vs_dijkstra'])
COST_TOLERANCE = 1e-9


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------
def mean_std(values):
    """(mean, sample std). Empty -> (nan, nan); single value -> (v, nan)."""
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), math.nan
    return statistics.mean(values), statistics.stdev(values)


def summarise_group(rows):
    """Summary statistics for rows that share one condition."""
    n_trials = len(rows)
    found_rows = [r for r in rows if r['found']]
    summary = {'n_trials': n_trials, 'n_found': len(found_rows),
               'success_rate': len(found_rows) / n_trials if n_trials else math.nan}
    for metric in METRICS:
        mean, std = mean_std([r[metric] for r in found_rows])
        summary[f'{metric}_mean'] = mean
        summary[f'{metric}_std'] = std
    return summary


def summarise(rows):
    """One summary row per (size, density, terrain, movement, algorithm, heuristic)."""
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in GROUP_KEYS), []).append(row)
    summaries = []
    for key, group_rows in groups.items():
        summary = dict(zip(GROUP_KEYS, key))
        summary.update(summarise_group(group_rows))
        summaries.append(summary)
    # Nodes expanded relative to Dijkstra in the same condition (1.0 = same work).
    baseline = {(s['rows'], s['cols'], s['density'], s['terrain_fraction'],
                 s['movement']): s['nodes_expanded_mean']
                for s in summaries if s['algorithm'] == 'dijkstra'}
    for s in summaries:
        ref = baseline.get((s['rows'], s['cols'], s['density'],
                            s['terrain_fraction'], s['movement']))
        s['nodes_vs_dijkstra'] = (s['nodes_expanded_mean'] / ref
                                  if ref and math.isfinite(ref) and ref > 0
                                  and math.isfinite(s['nodes_expanded_mean']) else math.nan)
    summaries.sort(key=lambda s: (s['rows'], s['cols'], s['density'],
                                  s['terrain_fraction'], s['movement'],
                                  s['algorithm'] != 'dijkstra', s['heuristic']))
    return summaries


def check_consistency(rows):
    """Validity checks across algorithms; returns a list of problem descriptions.

    1. found must equal the independent BFS ``solvable`` flag.
    2. Every A* run with an admissible heuristic must match Dijkstra's cost.
    Both should return an empty list; anything else means a bug or an
    inadmissible heuristic and must be investigated before trusting results.
    """
    problems = []
    for r in rows:
        where = (f"seed={r['seed']} {r['rows']}x{r['cols']} d={r['density']} "
                 f"move={r['movement']} {r['algorithm']}-{r['heuristic']}")
        if r['solvable'] is not None and r['found'] != r['solvable']:
            problems.append(f'found={r["found"]} but BFS solvable={r["solvable"]} ({where})')
        if r['optimal'] is False:
            problems.append(f'cost differs from Dijkstra ({where})')
    return problems


# --------------------------------------------------------------------------
# CSV I/O
# --------------------------------------------------------------------------
def _to_bool(text):
    return {'True': True, 'False': False}.get(text)


def _to_float(text):
    return float(text) if text not in ('', None) else None


def load_raw_csv(path):
    """Read raw_results.csv back into typed rows (empty cost -> None)."""
    rows = []
    with open(path, newline='') as handle:
        for r in csv.DictReader(handle):
            rows.append({
                'seed': int(r['seed']), 'trial': int(r['trial']),
                'rows': int(r['rows']), 'cols': int(r['cols']),
                'density': float(r['density']),
                'terrain_fraction': float(r['terrain_fraction']),
                'n_walls': int(r['n_walls']), 'movement': int(r['movement']),
                'algorithm': r['algorithm'], 'heuristic': r['heuristic'],
                'solvable': _to_bool(r['solvable']), 'found': _to_bool(r['found']),
                'cost': _to_float(r['cost']),
                'nodes_expanded': int(r['nodes_expanded']),
                'runtime_ms': float(r['runtime_ms']),
                'optimal': _to_bool(r['optimal']), 'grid_hash': r['grid_hash'],
            })
    return rows


def write_summary_csv(summaries, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: ('' if isinstance(s[k], float) and math.isnan(s[k])
                                 else s[k]) for k in SUMMARY_FIELDS})


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
CHART_METRICS = {
    'runtime_ms': ('Mean runtime (ms)', 'runtime'),
    'nodes_expanded': ('Mean nodes expanded', 'nodes_expanded'),
    'cost': ('Mean path cost', 'path_cost'),
}


def _condition_label(s):
    label = f"{s['rows']}x{s['cols']}\ndensity {s['density']:.2f}"
    if s['terrain_fraction'] > 0:
        label += f"\nterrain {s['terrain_fraction']:.2f}"
    return label


def _series_label(s):
    return 'dijkstra' if s['algorithm'] == 'dijkstra' else f"A* {s['heuristic']}"


def make_charts(summaries, output_dir):
    """Grouped bar charts (mean +/- sample std) per metric and movement mode.

    Returns the list of PNG paths written. Error bars are clipped at zero on
    the lower side because none of these quantities can be negative.
    """
    import matplotlib
    matplotlib.use('Agg')  # no display needed; works on servers and in tests
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for movement in sorted({s['movement'] for s in summaries}):
        subset = [s for s in summaries if s['movement'] == movement]
        conditions = []
        for s in subset:
            key = (s['rows'], s['cols'], s['density'], s['terrain_fraction'])
            if key not in conditions:
                conditions.append(key)
        series_names = []
        for s in subset:
            if _series_label(s) not in series_names:
                series_names.append(_series_label(s))
        for metric, (title, stem) in CHART_METRICS.items():
            fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(conditions) + 2), 4.5))
            width = 0.8 / len(series_names)
            for i, name in enumerate(series_names):
                means, errs_low, errs_high = [], [], []
                for key in conditions:
                    match = [s for s in subset if _series_label(s) == name
                             and (s['rows'], s['cols'], s['density'],
                                  s['terrain_fraction']) == key]
                    mean = match[0][f'{metric}_mean'] if match else math.nan
                    std = match[0][f'{metric}_std'] if match else math.nan
                    # Missing successful-run data must not look like a measured zero.
                    std = 0.0 if math.isnan(std) else std      # n < 2
                    means.append(mean)
                    errs_low.append(min(std, mean) if math.isfinite(mean) else 0)
                    errs_high.append(std)
                xs = [j + (i - (len(series_names) - 1) / 2) * width
                      for j in range(len(conditions))]
                ax.bar(xs, means, width, yerr=[errs_low, errs_high], capsize=3,
                       label=name)
            labels = [_condition_label(next(s for s in subset
                      if (s['rows'], s['cols'], s['density'], s['terrain_fraction']) == key))
                      for key in conditions]
            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel(title)
            ax.set_title(f'{title} - {movement}-way movement (error bars: 1 sample std)')
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            fig.tight_layout()
            path = os.path.join(output_dir, f'{stem}_{movement}way.png')
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(path)
    return paths


# --------------------------------------------------------------------------
# Orchestration + CLI
# --------------------------------------------------------------------------
def run_analysis(rows, output_dir):
    """Write summary_stats.csv and the PNG charts; print any validity problems."""
    summaries = summarise(rows)
    write_summary_csv(summaries, os.path.join(output_dir, 'summary_stats.csv'))
    charts = make_charts(summaries, output_dir)
    problems = check_consistency(rows)
    print(f'Wrote summary_stats.csv and {len(charts)} charts to {output_dir}')
    if problems:
        print(f'WARNING: {len(problems)} consistency problem(s); first 5:')
        for p in problems[:5]:
            print('  -', p)
    else:
        print('Consistency checks passed: found == BFS solvable and every A* cost '
              'equals Dijkstra\'s on every instance.')
    return summaries, charts, problems


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description='Re-create summary CSV and charts from raw_results.csv')
    p.add_argument('--raw', default='results/raw_results.csv')
    p.add_argument('--output-dir', default='results')
    args = p.parse_args(argv)
    run_analysis(load_raw_csv(args.raw), args.output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
