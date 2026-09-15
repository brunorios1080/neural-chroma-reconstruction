#!/usr/bin/env python3
"""Collect complete image-level results without conflating reproduced/reported scores."""
import csv
import json
from pathlib import Path
import sys


def collect(root):
    shard_plan = root / 'sharding.json'
    if shard_plan.exists():
        count = json.loads(shard_plan.read_text())['shard_count']
        paths = [root / 'groups' / 'kodak'] + [root / 'shards' / str(i) for i in range(count)]
    else:
        paths = [root / 'groups' / name for name in ('kodak', 'uhd240_4k', 'uhd240_6k', 'uhd240_8k')]
    rows, complete = [], []
    for path in paths:
        report = path / 'report.json'
        if not report.exists():
            continue
        metadata = json.loads(report.read_text())
        if metadata['status'] != 'complete':
            continue
        complete.append(path.name)
        rows.extend(json.loads(line) for line in (path / 'rows.jsonl').read_text().splitlines())
    groups = {}
    for row in rows:
        key = (row['dataset'], row['sampling'], row['method'])
        groups.setdefault(key, []).append(row)
    results = []
    for (dataset, sampling, method), entries in sorted(groups.items()):
        if len({row['id'] for row in entries}) != len(entries):
            raise ValueError('Duplicate image results')
        results.append({'dataset': dataset, 'sampling': sampling, 'method': method, 'images': len(entries),
                        **{key: sum(row['metrics'][key] for row in entries)/len(entries)
                           for key in entries[0]['metrics']}})
    # Per-image pairing: only compare methods that evaluated identical IDs.
    paired = []
    for (dataset, sampling, method), entries in sorted(groups.items()):
        if '/' in method:
            continue
        values = {row['id']: row['metrics']['cpsnr_rgb'] for row in entries}
        baselines = ['conventional/bilinear'] + [transform + '/' + interpolation
            for transform in ('scaled_matrix', 'scaled_hybrid', 'scaled_lifting', 'scaled_decode_first')
            for interpolation in ('bilinear', 'bicubic')]
        for baseline in baselines:
            other = groups.get((dataset, sampling, baseline), [])
            reference = {row['id']: row['metrics']['cpsnr_rgb'] for row in other}
            if set(values) != set(reference):
                continue
            differences = [values[key] - reference[key] for key in sorted(values)]
            paired.append({'dataset': dataset, 'sampling': sampling, 'model': method,
                           'comparator': baseline, 'images': len(differences),
                           'mean_cpsnr_gain_db': sum(differences)/len(differences),
                           'win_fraction': sum(value > 0 for value in differences)/len(differences)})
    all_complete = len(complete) == len(paths)
    if all_complete:
        for result in results:
            expected = 240 if result['dataset'] == 'uhd240' else 24
            if result['images'] != expected:
                raise ValueError('Completed campaign has missing or extra images: ' + str(result))
    report = {'status': 'complete' if all_complete else 'partial', 'completed_groups': complete,
              'results': results, 'paired_comparisons': paired,
              'exact_published_protocol_verified': False,
              'interpretation': 'Paired comparisons are against independent implementations on shared inputs. '
                  'They do not establish superiority over the published method until its reported baselines '
                  'and unspecified implementation details are independently reproduced.'}
    # Each group has its own collection file; the last completed job writes the
    # complete aggregate atomically. Unique temp names avoid concurrent writers.
    import os
    temporary = root / ('comparison.%d.tmp' % os.getpid())
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(root / 'comparison.json')
    if all_complete:
        with (root / 'comparison.csv').open('w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    print(json.dumps({'status': report['status'], 'completed_groups': complete}))


if __name__ == '__main__':
    import fcntl
    root = Path(sys.argv[1])
    with (root / '.collect.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        collect(root)
