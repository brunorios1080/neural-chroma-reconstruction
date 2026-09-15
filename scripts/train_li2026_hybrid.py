#!/usr/bin/env python3
"""Run the isolated paper-inspired hybrid pilot, or resume completed epochs."""
import argparse
import json
from pathlib import Path

from chroma.li2026_hybrid_training import run_case, select_records
from chroma.prism_training import atomic_json
from chroma.research_data import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--case', help='Optional single named case')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    campaign = json.loads(Path(config['campaign']).read_text())
    entries = {entry['name']: entry for entry in campaign['models']}
    records = select_records(config['manifest'], config['training'])
    cases = [case for case in config['cases'] if args.case is None or case['name'] == args.case]
    if not cases or len({c['name'] for c in cases}) != len(cases):
        raise ValueError('Expected unique, existing case names')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for case in cases:
        if Path(case['name']).name != case['name'] or case['name'] in {'.', '..'}:
            raise ValueError('Unsafe case name')
        entry = entries[case['model']]
        results[case['name']] = run_case(case, config['training'], entry, records,
                                         config['source'], output/case['name'], resume=args.resume)
    atomic_json(output / ('summary_'+args.case+'.json' if args.case else 'summary.json'),
                {'config_sha256': sha256_file(args.config),
                 'original_manifest_sha256': sha256_file(config['manifest']), 'cases': results})


if __name__ == '__main__':
    main()
