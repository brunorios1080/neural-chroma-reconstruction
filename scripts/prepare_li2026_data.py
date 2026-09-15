#!/usr/bin/env python3
"""Download the public Li et al. (2026) evaluation data, preserving provenance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile


def download(url, path, size=None, md5=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(path.suffix + '.partial')
        for attempt in range(4):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                headers = {'User-Agent': 'neural-chroma-reconstruction/research-reproduction'}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                    append = offset and response.status == 206
                    with partial.open('ab' if append else 'wb') as output:
                        while True:
                            chunk = response.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
                if size is not None and partial.stat().st_size != size:
                    raise ValueError(f'Incomplete download: {partial}')
                partial.replace(path)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    hashes = {key: hashlib.new(key) for key in ('md5', 'sha256')}
    with path.open('rb') as source:
        while True:
            chunk = source.read(8 * 1024 * 1024)
            if not chunk:
                break
            for digest in hashes.values():
                digest.update(chunk)
    result = {'path': str(path), 'url': url, 'bytes': path.stat().st_size,
              **{key: value.hexdigest() for key, value in hashes.items()}}
    if size is not None and result['bytes'] != size:
        raise ValueError(f'Size mismatch: {path}')
    if md5 is not None and result['md5'] != md5:
        raise ValueError(f'MD5 mismatch: {path}')
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f'Bad ZIP CRC: {bad}')
            result['members'] = [{'name': item.filename, 'size': item.file_size}
                                 for item in archive.infolist() if not item.is_dir()]
    print(json.dumps({key: value for key, value in result.items() if key != 'members'}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen('https://zenodo.org/api/records/17649711', timeout=60) as response:
        metadata = json.load(response)
    (args.root / 'zenodo_17649711.json').write_text(json.dumps(metadata, indent=2) + '\n')
    # Kodak is small and completes first, allowing a baseline reproduction check.
    with ThreadPoolExecutor(max_workers=3) as pool:
        kodak = list(pool.map(lambda i: download(
            f'https://r0k.us/graphics/kodak/kodak/kodim{i:02d}.png',
            args.root / 'kodak' / f'kodim{i:02d}.png'), range(1, 25)))
    (args.root / 'kodak_downloads.json').write_text(json.dumps(kodak, indent=2) + '\n')
    for entry in metadata['files']:
        result = download(entry['links']['self'], args.root / 'uhd240' / entry['key'],
                          entry['size'], entry['checksum'].split(':', 1)[1])
        (args.root / (entry['key'] + '.provenance.json')).write_text(json.dumps(result, indent=2) + '\n')
    (args.root / 'downloads_complete.json').write_text(json.dumps({'status': 'complete',
        'datasets': {'kodak': 24, 'uhd240': 240}}, indent=2) + '\n')


if __name__ == '__main__':
    main()
