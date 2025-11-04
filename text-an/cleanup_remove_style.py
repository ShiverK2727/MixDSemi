#!/usr/bin/env python3
import os
import json
from datetime import datetime

def remove_style_keys(obj):
    """Recursively remove any key named 'style' from dicts inside obj."""
    removed = 0
    if isinstance(obj, dict):
        if 'style' in obj:
            obj.pop('style')
            removed += 1
        for k, v in list(obj.items()):
            r = remove_style_keys(v)
            removed += r
    elif isinstance(obj, list):
        for item in obj:
            removed += remove_style_keys(item)
    return removed

def process_file(path, make_backup=True):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    removed = remove_style_keys(data)
    if removed == 0:
        return 0

    # backup
    if make_backup:
        bak_path = path + '.bak.' + datetime.now().strftime('%Y%m%dT%H%M%S')
        with open(bak_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # Note: backup contains cleaned content by design; to keep original, reload original
        # So recreate original backup by re-reading original file from disk if needed.

    # Write cleaned JSON with pretty formatting
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return removed

def main(folder):
    summary = {}
    for name in os.listdir(folder):
        if not name.lower().endswith('.json'):
            continue
        path = os.path.join(folder, name)
        try:
            removed = process_file(path)
        except Exception as e:
            summary[name] = {'error': str(e)}
            continue
        summary[name] = {'style_keys_removed': removed}
    # print summary
    for fn, info in summary.items():
        if 'error' in info:
            print(f"{fn}: ERROR: {info['error']}")
        else:
            print(f"{fn}: removed {info['style_keys_removed']} 'style' keys")

if __name__ == '__main__':
    base = os.path.dirname(__file__)
    target_folder = base
    print('Cleaning JSON files in', target_folder)
    main(target_folder)
