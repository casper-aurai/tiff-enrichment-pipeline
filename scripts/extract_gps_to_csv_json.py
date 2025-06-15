#!/usr/bin/env python3
import sys
import subprocess
import json
import csv
from pathlib import Path
import os

# Helper to parse exiftool output
import re
def parse_gps_output(output):
    gps = {}
    for line in output.splitlines():
        if 'GPS Latitude' in line and 'Ref' not in line:
            gps['latitude'] = line.split(':', 1)[1].strip()
        elif 'GPS Longitude' in line and 'Ref' not in line:
            gps['longitude'] = line.split(':', 1)[1].strip()
        elif 'GPS Altitude' in line and 'Ref' not in line:
            gps['altitude'] = line.split(':', 1)[1].strip()
    return gps

def main(input_dir, csv_out, json_out):
    tiff_files = list(Path(input_dir).glob('**/*.tif')) + list(Path(input_dir).glob('**/*.tiff'))
    results = []
    for file_path in tiff_files:
        cmd = [
            'exiftool',
            '-gps:all',
            str(file_path)
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            gps = parse_gps_output(proc.stdout)
            results.append({
                'file': str(file_path),
                'latitude': gps.get('latitude'),
                'longitude': gps.get('longitude'),
                'altitude': gps.get('altitude')
            })
        except Exception as e:
            results.append({
                'file': str(file_path),
                'latitude': None,
                'longitude': None,
                'altitude': None,
                'error': str(e)
            })
    # Write CSV
    with open(csv_out, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['file', 'latitude', 'longitude', 'altitude'])
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, '') for k in ['file', 'latitude', 'longitude', 'altitude']})
    # Write JSON
    with open(json_out, 'w') as jsonfile:
        json.dump(results, jsonfile, indent=2)
    print(f"Wrote {len(results)} results to {csv_out} and {json_out}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/extract_gps_to_csv_json.py <input_dir> [csv_out] [json_out]")
        sys.exit(1)
    input_dir = sys.argv[1]
    csv_out = sys.argv[2] if len(sys.argv) > 2 else 'gps_results.csv'
    json_out = sys.argv[3] if len(sys.argv) > 3 else 'gps_results.json'
    main(input_dir, csv_out, json_out) 