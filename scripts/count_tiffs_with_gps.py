#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path

def has_gps_broad(file_path):
    cmd = [
        'docker', 'compose', 'run', '--rm', 'pipeline',
        'exiftool', str(file_path)
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        gps_lines = []
        for line in proc.stdout.splitlines():
            l = line.lower()
            if ('gps' in l or 'lat' in l or 'lon' in l):
                gps_lines.append(line)
        if gps_lines:
            return True, gps_lines
    except Exception:
        pass
    return False, []

def print_full_exif(file_path):
    print(f"\n--- Full EXIF for {file_path} ---")
    cmd = [
        'docker', 'compose', 'run', '--rm', 'pipeline',
        'exiftool', str(file_path)
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(proc.stdout)
    except Exception as e:
        print(f"Error reading EXIF: {e}")

def main(input_dir):
    tiff_files = list(Path(input_dir).glob('**/*.tif')) + list(Path(input_dir).glob('**/*.tiff'))
    total = len(tiff_files)
    no_gps_count = 0
    for idx, file_path in enumerate(tiff_files, 1):
        print(f"[{idx}/{total}] Checking: {file_path}", end='... ')
        has_gps, gps_lines = has_gps_broad(file_path)
        if has_gps:
            print("GPS/Lat/Lon found!")
            print("  Matched lines:")
            for l in gps_lines:
                print(f"    {l}")
        else:
            print("No GPS/Lat/Lon.")
            print_full_exif(file_path)
            no_gps_count += 1
            if no_gps_count >= 5:
                print(f"\nStopped after printing EXIF for 5 files without GPS/Lat/Lon info.")
                return

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/count_tiffs_with_gps.py <input_dir>")
        sys.exit(1)
    main(sys.argv[1]) 