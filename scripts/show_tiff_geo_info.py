#!/usr/bin/env python3
import sys
from pathlib import Path
import tifffile

# Recursively print all tags and sub-IFDs
def print_tags(tags, prefix='  '):
    for tag in tags.values():
        try:
            value = tag.value
            if isinstance(value, dict):
                print(f"{prefix}{tag.name}: (sub-IFD)")
                print_tags(value, prefix + '    ')
            else:
                print(f"{prefix}{tag.name}: {value}")
        except Exception as e:
            print(f"{prefix}{tag.name}: <error: {e}>")

def main(input_dir, max_files=5):
    tiff_files = list(Path(input_dir).glob('**/*.tif')) + list(Path(input_dir).glob('**/*.tiff'))
    print(f"Found {len(tiff_files)} TIFF files in {input_dir}")
    for i, file_path in enumerate(tiff_files[:max_files]):
        print(f"\nFile {i+1}: {file_path}")
        try:
            with tifffile.TiffFile(file_path) as tif:
                for page_num, page in enumerate(tif.pages):
                    print(f"  Page {page_num}:")
                    print_tags(page.tags, prefix='    ')
        except Exception as e:
            print(f"  <error reading file: {e}>")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/show_tiff_geo_info.py <input_dir> [max_files]")
        sys.exit(1)
    input_dir = sys.argv[1]
    max_files = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    main(input_dir, max_files) 