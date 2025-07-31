#!/usr/bin/env python3
import os
import re
import glob
import argparse
from PIL import Image

def create_quilt(folder_path, pattern, regex, columns, rows, output_filename):
    # Find and sort all files matching the glob pattern numerically
    pattern_path = os.path.join(folder_path, pattern)
    files = glob.glob(pattern_path)
    files.sort(key=lambda path: int(re.search(regex, os.path.basename(path)).group(1)))

    # Verify count matches grid size
    if len(files) != columns * rows:
        print(f"Error: Found {len(files)} images but expected {columns * rows} (columns×rows).")
        return

    # Determine tile size and image mode from first file
    with Image.open(files[0]) as first:
        tile_width, tile_height = first.size
        mode = first.mode

    # Create a blank quilt image
    quilt_width = columns * tile_width
    quilt_height = rows * tile_height
    quilt = Image.new(mode, (quilt_width, quilt_height))

    # Paste each file into the correct spot (view 0 is bottom-left)
    for i, file_path in enumerate(files):
        col = i % columns
        row = rows - 1 - (i // columns)
        x = col * tile_width
        y = row * tile_height
        with Image.open(file_path) as tile:
            quilt.paste(tile, (x, y))

    # Save the resulting quilt
    output_path = os.path.join(folder_path, output_filename)
    quilt.save(output_path)
    print(f"Quilt saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a quilt image from a series of image files.')
    parser.add_argument('folder_path', help='Path to the folder containing image files.')
    parser.add_argument('--pattern', default='frame_*.png', help='Glob pattern for image filenames.')
    parser.add_argument('--regex', default=r'(\d+)', help='Regex to extract numeric index from filenames.')
    parser.add_argument('--columns', type=int, required=True, help='Number of columns in the quilt.')
    parser.add_argument('--rows', type=int, required=True, help='Number of rows in the quilt.')
    parser.add_argument('--output', default='quilt.png', help='Output filename for the quilt.')
    args = parser.parse_args()

    create_quilt(args.folder_path, args.pattern, args.regex, args.columns, args.rows, args.output)
