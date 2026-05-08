"""
Generates JSON label files from .osu beatmap files for use by spectrogram.py.

For each song folder in --audio_dir, finds the .osu file whose Version: field
contains --diff (case-insensitive), runs parse_osu on it, and saves the result
as --out_dir/<folder_name>.json. Skips folders with no matching difficulty.

Usage:
    python data/src/make_labels.py \\
        --audio_dir data/tracks \\
        --out_dir data/labels/insane \\
        --diff insane

Arguments:
    --audio_dir (str): Directory containing song folders (each with .osu files).
    --out_dir (str):   Directory to write JSON label files into (created if missing).
    --diff (str):      Difficulty substring to match against the Version: field (case-insensitive).
"""

import argparse
import json
import os

from parse_osu import parse_osu


def find_osu_files(folder: str) -> list[str]:
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".osu")
    ]


def make_labels(audio_dir: str, out_dir: str, diff: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    song_folders = sorted([
        os.path.join(audio_dir, name)
        for name in os.listdir(audio_dir)
        if os.path.isdir(os.path.join(audio_dir, name))
    ])

    ok, skipped_no_diff, skipped_wrong_mode, failed = 0, 0, 0, 0

    for folder in song_folders:
        folder_name = os.path.basename(folder)
        osu_files = find_osu_files(folder)

        match = None
        for path in osu_files:
            try:
                result = parse_osu(path)
            except ValueError:
                # wrong game mode
                continue
            except Exception as e:
                print(f"FAIL  {folder_name}: {e}")
                failed += 1
                break
            if diff.lower() in result["version"].lower():
                match = result
                break

        if match is None:
            skipped_no_diff += 1
            continue

        out_path = os.path.join(out_dir, f"{folder_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(match, f)

        print(f"OK    {folder_name} [{match['version']}]  {len(match['hit_objects'])} objects")
        ok += 1

    print(f"\n{ok} written, {skipped_no_diff} skipped (no matching diff), {failed} failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate JSON label files from .osu files, filtered by difficulty."
    )
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--diff", type=str, required=True,
                        help="Difficulty substring to match (e.g. 'insane', 'hard')")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_labels(args.audio_dir, args.out_dir, args.diff)
