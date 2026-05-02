"""
Delete song folders from data/tracks/ that have no matching JSON label in any difficulty tier.
Folders are matched by base name against data/labels/{easy,normal,hard,insane}/<name>.json.

Usage:
    python data/src/cleanup_unlabelled.py --tracks_dir data/tracks --labels_dir data/labels
    python data/src/cleanup_unlabelled.py --tracks_dir data/tracks --labels_dir data/labels --execute

    Without --execute: dry run (prints what would be deleted, deletes nothing).
    With    --execute: actually deletes the folders.
"""

import argparse
import os
import shutil


DIFFICULTIES = ["easy", "normal", "hard", "insane"]


def get_labelled_names(labels_dir: str) -> set[str]:
    labelled = set()
    for diff in DIFFICULTIES:
        diff_dir = os.path.join(labels_dir, diff)
        if not os.path.isdir(diff_dir):
            continue
        for fname in os.listdir(diff_dir):
            if fname.endswith(".json"):
                labelled.add(fname[:-5])  # strip .json
    return labelled


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete unlabelled song folders from tracks dir.")
    parser.add_argument("--tracks_dir", type=str, default="data/tracks")
    parser.add_argument("--labels_dir", type=str, default="data/labels")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete folders. Without this flag, only prints what would be deleted.",
    )
    args = parser.parse_args()

    labelled = get_labelled_names(args.labels_dir)
    print(f"Found {len(labelled)} labelled songs across all difficulties.")

    if not os.path.isdir(args.tracks_dir):
        raise RuntimeError(f"tracks_dir not found: {args.tracks_dir}")

    folders = [
        f for f in os.listdir(args.tracks_dir)
        if os.path.isdir(os.path.join(args.tracks_dir, f))
    ]
    print(f"Found {len(folders)} folders in {args.tracks_dir}.")

    to_delete = [f for f in folders if f not in labelled]
    to_keep = len(folders) - len(to_delete)

    print(f"\n{len(to_delete)} unlabelled folders {'will be' if args.execute else 'would be'} deleted, {to_keep} kept.\n")

    for folder in sorted(to_delete):
        path = os.path.join(args.tracks_dir, folder)
        if args.execute:
            shutil.rmtree(path)
            print(f"  deleted: {folder}")
        else:
            print(f"  would delete: {folder}")

    if not args.execute:
        print("\nDry run complete. Pass --execute to actually delete.")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()
