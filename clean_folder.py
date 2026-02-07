#!/usr/bin/env python3
"""
Safe folder cleaner with optional .trash purge.
Usage examples:
  python clean_folder.py . --dry-run
  python clean_folder.py . --no-dry-run --trash
  python clean_folder.py . --no-dry-run --delete --yes
  python clean_folder.py . --purge-trash --purge-days 60
  python clean_folder.py . --purge-trash --no-dry-run --purge-yes
"""

import argparse
import shutil
from pathlib import Path
import fnmatch
import sys
import time

DEFAULT_PATTERNS = [
    "__pycache__", "*.pyc", "*.pyo", "*~", "*.tmp",
    ".DS_Store", "Thumbs.db", ".pytest_cache", "*.log"
]


def matches(path: Path, patterns):
    name = path.name
    for pat in patterns:
        if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(str(path), pat):
            return True
    return False


def purge_trash(trash_dir: Path, days: int, dry_run: bool, yes: bool):
    if not trash_dir.exists() or not trash_dir.is_dir():
        print("No trash directory found at", trash_dir)
        return

    cutoff = time.time() - (days * 86400)
    candidates = [p for p in trash_dir.rglob("*") if p.exists() and p.stat().st_mtime < cutoff]

    if not candidates:
        print(f"No items older than {days} days in {trash_dir}.")
        return

    # sort deepest first so we remove children before parents
    candidates.sort(key=lambda p: -len(str(p)))

    print(f"Found {len(candidates)} item(s) older than {days} days in {trash_dir}:")
    for p in candidates:
        try:
            print("  ", p.relative_to(trash_dir))
        except Exception:
            print("  ", p)

    if dry_run:
        print("\nDRY RUN: no items will be removed. Use --no-dry-run to actually delete or move.")
        return

    if not yes:
        confirm = input(f"Confirm permanent deletion of these {len(candidates)} item(s)? Type 'yes' to proceed: ")
        if confirm.strip().lower() != "yes":
            print("Purge aborted.")
            return

    for p in candidates:
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            print("Removed:", p.relative_to(trash_dir))
        except Exception as e:
            print("Error removing", p, ":", e)


def main():
    p = argparse.ArgumentParser(description="Clean a folder of common temp/dev artifacts (safe by default).")
    p.add_argument("folder", nargs="?", default=".", help="Folder to clean (default: current dir)")
    p.add_argument("--pattern", "-p", action="append", help="Add extra glob patterns to match (can be used multiple times)")
    p.add_argument("--dry-run", action="store_true", default=True, help="Show what would be done (default)")
    p.add_argument("--no-dry-run", action="store_true", help="Disable dry-run (perform actions)")
    p.add_argument("--trash", action="store_true", help="Move matches to a .trash/ folder inside the target instead of deleting")
    p.add_argument("--trash-dir", default=".trash", help="Trash folder name (default: .trash)")
    p.add_argument("--delete", action="store_true", help="Permanently delete matched files/directories (requires confirmation or --yes)")
    p.add_argument("--remove-empty-dirs", action="store_true", help="Remove empty directories after cleaning")
    p.add_argument("--yes", action="store_true", help="Skip confirmation prompt when deleting")
    # Purge options
    p.add_argument("--purge-trash", action="store_true", help="Purge files in the trash dir older than --purge-days")
    p.add_argument("--purge-only", action="store_true", help="Only run the purge (skip normal cleaning)")
    p.add_argument("--purge-days", type=int, default=30, help="Purge items older than this many days (default 30)")
    p.add_argument("--purge-yes", action="store_true", help="Skip confirmation for purge")
    args = p.parse_args()

    dry_run = not args.no_dry_run
    folder = Path(args.folder).resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"Error: folder {folder} does not exist or is not a directory.")
        sys.exit(1)

    patterns = list(DEFAULT_PATTERNS)
    if args.pattern:
        patterns.extend(args.pattern)

    trash_dir = folder / args.trash_dir

    # If purge-only requested or purge requested, run purge path
    if args.purge_trash or args.purge_only:
        purge_trash(trash_dir, args.purge_days, dry_run, args.purge_yes)
        if args.purge_only:
            return

    matches_list = []
    for path in folder.rglob("*"):
        # don't act on the trash dir itself
        if str(path.resolve()).startswith(str(trash_dir.resolve())):
            continue
        if matches(path, patterns):
            matches_list.append(path)

    if not matches_list:
        print("No matches found for patterns:", patterns)
        return

    print("Found matches:")
    for pth in matches_list:
        try:
            print("  ", pth.relative_to(folder))
        except Exception:
            print("  ", pth)

    if dry_run:
        print("\nDRY RUN (no changes made). Use --no-dry-run plus --trash or --delete to act.")
        return

    if args.delete and not args.yes:
        confirm = input("You confirmed permanent deletion of these items? Type 'yes' to proceed: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return

    if args.trash:
        trash_dir.mkdir(exist_ok=True)
        for pth in matches_list:
            rel = pth.relative_to(folder)
            target = trash_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                pth.replace(target)
                print("Moved:", rel)
            except Exception as e:
                print("Error moving", rel, ":", e)
    elif args.delete:
        for pth in matches_list:
            try:
                if pth.is_dir():
                    shutil.rmtree(pth)
                else:
                    pth.unlink()
                print("Deleted:", pth.relative_to(folder))
            except Exception as e:
                print("Error deleting", pth, ":", e)
    else:
        print("No action specified. Use --trash or --delete with --no-dry-run.")

    if args.remove_empty_dirs:
        for d in sorted((folder.rglob("*")), key=lambda x: -len(str(x))):
            if d.is_dir() and not any(d.iterdir()) and d != folder:
                try:
                    d.rmdir()
                    print("Removed empty dir:", d.relative_to(folder))
                except Exception as e:
                    print("Error removing dir", d, ":", e)

if __name__ == "__main__":
    main()
