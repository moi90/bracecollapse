import argparse
from collections.abc import Iterable
from pathlib import Path
import sys

import pathspec

from bracecollapse import PrefixTreeNode


def list_files_recursively(
    directory: Path,
    *,
    exclude=None,
) -> Iterable[str]:
    """
    Recursively list all files in the given directory.
    """

    file_iterator = directory.rglob("*")

    if exclude is not None:
        spec = pathspec.PathSpec.from_lines("gitignore", exclude)
    else:
        spec = None

    for file_path in file_iterator:
        filename = str(file_path.relative_to(directory))
        if file_path.is_dir():
            continue

        if spec and spec.match_file(filename):
            continue

        yield str(file_path.relative_to(directory))


def main():
    """
    Docstring for main
    """

    arg_parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    group = arg_parser.add_mutually_exclusive_group()
    group.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to the input file. If not provided, list files in the current directory.",
        type=Path,
    )
    group.add_argument(
        "--names-from",
        "-f",
        type=Path,
        help="Path to a file containing a list of names to process.",
    )

    arg_parser.add_argument(
        "--exclude",
        "-e",
        action="append",
        help="Patterns to exclude from processing. Can be specified multiple times.",
    )
    arg_parser.add_argument(
        "--alpha",
        choices=["set", "glob", "format"],
        default="set",
        help="Method to collapse alphabetic tokens. Default is 'set'.",
    )
    arg_parser.add_argument(
        "--numeric",
        choices=["set", "rangeset", "range", "glob", "format"],
        default="format",
        help="Method to collapse numeric tokens. Default is 'format'.",
    )

    args = arg_parser.parse_args()

    # Read names
    if args.names_from:
        print(f"Reading names from {args.names_from} ...", file=sys.stderr)
        with args.names_from.open("r") as f:
            names = [line.strip() for line in f if line.strip()]
    else:
        target_dir = args.path or Path(".")

        print(f"Scanning directory {target_dir} ...", file=sys.stderr)

        if not target_dir.is_dir():
            raise ValueError(f"The path {target_dir} is not a directory.")

        # Recursively list all files in the specified directory.
        names = list(list_files_recursively(target_dir, exclude=args.exclude))

    # Build a prefix tree for tokens
    tree = PrefixTreeNode()
    for name in names:
        tree.insert(name)

    # Convert the prefix tree to patterns
    patterns = tree.to_patterns(alpha=args.alpha, numeric=args.numeric)

    for pattern in patterns:
        print("".join(str(p) for p in pattern))


if __name__ == "__main__":
    main()
