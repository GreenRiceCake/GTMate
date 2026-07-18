import argparse
from pathlib import Path

from updater_core.package import build_file_manifest, validate_file_manifest
from updater_core.paths import INSTALLED_MANIFEST_FILE, atomic_write_json


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the managed file manifest for a GTMate release folder."
    )
    parser.add_argument("root", type=Path, help="Assembled release folder")
    parser.add_argument("version", help="GTMate version, for example 1.2.0")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path; defaults to <root>/installed_manifest.json",
    )
    parser.add_argument(
        "--preserve",
        action="append",
        default=["bot_config.json"],
        help="Relative path preserved across updates; may be repeated",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    root = arguments.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Release folder not found: {root}")

    manifest = build_file_manifest(
        root,
        arguments.version,
        preserve_paths=arguments.preserve,
    )
    manifest = validate_file_manifest(manifest, require_updater=True)
    output = (arguments.output or (root / INSTALLED_MANIFEST_FILE)).resolve()
    atomic_write_json(output, manifest)
    total_size = sum(entry["size"] for entry in manifest["files"])
    print(f"Wrote {output}")
    print(f"Managed files: {len(manifest['files'])}")
    print(f"Managed bytes: {total_size}")


if __name__ == "__main__":
    main()
