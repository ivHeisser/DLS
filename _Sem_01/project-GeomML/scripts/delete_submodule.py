#!/usr/bin/env python3

from pathlib import Path
import argparse
import re


def normalize(name: str):
    """
    Same normalization logic as in create_submodule.py.
    """
    folder_name = name.strip().lower()
    registry_name = folder_name.upper()
    singular_name = folder_name  # fallback (can be improved if needed)
    return folder_name, registry_name, singular_name


def remove_registry_entry(registry_file: Path, registry_name: str):
    """
    Removes line like:
        MODELS = Registry("model")
    """
    if not registry_file.exists():
        print("✓ registry.py not found, skipping registry cleanup.")
        return

    text = registry_file.read_text(encoding="utf-8")
    lines = text.splitlines()

    pattern = re.compile(rf"^\s*{re.escape(registry_name)}\s*=\s*Registry\(")

    new_lines = []
    removed = False

    for line in lines:
        if pattern.match(line):
            removed = True
            continue
        new_lines.append(line)

    if not removed:
        print(f"✓ No registry entry '{registry_name}' found.")
        return

    registry_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"✓ Removed registry entry '{registry_name}'.")


def delete_folder(module_dir: Path):
    if not module_dir.exists():
        print(f"✓ Folder does not exist: {module_dir}")
        return

    if not module_dir.is_dir():
        raise RuntimeError(f"Not a directory: {module_dir}")

    # safety check: avoid deleting root accidentally
    if module_dir.name in {"src", "geomml", "tools"}:
        raise RuntimeError(f"Refusing to delete unsafe path: {module_dir}")

    for child in module_dir.rglob("*"):
        if child.is_file():
            child.unlink()

    for child in sorted(module_dir.rglob("*"), reverse=True):
        if child.is_dir():
            child.rmdir()

    module_dir.rmdir()

    print(f"✓ Deleted folder: {module_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Delete a registered submodule."
    )

    parser.add_argument(
        "name",
        help="Module name (Models, Losses, Symmetries, ...)"
    )

    parser.add_argument(
        "--package",
        default="geomml",
        help="Python package under src (default: geomml)"
    )

    args = parser.parse_args()

    folder_name, registry_name, _ = normalize(args.name)

    root = Path(__file__).resolve().parents[1]
    package_dir = root / "src" / args.package

    module_dir = package_dir / folder_name
    registry_file = package_dir / "registry.py"

    if not package_dir.exists():
        raise RuntimeError(f"Package not found: {package_dir}")

    print(f"Target folder : {module_dir}")
    print(f"Registry file : {registry_file}")
    print()

    # 1. delete folder
    delete_folder(module_dir)

    # 2. clean registry
    remove_registry_entry(registry_file, registry_name)

    print()
    print("Done.")
    print(f"Removed module: {folder_name}")
    print(f"Registry key  : {registry_name}")


if __name__ == "__main__":
    main()