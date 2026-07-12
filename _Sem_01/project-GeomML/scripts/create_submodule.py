#!/usr/bin/env python3

from pathlib import Path
import argparse

import inflect


def normalize(name: str):
    """
    Convert user input into folder name, registry name and singular name.

    Examples
    --------
    Models      -> ("models", "MODELS", "model")
    LOSSES      -> ("losses", "LOSSES", "loss")
    Symmetries  -> ("symmetries", "SYMMETRIES", "symmetry")
    """

    folder_name = name.strip().lower()
    registry_name = folder_name.upper()

    engine = inflect.engine()

    singular = engine.singular_noun(folder_name)
    if singular is False:
        singular = folder_name

    return folder_name, registry_name, singular


def create_init(package_name: str, registry_name: str) -> str:
    return f'''import importlib
import pkgutil

from {package_name}.registry import {registry_name}


for _, module_name, _ in pkgutil.iter_modules(__path__):
    try:
        importlib.import_module(f"{{__name__}}.{{module_name}}")
    except Exception as e:
        print(module_name, e)


def build(name, **kwargs):
    return {registry_name}.build(name, **kwargs)
'''


def append_registry(registry_file: Path,
                    registry_name: str,
                    singular_name: str):

    line = f'{registry_name} = Registry("{singular_name}")'

    text = registry_file.read_text(encoding="utf-8")

    if line in text:
        print(f"✓ Registry '{registry_name}' already exists.")
        return

    if not text.endswith("\n"):
        text += "\n"

    text += line + "\n"

    registry_file.write_text(text, encoding="utf-8")

    print(f"✓ Added registry '{registry_name}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Create a new registered submodule."
    )

    parser.add_argument(
        "name",
        help="Plural module name (Models, Losses, Symmetries, ...)"
    )

    parser.add_argument(
        "--package",
        default="geomml",
        help="Python package under src (default: geomml)"
    )

    args = parser.parse_args()

    folder_name, registry_name, singular_name = normalize(args.name)

    #
    # Project layout:
    #
    # project/
    # ├── scripts/
    # │   └── create_submodule.py
    # └── src/
    #     └── geomml/
    #
    root = Path(__file__).resolve().parents[1]

    package_dir = root / "src" / args.package

    if not package_dir.exists():
        raise RuntimeError(
            f"Package '{args.package}' not found:\n{package_dir}"
        )

    #
    # Create submodule directory
    #
    module_dir = package_dir / folder_name
    module_dir.mkdir(parents=True, exist_ok=True)

    print(f"✓ Folder: {module_dir}")

    #
    # Create __init__.py
    #
    init_file = module_dir / "__init__.py"

    if init_file.exists():
        print("✓ __init__.py already exists.")
    else:
        init_file.write_text(
            create_init(args.package, registry_name),
            encoding="utf-8",
        )
        print("✓ Created __init__.py")
    
    #
    # Create README.md
    #
    readme_file = module_dir / "README.md"

    if readme_file.exists():
        print("✓ README.md already exists.")
    else:
        readme_file.write_text(
            f"# {folder_name}\n\nSubmodule for {folder_name}.\n",
            encoding="utf-8",
        )
        print("✓ Created README.md")

    #
    # Update registry.py
    #
    registry_file = package_dir / "registry.py"

    if not registry_file.exists():
        raise RuntimeError(
            f"'registry.py' not found:\n{registry_file}"
        )

    append_registry(
        registry_file,
        registry_name,
        singular_name,
    )

    print()
    print("Done.")
    print(f"Package : {folder_name}")
    print(f"Registry: {registry_name}")


if __name__ == "__main__":
    main()