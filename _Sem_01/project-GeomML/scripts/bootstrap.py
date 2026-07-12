from pathlib import Path
import subprocess
import sys
import importlib

def find_project_root(start=Path.cwd()):
    for p in [start, *start.parents]:
        if (p / ".project_root").exists():
            return p
    raise RuntimeError("Project root not found")


def install_editable(root: Path | None = None):
    if root is not None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(root)],
            check=True
        )


def import_package(pkg_name: str):
    if pkg_name in sys.modules:
        importlib.reload(sys.modules[pkg_name])
    return importlib.import_module(pkg_name)


def setup(pkg_name: str):
    root = find_project_root()
    install_editable(root)
    return import_package(pkg_name)


def install_editable():
    root = find_project_root()
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(root)],
        check=True
    )
    return root


def setup_dev_env():
    root = find_project_root()
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(root)],
        check=True
    )
    return root

if __name__ == "__main__":
    setup_dev_env()

