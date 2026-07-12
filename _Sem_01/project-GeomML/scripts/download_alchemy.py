# scripts/download_alchemy.py

from pathlib import Path
import requests
import zipfile

URL = "https://alchemy.tencent.com/data/alchemy-v20191129.zip"

DATA_DIR = Path("data") / "alchemy"
DATA_DIR.mkdir(parents=True, exist_ok=True)
zip_path = DATA_DIR / "alchemy.zip"



def extract_flat(zip_path: Path, target_dir: Path):
    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()

        # находим общий корневой префикс (папку внутри архива)
        root = members[0].split("/")[0] + "/"

        for member in members:
            if member.endswith("/"):
                continue

            # убираем root-папку
            relative_path = member[len(root):]

            target_path = target_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)

            with z.open(member) as src, open(target_path, "wb") as dst:
                dst.write(src.read())





print("Downloading dataset...")
with requests.get(URL, stream=True) as r:
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

                if total:
                    print(f"\r{downloaded / total * 100:.2f}%", end="")

print("\nDownload complete.")
print("Extracting...")
extract_flat(zip_path, DATA_DIR)
print("Extraction complete.")
print("Removing archive...")
zip_path.unlink()
print(f"Done. Data ready at: {DATA_DIR}")