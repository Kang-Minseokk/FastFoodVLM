# dataset 디렉토리에 넣어서 사용할 것.

from pathlib import Path
import random
import shutil

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def trim_images_to_300(
    dataset_dir: str = ".",
    limit: int = 300,
    seed: int = 0,
    mode: str = "move",   # "move" or "delete"
    trash_dir: str = "trash_over_limit"
):
    dataset_path = Path(dataset_dir)
    trash_path = dataset_path / trash_dir
    random.seed(seed)

    food_dirs = [p for p in dataset_path.iterdir() if p.is_dir() and p.name != trash_dir]
    food_dirs.sort()

    for food in food_dirs:
        images_dir = food / "images"
        if not images_dir.is_dir():
            # 구조가 다른 폴더는 스킵
            continue

        files = [p for p in images_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in IMG_EXTS]
        n = len(files)

        if n <= limit:
            print(f"[OK] {food.name}: {n} images")
            continue

        # 재현 가능하게 랜덤으로 limit개만 남기고 제거
        files.sort()  # 섞기 전 고정
        random.shuffle(files)

        keep = set(files[:limit])
        remove = [p for p in files if p not in keep]

        print(f"[TRIM] {food.name}: {n} -> {limit} (remove {len(remove)})")

        if mode == "delete":
            for p in remove:
                p.unlink()
        elif mode == "move":
            dst_dir = trash_path / food.name / "images"
            dst_dir.mkdir(parents=True, exist_ok=True)
            for p in remove:
                shutil.move(str(p), str(dst_dir / p.name))
        else:
            raise ValueError("mode must be 'move' or 'delete'")

if __name__ == "__main__":
    trim_images_to_300(dataset_dir=".", limit=100, seed=0, mode="delete")
    # 삭제로 하고 싶으면 mode="delete"로 바꾸세요.
