from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import yaml
from tqdm import tqdm


class BDD100KYOLOPreparer:
    """
    Convert BDD100K semantic masks into YOLO-seg polygon labels and build YOLO dataset YAML.

    Output structure:
      <exp_dir>/bdd100k_yolo_seg/
        ├── images/
        │   ├── train -> symlink to bdd100k/images/train
        │   └── val   -> symlink to bdd100k/images/val
        ├── labels/
        │   ├── train/*.txt
        │   └── val/*.txt
        └── bdd100k_yolo.yaml
    """

    def __init__(
        self,
        bdd100k_base: str,
        num_classes: int,
        rebuild_labels: bool = False,
        ignore_index: Optional[int] = None,
    ):
        self.bdd_base = Path(bdd100k_base).resolve()
        self.num_classes = num_classes
        self.rebuild_labels = rebuild_labels
        self.ignore_index = num_classes - 1 if ignore_index is None else ignore_index

    def prepare(self, exp_dir: str) -> str:
        """Convert masks and write YAML.  Returns path to the YAML file."""
        exp_dir = Path(exp_dir).resolve()

        for split in ("train", "val"):
            src_img_dir = self.bdd_base / "images" / split
            src_mask_dir = self.bdd_base / "labels" / split
            # Write .txt labels next to the .png masks in the ORIGINAL dir
            # so YOLO's /images/→/labels/ path derivation works after symlink resolution
            dst_lbl_dir = src_mask_dir

            # Check whether conversion is needed
            existing_txts = list(dst_lbl_dir.glob("*.txt"))
            if not self.rebuild_labels and existing_txts:
                print(
                    f"[YOLO prep] Reusing {len(existing_txts)} existing "
                    f"'{split}' label files "
                    f"(pass --yolo_rebuild_labels true to regenerate)"
                )
                continue

            # Remove old .txt labels if rebuilding
            for old_txt in existing_txts:
                old_txt.unlink()

            img_paths = sorted(self._iter_images(src_img_dir))
            print(
                f"[YOLO prep] Converting {len(img_paths)} '{split}' "
                f"masks → YOLO polygon labels …"
            )

            for img_path in tqdm(img_paths, desc=f"  {split}", unit="img"):
                stem = img_path.stem
                txt_path = dst_lbl_dir / f"{stem}.txt"
                mask_path = self._find_mask_path(src_mask_dir, stem)

                if not mask_path.is_file():
                    txt_path.write_text("")
                    continue

                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    txt_path.write_text("")
                    continue

                lines = self._mask_to_yolo_segments(mask)
                txt_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        # Delete stale YOLO cache files so labels are re-scanned
        for split in ("train", "val"):
            cache = self.bdd_base / "labels" / f"{split}.cache"
            if cache.exists():
                cache.unlink()
                print(f"[YOLO prep] Removed stale cache: {cache}")

        # Write YAML — use bdd100k base as the dataset root
        yolo_yaml = exp_dir / "bdd100k_yolo.yaml"
        yolo_yaml.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "path": str(self.bdd_base),
            "train": "images/train",
            "val": "images/val",
            "names": {i: str(i) for i in range(self.num_classes)},
        }
        with open(yolo_yaml, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

        print(f"[YOLO prep] Dataset YAML written to {yolo_yaml}")
        return str(yolo_yaml)


    @staticmethod
    def _iter_images(img_dir: Path):
        exts = ("*.jpg", "*.jpeg", "*.png")
        for ext in exts:
            for p in Path(img_dir).glob(ext):
                yield p



    @staticmethod
    def _find_mask_path(mask_dir: Path, stem: str) -> Path:
        mask_dir = Path(mask_dir)
        p1 = mask_dir / f"{stem}_train_id.png"
        if p1.is_file():
            return p1
        return mask_dir / f"{stem}.png"

    def _mask_to_yolo_segments(self, mask: np.ndarray) -> List[str]:
        h, w = mask.shape
        mask = mask.copy()
        mask[mask == 255] = self.ignore_index

        lines: List[str] = []
        for cls_id in range(self.num_classes):
            if cls_id == self.ignore_index:
                continue  # skip the ignore/unknown class

            bin_mask = (mask == cls_id).astype(np.uint8)
            if bin_mask.sum() == 0:
                continue

            found = cv2.findContours(
                bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = found[0] if len(found) == 2 else found[1]

            for cnt in contours:
                if len(cnt) < 3 or cv2.contourArea(cnt) < 4:
                    continue

                pts = cnt.reshape(-1, 2).astype(np.float32)
                pts[:, 0] /= float(w)
                pts[:, 1] /= float(h)

                coords = pts.reshape(-1).tolist()
                if len(coords) < 6:
                    continue

                lines.append(f"{cls_id} " + " ".join(f"{v:.6f}" for v in coords))

        return lines
