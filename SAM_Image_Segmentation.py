import os
import urllib.request
import cv2
import torch
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# ─── FUNCTIONS ─────────────────────────────────────────────────────────────────

def download_checkpoint_if_missing(backbone_type: str, checkpoint_dir: str, checkpoint_info: dict) -> str:
    """
    Check if the checkpoint file exists locally; if not, download it.
    Returns the full local path to the checkpoint.
    """
    if backbone_type not in checkpoint_info:
        raise ValueError(f"Unknown backbone '{backbone_type}'. Choose from {list(checkpoint_info.keys())}.")
    info = checkpoint_info[backbone_type]
    local_path = os.path.join(checkpoint_dir, info["filename"])
    if not os.path.exists(local_path):
        print(f"[INFO] Checkpoint for '{backbone_type}' not found. Downloading …")
        try:
            urllib.request.urlretrieve(info["url"], local_path)
            print(f"[INFO] Downloaded checkpoint to: {local_path}")
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to download checkpoint: {e}")
    else:
        print(f"[INFO] Found existing checkpoint at: {local_path}")
    return local_path

def load_image(image_path: str) -> np.ndarray:
    """
    Load an image in RGB format (uint8).
    - If .tif/.tiff: use tifffile (handles multi-channel, 16-bit, etc.)
    - Else: use OpenCV for .jpg/.png
    """
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".tif", ".tiff"]:
        arr = tifffile.imread(image_path)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        elif arr.ndim == 3:
            if arr.shape[0] in [1, 3, 4] and arr.shape[0] != arr.shape[1]:
                arr = np.moveaxis(arr, 0, -1)
            if arr.shape[2] > 3:
                arr = arr[:, :, :3]
            if arr.shape[2] == 1:
                arr = np.concatenate([arr]*3, axis=-1)
        else:
            raise ValueError(f"Unexpected TIFF shape: {arr.shape}")
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            mn, mx = arr.min(), arr.max()
            if mx > mn:
                arr = (arr - mn) / (mx - mn) * 255.0
            else:
                arr = np.zeros_like(arr)
            arr = arr.astype(np.uint8)
        return arr
    else:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"[ERROR] Could not read '{image_path}'")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

def show_and_save_masks(image_rgb: np.ndarray, masks: list, output_dir: str):
    """
    For each mask in `masks`, overlay it on `image_rgb` with a random semi-transparent color,
    then save both the overlay and the binary mask to `output_dir`.
    """
    img_float = image_rgb.astype(np.float32) / 255.0

    def random_color():
        return np.random.rand(3,)

    for idx, mask_info in enumerate(masks):
        mask = mask_info["segmentation"]  # boolean array (H×W)
        color = random_color()

        overlay = img_float.copy()
        alpha = 0.5
        overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha

        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.axis("off")
        plt.title(f"Mask {idx}")
        save_vis = os.path.join(output_dir, f"mask_vis_{idx}.png")
        plt.savefig(save_vis, bbox_inches="tight", pad_inches=0)
        plt.close()

        mask_uint8 = (mask * 255).astype(np.uint8)
        save_mask = os.path.join(output_dir, f"mask_{idx}.png")
        cv2.imwrite(save_mask, mask_uint8)

def overlay_all_masks(image_rgb: np.ndarray, masks: list, alpha: float = 0.4) -> np.ndarray:
    """
    Create a single overlay image where all masks are drawn on top of `image_rgb` with random colors.
    Returns an H×W×3 float32 array in [0,1].
    """
    overlay = image_rgb.astype(np.float32) / 255.0

    def random_color():
        return np.random.rand(3,)

    for mask_info in masks:
        mask = mask_info["segmentation"]
        color = random_color()
        overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha

    return overlay

# ─── MAIN PROCESS ───────────────────────────────────────────────────────────────

def main(
    backbone: str,
    input_image_path: str,
    checkpoint_dir: str,
    checkpoint_info: dict,
    output_dir: str
):
    # 1. Ensure checkpoint is downloaded
    checkpoint_path = download_checkpoint_if_missing(backbone, checkpoint_dir, checkpoint_info)

    # 2. Load SAM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[backbone](checkpoint=checkpoint_path)
    sam.to(device=device)

    # 3. Automatic mask generator (using correct parameter names)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_batch=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        box_nms_thresh=0.3,
        crop_n_layers=0,
        crop_nms_thresh=0.5,
        crop_overlap_ratio=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=1000,
    )

    # 4. Load image
    image_rgb = load_image(input_image_path)
    print(f"[INFO] Loaded image '{input_image_path}' with shape {image_rgb.shape}")

    # 5. Generate masks
    print("[INFO] Running SamAutomaticMaskGenerator …")
    masks = mask_generator.generate(image_rgb)
    print(f"[INFO] Generated {len(masks)} mask proposals.")

    # 6. Save each individual mask & its overlay
    show_and_save_masks(image_rgb, masks, output_dir)
    print(f"[INFO] Saved individual mask overlays and binary masks to: {output_dir}")

    # 7. Create & save composite overlay of all masks
    combined_overlay = overlay_all_masks(image_rgb, masks, alpha=0.4)
    plt.figure(figsize=(8, 8))
    plt.imshow(combined_overlay)
    plt.axis("off")
    plt.title("All Masks Overlaid on Original Image")
    combined_path = os.path.join(output_dir, "overlay_all_masks.png")
    plt.savefig(combined_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"[INFO] Saved combined overlay to: {combined_path}")

    # 8. (Optional) Example of interactive refinement for the first mask
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)
    box = masks[0]["bbox"]  # (x_min, y_min, width, height)
    input_box = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]])
    transformed_box = predictor.transform.apply_boxes_torch(
        torch.tensor([input_box], device=device), image_rgb.shape[:2]
    )

    with torch.no_grad():
        mask_logits, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_box,
            multimask_output=False
        )
        fine_mask = mask_logits[0][0].cpu().numpy() > 0.0

    # Save refined mask overlay
    color = np.array([1.0, 0.0, 0.0])
    alpha_refine = 0.4
    overlay2 = image_rgb.astype(np.float32) / 255.0
    overlay2[fine_mask] = overlay2[fine_mask] * (1 - alpha_refine) + color * alpha_refine
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay2)
    plt.axis("off")
    plt.title("Refined Mask Example")
    refined_path = os.path.join(output_dir, "refined_mask.png")
    plt.savefig(refined_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"[INFO] Saved refined mask overlay to: {refined_path}")


# ─── USER PARAMETERS (EDIT BELOW) ───────────────────────────────────────────────

# Choose which SAM backbone to use/download: "vit_h", "vit_l", or "vit_b"
BACKBONE = "vit_h"

# Path to your input image (.tif, .tiff, .jpg, or .png). Change as needed:
INPUT_IMAGE_PATH = "//storage-ume.slu.se/home$/joms0005/Desktop/SLU/materials/presentations/presentation_04_06/images_demo/image2.jpg"

# Directory where checkpoints are stored (or will be downloaded to). Change if desired:
CHECKPOINT_DIR = "checkpoints"

# SAM checkpoint URLs & filenames (no need to edit unless you want a different version)
CHECKPOINT_INFO = {
    "vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "filename": "sam_vit_h.pth"
    },
    "vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "filename": "sam_vit_l.pth"
    },
    "vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "filename": "sam_vit_b.pth"
    }
}

# Directory where masks, overlays, and refined results will be saved:
OUTPUT_DIR = "sam_output"

# ─── RUN ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    main(
        backbone=BACKBONE,
        input_image_path=INPUT_IMAGE_PATH,
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_info=CHECKPOINT_INFO,
        output_dir=OUTPUT_DIR
    )
