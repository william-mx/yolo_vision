import numpy as np
import cv2

def parse_predictions(predictions):
    """
    Parses Ultralytics YOLO results and extracts both bboxes and a combined mask.

    Returns
    -------
    success : bool
        True if detections were found, False otherwise.
    results : list of tuples
        Each entry: (label: str, score: float, cx, cy, w, h, shifted_cls_id).
    combined_mask : np.ndarray
        A 2D uint8 array where pixel value = shifted class ID (0 = background).
    """

    if not predictions:
        return False, [], None

    if len(predictions[0].boxes) == 0:
        h, w = predictions[0].orig_shape
        return False, [], np.zeros((h, w), dtype=np.uint8)

    result = predictions[0]
    boxes = result.boxes

    # 1. Get Original Dimensions (e.g., 1080, 1920)
    h_orig, w_orig = result.orig_shape

    # 2. Extract Box Data — shift cls_ids by +1 to reserve 0 for background
    scores = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int) + 1
    xywh = boxes.xywh.cpu().numpy()

    # 3. Build the Detection List — look up label via unshifted ID (cls_id - 1)
    results_list = [
        (
            str(result.names[cls_id - 1]),
            float(score),
            float(cx), float(cy), float(w), float(h),
            int(cls_id)
        )
        for cls_id, score, (cx, cy, w, h) in zip(cls_ids, scores, xywh)
    ]

    # 4. Handle Masks — cls_ids already shifted, use directly
    full_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
    if result.masks is not None:
        masks_np = result.masks.data.cpu().numpy()  # (N, H, W)
        num_objs = masks_np.shape[0]

        class_ids = cls_ids[:num_objs].reshape(-1, 1, 1)

        combined_small = np.max(masks_np * class_ids, axis=0).astype(np.uint8)

        full_mask = cv2.resize(
            combined_small,
            (w_orig, h_orig),
            interpolation=cv2.INTER_NEAREST
        )

    return True, results_list, full_mask