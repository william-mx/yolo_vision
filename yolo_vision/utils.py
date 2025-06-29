def parse_predictions(predictions):
    """
    Parses an Ultralytics YOLO prediction result and extracts detections as a list.

    Assumes live video processing (frame-by-frame), so batch size = 1.

    Parameters
    ----------
    predictions : list
        Output from the Ultralytics model's predict() call.

    Returns
    -------
    success : bool
        True if detections were found, False otherwise.

    results : list of lists
        Each inner list contains [score, class_id, cx, cy, w, h] for a detected object.
        Returns an empty list if no detections are present.
    """
    
    if len(predictions) == 0:
        return False, []

    # We are live-streaming video, so we parse one frame at a time (batch size = 1)
    boxes = predictions[0].boxes

    scores = boxes.conf.cpu().numpy()
    ids = boxes.cls.cpu().numpy().astype(int)
    xywh = boxes.xywh.cpu().numpy()

    return True, [
        [
            float(score),
            int(cls_id),
            float(cx),
            float(cy),
            float(w),
            float(h)
        ]
        for score, cls_id, (cx, cy, w, h) in zip(scores, ids, xywh)
    ]
