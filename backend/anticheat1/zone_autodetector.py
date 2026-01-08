"""
zone_autodetector.py

This module automatically detects benches and people on a given frame,
associates them, and creates static zones to be used for monitoring.
"""
import numpy as np
import math

def create_zones_from_frame(frame, yolo_model, padding=20, max_dist=200):
    """
    Analyzes a single frame to automatically generate zones around person-bench pairs.

    Args:
        frame: The video frame to analyze.
        yolo_model: The pre-loaded YOLO model.
        padding (int): Pixels to add around the combined person-bench box to create the zone.
        max_dist (int): Maximum pixel distance between centers of a person and a bench to be considered a pair.

    Returns:
        A list of zones, where each zone is a numpy array of 4 points (a rectangle).
    """
    print("[INFO] Starting automatic zone detection...")
    
    # Detect both people (0) and benches (13) in the COCO dataset
    results = yolo_model(frame, verbose=False, classes=[0, 13])[0]

    people_boxes = []
    bench_boxes = []

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf > 0.4:
            class_id = int(box.cls[0])
            coords = box.xyxy[0].cpu().numpy()
            if class_id == 0: # Person
                people_boxes.append(coords)
            elif class_id == 13: # Bench
                bench_boxes.append(coords)

    if not people_boxes or not bench_boxes:
        print("[WARN] Automatic zone detection failed: No people or benches found in the first frame.")
        return []

    zones = []
    used_bench_indices = set()

    # Associate each person with the closest, unused bench
    for p_box in people_boxes:
        px1, py1, px2, py2 = p_box
        p_center = ( (px1 + px2) / 2, (py1 + py2) / 2 )
        
        closest_bench_idx = -1
        min_dist = float('inf')

        for i, b_box in enumerate(bench_boxes):
            if i in used_bench_indices:
                continue
            
            bx1, by1, bx2, by2 = b_box
            b_center = ( (bx1 + bx2) / 2, (by1 + by2) / 2 )
            
            dist = math.hypot(p_center[0] - b_center[0], p_center[1] - b_center[1])

            if dist < min_dist:
                min_dist = dist
                closest_bench_idx = i
        
        # If a close enough bench was found, create a zone and mark the bench as used
        if closest_bench_idx != -1 and min_dist < max_dist:
            used_bench_indices.add(closest_bench_idx)
            b_box = bench_boxes[closest_bench_idx]
            
            # Create a union bounding box for the person and the bench
            x1 = min(p_box[0], b_box[0])
            y1 = min(p_box[1], b_box[1])
            x2 = max(p_box[2], b_box[2])
            y2 = max(p_box[3], b_box[3])
            
            # Add padding to create the final zone
            zone_rect = np.array([
                [x1 - padding, y1 - padding],
                [x2 + padding, y1 - padding],
                [x2 + padding, y2 + padding],
                [x1 - padding, y2 + padding]
            ], dtype=np.int32)
            
            zones.append(zone_rect)

    print(f"[INFO] Automatically generated {len(zones)} zones.")
    return zones