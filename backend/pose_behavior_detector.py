import numpy as np
import math


class YOLOPoseBehaviorDetector:
    """
    Behavior detection using YOLOv8 Pose keypoints + DeepSORT tracks.

    Detects:
      - Standing / Walking / Leaving Seat  (hip movement)
      - Passing Item / Interaction         (hand-to-hand proximity over time)

    NOTE: 'Leaning' and 'Looking Away' are intentionally NOT implemented here.
    """

    def __init__(
        self,
        fps: float = 30.0,
        walk_move_thresh: float = 20.0,     # pixels between frames to call it walking/standing
        pass_dist_thresh: float = 60.0,     # max distance between wrists to consider "close"
        same_row_thresh: float = 80.0,      # max vertical difference for same-row interaction
        pass_min_frames: int = 3            # consecutive close frames to trigger passing
    ):
        self.fps = fps
        self.walk_move_thresh = walk_move_thresh
        self.pass_dist_thresh = pass_dist_thresh
        self.same_row_thresh = same_row_thresh
        self.pass_min_frames = pass_min_frames

        # Per-track hip centers for motion (walking/standing)
        self.prev_hip_centers = {}  # track_id -> np.array([x, y])

        # For passing detection: consecutive frames of close hands per pair
        self.pair_close_frames = {}  # frozenset({id1, id2}) -> int

    # ----------------- Utility helpers -----------------

    @staticmethod
    def _iou(box1, box2):
        """Compute IoU between two [x1,y1,x2,y2] boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter = inter_w * inter_h
        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        denom = area1 + area2 - inter + 1e-6
        return inter / denom if denom > 0 else 0.0

    # ----------------- Main update -----------------

    def update(self, frame, tracks, results):
        """
        frame   : current frame (resized)
        tracks  : DeepSORT tracks (with .track_id, .to_ltrb(), .is_confirmed())
        results : YOLO results object (pose model) for this frame
        """
        alerts = {}

        # 1) Build mapping: YOLO 'person' box -> keypoints
        person_boxes = []
        person_kps = []

        if results.keypoints is None or results.boxes is None:
            return alerts

        boxes = results.boxes
        kps_tensor = results.keypoints.data  # shape: (num, 17, 3)

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            label = results.names[cls_id]
            if label != "person":
                continue
            xyxy = box.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2]
            kps = kps_tensor[i].cpu().numpy()  # (17, 3)
            person_boxes.append(xyxy)
            person_kps.append(kps)

        # 2) Match tracks to closest YOLO person box via IoU
        track_kps = {}  # track_id -> keypoints (17,3)
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            tx1, ty1, tx2, ty2 = track.to_ltrb()
            t_box = np.array([tx1, ty1, tx2, ty2], dtype=float)

            best_iou = 0.0
            best_idx = -1
            for idx, p_box in enumerate(person_boxes):
                iou_val = self._iou(t_box, p_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = idx

            if best_idx >= 0 and best_iou >= 0.3:  # decent match
                track_kps[tid] = person_kps[best_idx]

        # 3) Standing / Walking detection based on hip movement
        for tid, kps in track_kps.items():
            # hips: indices 11 (L hip), 12 (R hip) in COCO-style layout
            hip_center = np.mean(kps[11:13, :2], axis=0)  # (x, y)
            prev_center = self.prev_hip_centers.get(tid, None)

            if prev_center is not None:
                move_dist = float(np.linalg.norm(hip_center - prev_center))
                if move_dist > self.walk_move_thresh:
                    alerts.setdefault(tid, set()).add("Standing / Walking / Leaving Seat")

            self.prev_hip_centers[tid] = hip_center

        # 4) Passing item / interaction based on hand-to-hand distance across tracks
        tids = list(track_kps.keys())
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                tid1, tid2 = tids[i], tids[j]
                kps1, kps2 = track_kps[tid1], track_kps[tid2]

                # Wrist positions: indices 9 (L wrist), 10 (R wrist)
                wrists1 = np.mean(kps1[9:11, :2], axis=0)
                wrists2 = np.mean(kps2[9:11, :2], axis=0)

                # Hip centers for row check
                hip1 = np.mean(kps1[11:13, :2], axis=0)
                hip2 = np.mean(kps2[11:13, :2], axis=0)

                hand_dist = float(np.linalg.norm(wrists1 - wrists2))
                same_row = abs(float(hip1[1] - hip2[1])) < self.same_row_thresh

                pair_key = frozenset((tid1, tid2))

                if hand_dist < self.pass_dist_thresh and same_row:
                    # close hands this frame -> increment
                    self.pair_close_frames[pair_key] = self.pair_close_frames.get(pair_key, 0) + 1
                else:
                    # not close -> reset
                    self.pair_close_frames[pair_key] = 0

                # If hands close for enough consecutive processed frames -> passing
                if self.pair_close_frames.get(pair_key, 0) >= self.pass_min_frames:
                    alerts.setdefault(tid1, set()).add("Passing Item / Interaction")
                    alerts.setdefault(tid2, set()).add("Passing Item / Interaction")

        return alerts
