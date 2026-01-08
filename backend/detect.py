import argparse
import sys
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import torch
from PIL import Image

try:
    import clip
except Exception:
    print("Install CLIP: pip install git+https://github.com/openai/CLIP.git", file=sys.stderr)
    raise

from ultralytics import YOLO


# ---------- Utility Functions ----------
def _select_device(device_arg: str) -> torch.device:
    device_arg = (device_arg or "cpu").lower()
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _avg_normalized(feats: torch.Tensor) -> torch.Tensor:
    feats = feats / feats.norm(dim=-1, keepdim=True)
    avg = feats.mean(dim=0, keepdim=True)
    return avg / avg.norm(dim=-1, keepdim=True)


def build_class_embeddings(model, device, prompts: Dict[str, List[str]]):
    with torch.no_grad():
        out = {}
        for cls, texts in prompts.items():
            tokens = clip.tokenize(texts).to(device)
            txt = model.encode_text(tokens)
            out[cls] = _avg_normalized(txt)
        return out


def classify_with_clip(model, preprocess, device, image_bgr, class_embs):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    img_in = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feat = model.encode_image(img_in)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        labels = list(class_embs.keys())
        txt_mat = torch.cat([class_embs[lbl] for lbl in labels], dim=0)
        logits = 100.0 * (img_feat @ txt_mat.t())
        probs = logits.softmax(dim=-1).squeeze(0)
    probs_np = probs.detach().float().cpu().numpy()
    idx = int(probs_np.argmax())
    return labels[idx], float(probs_np[idx])


def draw_box(frame, box, text_lines, color=(0, 255, 0), alpha: float = 0.3, thickness: int = 2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    # Label background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    t_th = 1
    pad = 4
    sizes = [cv2.getTextSize(t, font, font_scale, t_th)[0] for t in text_lines]
    box_w = (max(s[0] for s in sizes) if sizes else 0) + 2 * pad
    box_h = sum(s[1] for s in sizes) + (len(sizes) - 1) * 2 + 2 * pad
    y0 = max(0, y1 - box_h - 2)
    x0 = x1
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    y = y0 + pad + (sizes[0][1] if sizes else 0)
    for i, text in enumerate(text_lines):
        cv2.putText(frame, text, (x0 + pad, y), font, font_scale, (0, 0, 0), t_th, cv2.LINE_AA)
        if i + 1 < len(sizes):
            y += sizes[i + 1][1] + 2


def draw_overlay_legend(frame, show_prompts: bool, uniform_pos: List[str], id_pos: List[str]):
    h, w = frame.shape[:2]
    lines = [
        "YOLOv8 + CLIP",
        "q: quit | p: toggle prompts",
    ]
    if show_prompts:
        if uniform_pos:
            lines.append(f"Uniform prompt: {uniform_pos[0]}")
        if id_pos:
            lines.append(f"ID prompt: {id_pos[0]}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    pad = 6
    sizes = [cv2.getTextSize(t, font, font_scale, thickness)[0] for t in lines]
    box_w = min(440, (max(s[0] for s in sizes) if sizes else 0) + 2 * pad)
    box_h = sum(s[1] for s in sizes) + (len(sizes) - 1) * 3 + 2 * pad
    x0, y0 = 10, 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (32, 32, 32), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    y = y0 + pad + (sizes[0][1] if sizes else 0)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (x0 + pad, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        if i + 1 < len(sizes):
            y += sizes[i + 1][1] + 3


def parse_prompt_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    raw: List[str] = []
    for part in s.replace("\n", ";").replace("|", ";").split(";"):
        t = part.strip()
        if t:
            raw.append(t)
    return raw


def torso_crop(box: Tuple[int, int, int, int], frame_h: int, frame_w: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw, bh = (x2 - x1), (y2 - y1)
    ty1 = int(y1 + 0.2 * bh)
    ty2 = int(y1 + 0.7 * bh)
    tx1 = x1
    tx2 = x2
    # clamp
    tx1 = max(0, min(frame_w - 1, tx1))
    tx2 = max(0, min(frame_w - 1, tx2))
    ty1 = max(0, min(frame_h - 1, ty1))
    ty2 = max(0, min(frame_h - 1, ty2))
    if tx2 <= tx1 or ty2 <= ty1:
        return x1, y1, x2, y2
    return tx1, ty1, tx2, ty2


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="YOLOv8 + CLIP: Blue Uniform + ID Card")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every nth frame")
    parser.add_argument("--scale", type=float, default=0.5, help="Downscale factor for faster detection")
    parser.add_argument("--uniform-threshold", type=float, default=0.5, dest="uniform_thr", help="Uniform probability threshold")
    parser.add_argument("--id-threshold", type=float, default=0.5, dest="id_thr", help="ID card probability threshold")
    parser.add_argument("--show-prompts", action="store_true", help="Overlay primary prompts used")
    parser.add_argument("--uniform-positive", type=str, default="", help="Semicolon-separated positive prompts for Uniform")
    parser.add_argument("--uniform-negative", type=str, default="", help="Semicolon-separated negative prompts for No Uniform")
    parser.add_argument("--id-positive", type=str, default="", help="Semicolon-separated positive prompts for ID Card")
    parser.add_argument("--id-negative", type=str, default="", help="Semicolon-separated negative prompts for No ID Card")
    args = parser.parse_args()

    device = _select_device(args.device)
    clip_model, preprocess = clip.load("ViT-B/32", device=str(device))
    clip_model.eval()
    yolo = YOLO("yolov8n.pt")

    # Prompts (with optional CLI overrides)
    default_uniform_pos = [
        "a person wearing a blue uniform shirt",
        "a person in a blue collared uniform",
        "a student in a blue uniform shirt",
        "a worker wearing a blue uniform",
    ]
    default_uniform_neg = [
        "a person not wearing a blue uniform shirt",
        "a person in casual clothes without a blue uniform",
    ]
    default_id_pos = [
        "a person wearing an ID card on a lanyard",
        "a person with an identity badge around the neck",
        "an employee with an ID badge visible",
    ]
    default_id_neg = [
        "a person without any ID card visible",
        "a person not wearing an ID badge",
    ]

    user_uniform_pos = parse_prompt_list(args.uniform_positive)
    user_uniform_neg = parse_prompt_list(args.uniform_negative)
    user_id_pos = parse_prompt_list(args.id_positive)
    user_id_neg = parse_prompt_list(args.id_negative)

    uniform_prompts = {
        "Uniform": user_uniform_pos if user_uniform_pos else default_uniform_pos,
        "No Uniform": user_uniform_neg if user_uniform_neg else default_uniform_neg,
    }
    id_prompts = {
        "ID Card": user_id_pos if user_id_pos else default_id_pos,
        "No ID Card": user_id_neg if user_id_neg else default_id_neg,
    }
    uniform_embs = build_class_embeddings(clip_model, device, uniform_prompts)
    id_embs = build_class_embeddings(clip_model, device, id_prompts)

    # Open source
    src = args.source
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        print(f"Cannot open {src}")
        return

    window_name = "Optimized Uniform + ID Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % args.frame_skip != 0:
            continue

        # Resize for speed
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (int(w * args.scale), int(h * args.scale)))
        results = yolo.predict(source=resized, conf=args.conf, classes=[0], device=str(device), verbose=False)
        if not results:
            continue

        res = results[0]
        if res.boxes is None or res.boxes.xyxy is None:
            continue

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()

        # Scale boxes back
        scale_x, scale_y = w / resized.shape[1], h / resized.shape[0]
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = (box * [scale_x, scale_y, scale_x, scale_y]).astype(int)
            # torso crop for CLIP classification
            tx1, ty1, tx2, ty2 = torso_crop((x1, y1, x2, y2), h, w)
            crop = frame[ty1:ty2, tx1:tx2]
            if crop.size == 0:
                continue
            u_lbl, u_prob = classify_with_clip(clip_model, preprocess, device, crop, uniform_embs)
            i_lbl, i_prob = classify_with_clip(clip_model, preprocess, device, crop, id_embs)

            has_uniform = (u_lbl == "Uniform") and (u_prob >= args.uniform_thr)
            has_id = (i_lbl == "ID Card") and (i_prob >= args.id_thr)
            if has_uniform and has_id:
                color = (0, 200, 0)  # green
            elif has_uniform or has_id:
                color = (0, 165, 255)  # orange
            else:
                color = (0, 0, 220)  # red
            lines = [
                f"Uniform: {'Yes' if has_uniform else 'No'} ({u_prob:.2f})",
                f"ID Card: {'Yes' if has_id else 'No'} ({i_prob:.2f})",
                f"YOLO {conf:.2f}",
            ]
            draw_box(frame, (x1, y1, x2, y2), lines, color=color, alpha=0.3, thickness=2)

        draw_overlay_legend(frame, args.show_prompts, uniform_prompts["Uniform"], id_prompts["ID Card"])
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            args.show_prompts = not args.show_prompts

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
