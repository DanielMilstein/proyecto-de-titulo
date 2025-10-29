# yolov2_onnx_infer_flexible.py
import argparse
import os
import glob
import time
from pathlib import Path

import numpy as np
import cv2

try:
    import onnxruntime as ort
except ImportError as e:
    raise SystemExit("Falta onnxruntime. Instala: pip install onnxruntime o onnxruntime-gpu") from e

# Anchors habituales de YOLOv2
VOC_ANCHORS = [(1.08,1.19), (3.42,4.41), (6.63,11.38), (9.42,5.11), (16.62,10.52)]
COCO_ANCHORS = [(0.57273,0.677385), (1.87446,2.06253), (3.33843,5.47434), (7.88282,3.52778), (9.77052,9.16828)]

# Clases VOC por si quieres testear rápido con num_classes=20 y sin archivo de nombres
VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"
]

def load_class_names(path: str, num_classes: int):
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        if len(names) != num_classes:
            print(f"[WARN] {len(names)} nombres en {path}, pero num_classes={num_classes}. Continuando igual.")
        return names
    # fallback genérico
    if num_classes == 20:
        return VOC_CLASSES
    return [f"class_{i}" for i in range(num_classes)]

def parse_anchors(arg: str):
    # formato: "w1,h1 w2,h2 ..."
    out = []
    for pair in arg.strip().split():
        w, h = pair.split(",")
        out.append((float(w), float(h)))
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)

def nms_xyxy(boxes, scores, iou_thres=0.45):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)
        inds = np.where(iou < iou_thres)[0]
        order = order[inds + 1]
    return keep

def preprocess_bgr(img_bgr, size):
    # YOLOv2 suele reescalar directo (sin letterbox). Ajusta si tu export asumió letterbox.
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(img_rgb, (2, 0, 1))[None]  # NCHW
    return blob

def draw_detections(img_bgr, boxes_xyxy_norm, scores, cls_ids, names):
    h, w = img_bgr.shape[:2]
    for (x1, y1, x2, y2), s, c in zip(boxes_xyxy_norm, scores, cls_ids):
        x1i = int(max(0, min(w - 1, x1 * w)))
        y1i = int(max(0, min(h - 1, y1 * h)))
        x2i = int(max(0, min(w - 1, x2 * w)))
        y2i = int(max(0, min(h - 1, y2 * h)))
        cv2.rectangle(img_bgr, (x1i, y1i), (x2i, y2i), (0,255,0), 2)
        cname = names[c] if c < len(names) else str(c)
        label = f"{cname}: {s:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(th + 3, y1i)
        cv2.rectangle(img_bgr, (x1i, ty - th - 3), (x1i + tw + 2, ty + baseline - 3), (0,0,0), -1)
        cv2.putText(img_bgr, label, (x1i + 1, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return img_bgr

# ------------------- Decodificador flexible -------------------

def _softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)

def _nms_per_class(boxes, scores, cls_ids, iou_thres):
    if len(boxes) == 0:
        return np.empty((0,4), np.float32), np.array([]), np.array([])
    fb, fs, fc = [], [], []
    for c in np.unique(cls_ids):
        m = cls_ids == c
        keep = nms_xyxy(boxes[m], scores[m], iou_thres=iou_thres)
        if keep:
            fb.append(boxes[m][keep])
            fs.append(scores[m][keep])
            fc.append(np.full(len(keep), int(c), dtype=int))
    if fb:
        return np.concatenate(fb), np.concatenate(fs), np.concatenate(fc)
    return np.empty((0,4), np.float32), np.array([]), np.array([])

def decode_flexible_yolov2(outs, num_classes, anchors, conf_thres=0.25, iou_thres=0.45):
    """
    Soporta exportes típicos y variantes mañosas:
      1) Única salida [1, C, H, W] con C = A*(5+K)
      2) Única salida "lista" [1, S, (5+K)] con S = H*W*A
      3) Dos salidas: boxes [1, S, 4] + scores [1, S, K] (K==num_classes o num_classes+1)
      4) Raro: [1, C, 1, 1] con C = H*W*A  -> se reconstruye a [1, S, 5+K] si se puede
      5) Raro: dos salidas arbitrarias -> se aplanan y concatenan para formar [1, S, 5+K]
    Devuelve: boxes_norm_xyxy, scores, cls_ids
    """

    # --- Normalización de formas torpes: squeeze dims inútiles ---
    norm_outs = []
    for o in outs:
        arr = np.asarray(o)
        # Si viene como (1, S, 1, 4) -> (1, S, 4)
        if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[2] == 1 and arr.shape[-1] in (1,4):
            arr = arr.reshape(1, arr.shape[1], arr.shape[-1])
        # Si viene como (1, S, 1) -> (1, S, 1)
        elif arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] == 1:
            arr = arr.reshape(1, arr.shape[1], 1)
        norm_outs.append(arr)
    outs = norm_outs
    
    expK = 5 + num_classes
    A = len(anchors)

    def _decode_map_style(output, num_classes, anchors, conf_thres):
        _, C, H, W = output.shape
        expC = A * expK
        if C != expC:
            raise ValueError(f"Salida map-style con C={C} no coincide con {expC}")
        out = output.reshape(A, expK, H, W).transpose(0, 2, 3, 1)
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        boxes = []; scores = []; cls_ids = []
        for a in range(A):
            pred = out[a]
            tx, ty, tw, th, tc = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3], pred[..., 4]
            tcls = pred[..., 5:]
            bx = (1/(1+np.exp(-tx)) + grid_x) / W
            by = (1/(1+np.exp(-ty)) + grid_y) / H
            bw = (np.exp(tw) * anchors[a][0]) / W
            bh = (np.exp(th) * anchors[a][1]) / H
            conf = 1/(1+np.exp(-tc))
            cls_prob = softmax(tcls)
            cls_id = np.argmax(cls_prob, axis=-1)
            cls_sc = np.max(cls_prob, axis=-1)
            sc = conf * cls_sc
            mask = sc >= conf_thres
            if not np.any(mask): 
                continue
            x1 = (bx - bw/2)[mask]; y1 = (by - bh/2)[mask]
            x2 = (bx + bw/2)[mask]; y2 = (by + bh/2)[mask]
            boxes.append(np.stack([x1,y1,x2,y2], axis=-1))
            scores.append(sc[mask]); cls_ids.append(cls_id[mask])
        if not boxes:
            return np.empty((0,4), np.float32), np.array([]), np.array([])
        return np.concatenate(boxes), np.concatenate(scores), np.concatenate(cls_ids)

    def _decode_list_style(output, num_classes, anchors, conf_thres):
        # output: [1, S, (5+num_classes)]
        out = np.squeeze(output, axis=0)
        if out.ndim != 2:
            raise ValueError(f"Esperaba [1,S,5+cls], recibí {output.shape}")
        S, K = out.shape
        if K != expK:
            raise ValueError(f"K={K} no coincide con 5+num_classes={expK}")
        # reconstruye indices de celda/anchor lo mejor posible
        HW = max(1, S // A)
        H = W = int(round(np.sqrt(HW)))
        tx, ty, tw, th, tc = out[:,0], out[:,1], out[:,2], out[:,3], out[:,4]
        tcls = out[:,5:]
        conf = 1/(1+np.exp(-tc))
        cls_prob = _softmax(tcls)
        cls_id = np.argmax(cls_prob, axis=-1)
        cls_sc = np.max(cls_prob, axis=-1)
        sc = conf * cls_sc
        mask = sc >= conf_thres
        if not np.any(mask):
            return np.empty((0,4), np.float32), np.array([]), np.array([])
        idxs = np.arange(S)
        a_idx = idxs % A
        cell = idxs // A
        j = cell % max(W,1)
        i = cell // max(W,1)
        bx = (1/(1+np.exp(-tx)) + j) / max(W,1)
        by = (1/(1+np.exp(-ty)) + i) / max(H,1)
        bw = (np.exp(tw) * np.array([anchors[a][0] for a in a_idx])) / max(W,1)
        bh = (np.exp(th) * np.array([anchors[a][1] for a in a_idx])) / max(H,1)
        bx = bx[mask]; by = by[mask]; bw = bw[mask]; bh = bh[mask]
        x1 = bx - bw/2; y1 = by - bh/2; x2 = bx + bw/2; y2 = by + bh/2
        boxes = np.stack([x1,y1,x2,y2], axis=-1)
        return boxes, sc[mask], cls_id[mask]

    def _decode_two_outputs(box_out, score_out, num_classes, conf_thres):
        """
        boxes:  [1, S, 4]  en XYWH (normalizado [0,1] o en píxeles)
        scores: [1, S, 1]  objectness (logits o prob); si fueran [1,S,K] con K>1 se toma columna 0
        Devuelve: boxes_xyxy_norm, scores, cls_ids
        """
        B = np.squeeze(np.asarray(box_out), 0)      # (S,4)
        S = B.shape[0]

        S_probs = np.squeeze(np.asarray(score_out), 0)  # (S,) o (S,1) o (S,K)
        if S_probs.ndim == 1:
            obj_logit = S_probs
        elif S_probs.ndim == 2:
            # si K==1 es objectness; si K>1 asumimos que la primera columna es objectness
            obj_logit = S_probs[:, 0]
        else:
            raise ValueError(f"score_out con forma no soportada: {score_out.shape}")

        # objectness -> prob
        obj = 1.0 / (1.0 + np.exp(-obj_logit))

        # con 1 clase, score_final = objectness
        sc = obj

        # ¿cajas normalizadas o en píxeles?
        # si vemos valores mayores que 1 en w/h/x/y, asumimos píxeles y normalizamos a 416
        # ajusta IMG_SIZE si usas otro --imgsz
        IMG_SIZE = 416.0
        px_scale = 1.0
        if np.nanmax(B) > 1.0:
            px_scale = IMG_SIZE

        x, y, w, h = B[:, 0] / px_scale, B[:, 1] / px_scale, B[:, 2] / px_scale, B[:, 3] / px_scale
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0
        boxes = np.stack([x1, y1, x2, y2], axis=-1)

        # filtra por confianza
        mask = sc >= conf_thres
        if not np.any(mask):
            return np.empty((0, 4), np.float32), np.array([]), np.array([])

        boxes = boxes[mask]
        sc = sc[mask]

        # única clase -> todos 0
        cls_id = np.zeros(len(sc), dtype=int)
        return boxes, sc, cls_id

    # 1) Un solo output
    if len(outs) == 1:
        out = outs[0]
        if out.ndim == 4:
            _, C, H, W = out.shape
            # Caso raro: [1, C, 1, 1] con C múltiplo de (5+K)
            if H == 1 and W == 1 and (C % expK == 0):
                S = C // expK
                arr = out.reshape(1, C).reshape(1, S, expK)
                return _nms_per_class(*_decode_list_style(arr, num_classes, anchors, conf_thres), iou_thres)
            # Caso map-style normal o fallback a lista
            try:
                boxes, scores, cls_ids = _decode_map_style(out, num_classes, anchors, conf_thres)
            except ValueError:
                flat = np.squeeze(out)
                if flat.ndim == 1 and (flat.size % expK == 0):
                    S = flat.size // expK
                    arr = flat.reshape(1, S, expK)
                    boxes, scores, cls_ids = _decode_list_style(arr, num_classes, anchors, conf_thres)
                else:
                    raise
        elif out.ndim == 3:
            boxes, scores, cls_ids = _decode_list_style(out, num_classes, anchors, conf_thres)
        elif out.ndim == 2:
            flat = np.squeeze(out)
            if flat.size % expK != 0:
                raise ValueError(f"No puedo reconstruir [1,S,{expK}] desde {out.shape}")
            S = flat.size // expK
            arr = flat.reshape(1, S, expK)
            boxes, scores, cls_ids = _decode_list_style(arr, num_classes, anchors, conf_thres)
        else:
            raise ValueError(f"Formato de salida no soportado: {out.shape}")

        return _nms_per_class(boxes, scores, cls_ids, iou_thres)

    # 2) Dos salidas
    a, b = outs[0], outs[1]

    # 2.a) Caso clásico boxes+scores
    if a.ndim == 3 and a.shape[-1] == 4:
        return _nms_per_class(*_decode_two_outputs(a, b, num_classes, conf_thres), iou_thres)
    if b.ndim == 3 and b.shape[-1] == 4:
        return _nms_per_class(*_decode_two_outputs(b, a, num_classes, conf_thres), iou_thres)

    # 2.b) Si alguna es 4D y tiene [1, C, 1, 1] con C % expK == 0, reinterpretar como lista
    for out in (a, b):
        if out.ndim == 4:
            _, C, H, W = out.shape
            if H == 1 and W == 1 and (C % expK == 0):
                S = C // expK
                arr = out.reshape(1, C).reshape(1, S, expK)
                try:
                    return _nms_per_class(*_decode_list_style(arr, num_classes, anchors, conf_thres), iou_thres)
                except Exception:
                    pass

    # 2.c) Plan C: aplanar y concatenar en ambos órdenes para formar [1,S,5+K]
    flat_a = a.reshape(1, -1)
    flat_b = b.reshape(1, -1)
    for first, second in ((flat_a, flat_b), (flat_b, flat_a)):
        total = first.shape[1] + second.shape[1]
        if total % expK == 0:
            arr = np.concatenate([first, second], axis=1).reshape(1, total // expK, expK)
            try:
                return _nms_per_class(*_decode_list_style(arr, num_classes, anchors, conf_thres), iou_thres)
            except Exception:
                continue

    # 2.d) Último intento: si alguna es 3D lista, úsala
    for out in (a, b):
        if out.ndim == 3:
            try:
                return _nms_per_class(*_decode_list_style(out, num_classes, anchors, conf_thres), iou_thres)
            except Exception:
                pass

    raise ValueError(f"No pude interpretar las salidas del modelo: {[o.shape for o in outs]}")

# ------------------- Main -------------------

def main():
    ap = argparse.ArgumentParser(description="YOLOv2 ONNX inferencia por carpeta (decodificación flexible)")
    ap.add_argument("--model", type=str, required=True, help="Ruta al modelo .onnx")
    ap.add_argument("--source", type=str, required=True, help="Imagen, patrón glob o carpeta")
    ap.add_argument("--save-dir", type=str, default="./out", help="Carpeta de salida")
    ap.add_argument("--imgsz", type=int, default=416, help="Tamaño de entrada (416 típico)")
    ap.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza")
    ap.add_argument("--iou", type=float, default=0.45, help="Umbral IoU NMS")
    ap.add_argument("--num-classes", type=int, default=20, help="Número de clases del modelo")
    ap.add_argument("--anchors-preset", type=str, default="voc", choices=["voc","coco","custom"], help="Preset de anchors")
    ap.add_argument("--anchors", type=str, default="", help='Anchors custom: "w1,h1 w2,h2 ..."')
    ap.add_argument("--classes", type=str, default="", help="Archivo con nombres de clases (uno por línea)")
    ap.add_argument("--providers", type=str, default="coreml,cpu", help="Proveedores ONNX: cuda,cpu,coreml,azure")
    args = ap.parse_args()

    # Anchors
    if args.anchors_preset == "voc":
        anchors = VOC_ANCHORS
    elif args.anchors_preset == "coco":
        anchors = COCO_ANCHORS
    else:
        if not args.anchors.strip():
            raise SystemExit("Con --anchors-preset custom debes pasar --anchors \"w,h ...\"")
        anchors = parse_anchors(args.anchors)

    # Providers
    provs = []
    for p in args.providers.split(","):
        p = p.strip().lower()
        if p == "cuda":
            provs.append("CUDAExecutionProvider")
        elif p == "cpu":
            provs.append("CPUExecutionProvider")
        elif p == "coreml":
            provs.append("CoreMLExecutionProvider")
        elif p == "azure":
            provs.append("AzureExecutionProvider")
    if not provs:
        provs = ["CPUExecutionProvider"]

    # Sesión ONNX
    try:
        session = ort.InferenceSession(args.model, providers=provs)
    except Exception as e:
        raise SystemExit(f"No se pudo crear la sesión ONNX: {e}")

    inputs = session.get_inputs()
    if len(inputs) != 1:
        print(f"[WARN] Se esperaban 1 input, el modelo tiene {len(inputs)}. Se usará el primero.")
    input_name = inputs[0].name

    outputs_info = session.get_outputs()
    if len(outputs_info) != 1:
        print(f"[WARN] Se esperaban 1 salida, el modelo tiene {len(outputs_info)}. Se intentará decodificación flexible.")
    # No bloqueamos por nombre; pedimos todas las salidas
    output_names = [o.name for o in outputs_info]

    # Fuente
    if os.path.isdir(args.source):
        patterns = [os.path.join(args.source, "*.*")]
    else:
        patterns = [args.source]
    image_paths = []
    for pat in patterns:
        image_paths.extend(glob.glob(pat))
    image_paths = [p for p in image_paths if p.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff"))]
    if not image_paths:
        raise SystemExit(f"No se encontraron imágenes en {args.source}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Nombres de clase
    names = load_class_names(args.classes, args.num_classes)

    total_t = 0.0
    for idx, path in enumerate(sorted(image_paths)):
        img0 = cv2.imread(path)
        if img0 is None:
            print(f"[WARN] No se pudo leer {path}")
            continue

        blob = preprocess_bgr(img0, args.imgsz)
        t0 = time.time()
        outs = session.run(None, {input_name: blob})  # TODAS las salidas
        dt = time.time() - t0
        total_t += dt

        try:
            boxes, scores, cls_ids = decode_flexible_yolov2(
                outs, args.num_classes, anchors,
                conf_thres=args.conf, iou_thres=args.iou
            )
        except Exception as e:
            # Intento de compatibilidad adicional: si la primera salida viene vectorizada.
            out0 = outs[0]
            if out0.ndim == 2:
                flat = np.squeeze(out0)
                expK = 5 + args.num_classes
                S = int(flat.size // expK)
                arr = flat.reshape(1, S, expK)
                boxes, scores, cls_ids = decode_flexible_yolov2([arr], args.num_classes, anchors, args.conf, args.iou)
            else:
                raise

        vis = img0.copy()
        vis = draw_detections(vis, boxes, scores, cls_ids, names)
        out_path = os.path.join(args.save_dir, Path(path).name)
        cv2.imwrite(out_path, vis)
        print(f"[{idx+1}/{len(image_paths)}] {path} -> {out_path}  boxes={len(boxes)}  time={dt*1000:.1f}ms")

    if len(image_paths):
        print(f"Promedio por imagen: {(total_t/len(image_paths))*1000:.1f} ms")

if __name__ == "__main__":
    main()


