import torch
import cv2
import numpy as np
import base64
from PIL import Image

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def integrated_gradients(input_tensor, model, gallery_embed, device, steps=30):
    model.eval()
    gallery_t = torch.tensor(
        gallery_embed, dtype=torch.float32, device=device
    ).unsqueeze(0)

    # Use the mean pixel value as baseline — closer to the natural image
    # manifold than zero, which reduces path oscillations.
    baseline = torch.full_like(input_tensor, input_tensor.mean().item())
    grad_sum = torch.zeros_like(input_tensor)

    for k in range(steps + 1):
        alpha  = k / steps
        interp = (baseline + alpha * (input_tensor - baseline)).detach().requires_grad_(True)

        output = model(interp)
        score  = torch.mm(output, gallery_t.t()).squeeze()
        score.backward()

        grad_sum += interp.grad.detach()

    avg_grads = grad_sum / (steps + 1)
    ig        = avg_grads * (input_tensor - baseline).detach()

    attr = ig.abs().sum(dim=1).squeeze().cpu().numpy()

    # Smooth out high-frequency gradient noise
    attr = cv2.GaussianBlur(attr, (0, 0), sigmaX=4)

    if attr.max() > 0:
        attr /= attr.max()

    return attr


def apply_heatmap(image_np, heatmap):
    heatmap_resized = cv2.resize(
        heatmap, (image_np.shape[1], image_np.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_np, 0.4, heatmap_color, 0.6, 0)


def _face_mask(h, w, face_rect=None):
    """Return a soft elliptical mask [0,1] the size of (h, w).
    If face_rect=(x,y,fw,fh) is given the ellipse is fitted to that box;
    otherwise it covers the upper-centre of the image (typical portrait crop).
    """
    mask = np.zeros((h, w), dtype=np.float32)

    if face_rect is not None:
        x, y, fw, fh = face_rect
        cx, cy = x + fw // 2, y + fh // 2
        rx, ry = int(fw * 0.55), int(fh * 0.60)
    else:
        cx = w // 2
        cy = int(h * 0.35)
        rx = int(w * 0.35)
        ry = int(h * 0.30)

    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
    # Blur the hard ellipse edge for a gentle fall-off
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(rx, ry) * 0.4)
    if mask.max() > 0:
        mask /= mask.max()
    return mask


def _detect_face(img_bgr):
    """Try to find a face with progressively looser parameters.
    Returns (x, y, w, h) of the largest detection, or None."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ih, iw = img_bgr.shape[:2]
    min_dim = min(ih, iw)

    for neighbors, min_size in [(4, max(20, min_dim // 6)),
                                  (3, max(15, min_dim // 8)),
                                  (2, max(10, min_dim // 10))]:
        faces = _face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=neighbors,
            minSize=(min_size, min_size)
        )
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            return tuple(faces[0])
    return None


def get_base64_heatmap(face_img, model, transform, device, gallery_embed):
    """Returns (heatmap_b64, crop_b64) or (None, None) if no face found."""
    try:
        ih, iw = face_img.shape[:2]

        face_rect = _detect_face(face_img)
        if face_rect is None:
            return None, None

        x, y, fw, fh = face_rect
        pad = int(max(fw, fh) * 0.15)
        x1 = max(0, x - pad);  y1 = max(0, y - pad)
        x2 = min(iw, x + fw + pad); y2 = min(ih, y + fh + pad)
        face_crop = face_img[y1:y2, x1:x2]

        # Encode the raw crop for the "original" panel
        _, cbuf = cv2.imencode('.jpg', face_crop)
        crop_b64 = base64.b64encode(cbuf).decode('utf-8')

        img_rgb      = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        input_tensor = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(device)

        attr = integrated_gradients(input_tensor, model, gallery_embed, device)

        ch, cw = face_crop.shape[:2]
        mask = _face_mask(ch, cw, face_rect=(0, 0, cw, ch))

        attr_resized = cv2.resize(attr, (cw, ch), interpolation=cv2.INTER_CUBIC)
        attr_masked  = attr_resized * mask
        if attr_masked.max() > 0:
            attr_masked /= attr_masked.max()

        overlay = apply_heatmap(face_crop, attr_masked)

        _, hbuf = cv2.imencode('.jpg', overlay)
        heatmap_b64 = base64.b64encode(hbuf).decode('utf-8')

        return heatmap_b64, crop_b64

    except Exception as e:
        print(f"XAI Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None
