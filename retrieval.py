"""Lõi tìm kiếm cho demo Fashion Image Retrieval (không phụ thuộc Streamlit).

Giữ ĐÚNG cách tính đặc trưng như trong notebook huấn luyện, để embedding của ảnh truy vấn
nằm cùng không gian với embedding của gallery đã lưu trong demo_assets.npz.
"""
import os
import numpy as np
import pandas as pd
import cv2
from skimage.feature import local_binary_pattern

IMG = 128  # độ phân giải làm việc (giống notebook)


# ----------------------------------------------------------------------------- tiện ích ảnh
def resize_with_padding(img, size=IMG):
    """Resize giữ tỉ lệ rồi đệm nền trắng thành ảnh vuông size×size."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    canvas = np.full((size, size, 3), 255, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return canvas


def decode_image(data: bytes):
    """Bytes (jpg/png...) -> ảnh BGR uint8, hoặc None nếu không đọc được."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


# ----------------------------------------------------------------------------- tiền xử lý
def denoise_image(img):
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)


def bg_floodfill(img, thr=235):
    """Nền = vùng gần trắng LIỀN VỚI VIỀN ảnh (giữ phần trắng bên trong sản phẩm)."""
    near_white = (img.min(axis=2) >= thr).astype(np.uint8)
    _, lab_, _, _ = cv2.connectedComponentsWithStats(near_white, connectivity=4)
    border = np.unique(np.concatenate([lab_[0], lab_[-1], lab_[:, 0], lab_[:, -1]]))
    border = border[border != 0]
    bg = np.isin(lab_, border) & (near_white > 0)
    mask = (~bg).astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    ratio = (mask > 0).mean()
    if ratio < 0.05 or ratio > 0.95:
        mask[:] = 255
    return np.where(mask[..., None] > 0, img, 255).astype(np.uint8), mask


def apply_clahe(img):
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def preprocess_image(img):
    out, mask = bg_floodfill(denoise_image(img))
    return apply_clahe(out), mask


# ----------------------------------------------------------------------------- handcrafted
def l2n(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def hc_cells(img, mask, g):
    """HSV histogram 8x4x4 + LBP uniform 10 bin cho mỗi ô của lưới g×g; chuẩn hóa tổng=1 rồi lấy căn."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, 'uniform').astype(int)
    H, W = gray.shape
    hs, ls = [], []
    for i in range(g):
        for j in range(g):
            ys, ye, xs, xe = i * H // g, (i + 1) * H // g, j * W // g, (j + 1) * W // g
            mm = None if mask is None else np.ascontiguousarray(mask[ys:ye, xs:xe])
            h = cv2.calcHist([np.ascontiguousarray(hsv[ys:ye, xs:xe])], [0, 1, 2], mm,
                             [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
            lv = lbp[ys:ye, xs:xe] if mm is None else lbp[ys:ye, xs:xe][mm > 0]
            l = np.bincount(lv.ravel(), minlength=10)[:10].astype(np.float32)
            hs.append(np.sqrt(h / (h.sum() + 1e-9)))
            ls.append(np.sqrt(l / (l.sum() + 1e-9)))
    return np.concatenate(hs).astype(np.float32), np.concatenate(ls).astype(np.float32)


def hc_vector(img_pad, name):
    """Vector handcrafted (đã chuẩn hóa) của 1 ảnh 128×128, theo tên biến thể trong notebook.

    Hỗ trợ: 'HC raw gxg' (ảnh gốc, không mask) và 'HC proc+mask gxg' (pipeline: khử nhiễu + flood-fill + CLAHE).
    """
    g = int(name[-3])
    if name.startswith('HC raw'):
        h, l = hc_cells(cv2.GaussianBlur(img_pad, (3, 3), 0), None, g)
    elif name.startswith('HC proc+mask'):
        p, m = preprocess_image(img_pad)
        h, l = hc_cells(p, m, g)
    else:
        raise ValueError(f'Biến thể handcrafted chưa hỗ trợ trong demo: {name}')
    v = np.concatenate([l2n(h[None])[0], l2n(l[None])[0]])
    return l2n(v[None])[0].astype(np.float32)


# ----------------------------------------------------------------------------- mô hình deep
class Embedder:
    """ResNet50 đã fine-tune (triplet) -> embedding 256 chiều, L2-normalize. Chạy được trên CPU."""

    def __init__(self, weights_path, dim=256):
        import torch
        import torch.nn as nn
        import torch.nn.functional as TF
        import torchvision

        class EmbNet(nn.Module):
            def __init__(self):
                super().__init__()
                r = torchvision.models.resnet50(weights=None)   # không cần tải weights ImageNet; sẽ nạp từ file
                self.body = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool,
                                          r.layer1, r.layer2, r.layer3, r.layer4)
                self.fc = nn.Linear(2048, dim)

            def forward(self, x):
                return TF.normalize(self.fc(self.body(x).mean((2, 3))).float(), dim=1)

        self.torch = torch
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = EmbNet()
        try:
            state = torch.load(weights_path, map_location='cpu', weights_only=True)
        except TypeError:                                       # torch cũ chưa có weights_only
            state = torch.load(weights_path, map_location='cpu')
        net.load_state_dict(state)
        self.net = net.to(self.dev).eval()
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def __call__(self, imgs_bgr):
        """imgs_bgr: (N,128,128,3) uint8 BGR -> (N,dim) float32."""
        torch = self.torch
        x = np.ascontiguousarray(imgs_bgr[..., ::-1]).astype(np.float32) / 255.0
        x = torch.from_numpy(((x - self.mean) / self.std).transpose(0, 3, 1, 2).copy()).to(self.dev)
        with torch.no_grad():
            return self.net(x).float().cpu().numpy()


# ----------------------------------------------------------------------------- dữ liệu gallery
def load_assets(assets_dir):
    """Nạp embedding gallery + metadata + cấu hình (α, tên biến thể handcrafted)."""
    z = np.load(os.path.join(assets_dir, 'demo_assets.npz'), allow_pickle=False)
    keys = set(z.files)
    if 'hc_name' not in keys:
        raise ValueError('demo_assets.npz thiếu "hc_name" – hãy dùng file do notebook v3 tạo ra.')
    meta = pd.read_csv(os.path.join(assets_dir, 'demo_gallery.csv'))
    G_ft = l2n(z['gal_emb_ft'].astype(np.float32))
    G_hc = l2n(z['gal_emb_hc'].astype(np.float32))
    assert len(meta) == len(G_ft) == len(G_hc), 'Số dòng metadata và embedding không khớp'
    cfg = dict(alpha=float(z['alpha']),
               alpha_colour=float(z['alpha_colour']) if 'alpha_colour' in keys else float(z['alpha']),
               hc_name=str(z['hc_name']))
    return G_ft, G_hc, meta, cfg


def image_path(row, images_dir):
    """Tìm file ảnh của 1 dòng metadata: ưu tiên thư mục ảnh cục bộ, sau đó đường dẫn trong CSV."""
    cand = [os.path.join(images_dir, f"{row['id']}.jpg")] if images_dir else []
    if 'path' in row and isinstance(row['path'], str):
        cand.append(row['path'])
    for p in cand:
        if os.path.exists(p):
            return p
    return None


# ----------------------------------------------------------------------------- tìm kiếm
def search(q_ft, q_hc, G_ft, G_hc, alpha, k, allowed=None, exclude=None):
    """Xếp hạng gallery theo  sim = α·cos(fine-tuned) + (1−α)·cos(handcrafted).

    allowed: mảng bool (lọc theo metadata); exclude: chỉ số bị loại (khi truy vấn bằng ảnh có sẵn trong gallery).
    Trả về (chỉ số, điểm tổng, điểm fine-tuned, điểm handcrafted).
    """
    s_ft, s_hc = G_ft @ q_ft, G_hc @ q_hc
    s = alpha * s_ft + (1 - alpha) * s_hc
    if allowed is not None:
        s = np.where(allowed, s, -np.inf)
    if exclude is not None:
        s[exclude] = -np.inf
    k = int(min(k, np.isfinite(s).sum()))
    idx = np.argpartition(-s, k - 1)[:k] if k > 0 else np.array([], int)
    idx = idx[np.argsort(-s[idx])]
    return idx, s[idx], s_ft[idx], s_hc[idx]


def load_embedder(weights_path):
    """Nạp mô hình (import torch chỉ khi cần, để phần còn lại test được không cần torch)."""
    return Embedder(weights_path)


# ----------------------------------------------------------------------------- tách nền cho ảnh người dùng
def border_is_white(img, thr=235, frac=0.9):
    """True nếu viền ảnh gần như trắng (ảnh studio) -> không cần tách nền."""
    b = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    return float((b.min(axis=1) >= thr).mean()) >= frac


def remove_background(img, max_side=512, iters=6, margin=0.04):
    """Tách sản phẩm khỏi nền bất kỳ bằng GrabCut, đặt lên nền trắng và cắt sát vật thể.

    Trả về (ảnh BGR mới, True) nếu tách được; (ảnh gốc, False) nếu kết quả không đáng tin
    (vùng tách quá nhỏ/quá lớn). Ảnh nền trắng sẵn thì giữ nguyên.
    """
    if border_is_white(img):
        return img, False
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    H, W = small.shape[:2]
    m = max(2, int(margin * min(H, W)))
    rect = (m, m, W - 2 * m, H - 2 * m)
    mask = np.zeros((H, W), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return img, False
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab_, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return img, False
    fg = (lab_ == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)   # giữ vật thể lớn nhất
    # lấp lỗ bên trong vật thể
    inv = (1 - fg).astype(np.uint8)
    _, l2, _, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    border = np.unique(np.concatenate([l2[0], l2[-1], l2[:, 0], l2[:, -1]]))
    fg = (fg > 0) | ~np.isin(l2, border)
    ratio = fg.mean()
    if ratio < 0.03 or ratio > 0.92:
        return img, False
    fg = cv2.resize(fg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    out = np.where(fg[..., None], img, 255).astype(np.uint8)
    ys, xs = np.where(fg)
    pad = int(0.04 * max(h, w))
    y0, y1, x0, x1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1), max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    return out[y0:y1, x0:x1], True
