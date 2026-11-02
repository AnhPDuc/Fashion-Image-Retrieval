# Demo – Fashion Image Retrieval (CS406, Đề tài 3)

Tải ảnh / chụp ảnh / chọn ảnh mẫu → trả về Top-K sản phẩm giống nhất.
Điểm giống nhau = **α · cos(ResNet50 fine-tune triplet) + (1 − α) · cos(HSV hist + LBP)**.

## Cấu trúc thư mục
```
demo/
├── app.py            # giao diện Streamlit
├── retrieval.py      # trích đặc trưng + tìm kiếm
├── requirements.txt
├── assets/           # lấy từ /kaggle/working sau khi chạy notebook
│   ├── ft_resnet50_triplet.pt
│   ├── demo_assets.npz
│   └── demo_gallery.csv
└── images/           # ảnh dataset: images/<id>.jpg (để hiển thị kết quả)
```

## Cài đặt & chạy
```bash
pip install -r requirements.txt
streamlit run app.py
```
Chạy được trên CPU (mỗi truy vấn ~vài chục ms – 1 giây). Có thể đổi đường dẫn trong thanh bên,
hoặc đặt biến môi trường `DEMO_ASSETS`, `DEMO_IMAGES`.

## Lấy ảnh dataset
Tải "Fashion Product Images (Small)" (paramaggarwal) trên Kaggle, giải nén, copy thư mục `images/` vào `demo/images/`.
Nếu không có ảnh, demo vẫn chạy nhưng kết quả chỉ hiện chữ (loại, màu, điểm).

## Chức năng
- 3 nguồn truy vấn: tải ảnh, camera, ảnh ngẫu nhiên trong gallery (tự loại chính nó khỏi kết quả).
- Top-K 3–20; α theo mục tiêu "đúng loại" (α cao), "đúng loại + màu" (α thấp) hoặc tự chỉnh.
- Lọc theo articleType / baseColour; xem điểm deep vs thủ công.

## Lưu ý
- `demo_assets.npz` phải do notebook v3 tạo (có `hc_name`), nếu không vector thủ công sẽ lệch không gian.
- Ảnh chụp nền đơn giản/sáng cho kết quả tốt nhất (dataset là ảnh studio nền trắng).
