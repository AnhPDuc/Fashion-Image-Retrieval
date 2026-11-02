"""Tải 3 file cần cho demo (mô hình, embedding gallery, metadata) từ GitHub Release vào ./assets.

Chạy:  python download_assets.py
"""
import os
import sys
import urllib.request

REPO = 'AnhPDuc/fashion-image-retrieval'
TAG = 'v1.0'
FILES = ['ft_resnet50_triplet.pt', 'demo_assets.npz', 'demo_gallery.csv']
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')


def progress(blocks, bsize, total):
    if total > 0:
        pct = min(100, blocks * bsize * 100 // total)
        print(f'\r  {pct:3d}%', end='', flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in FILES:
        dst = os.path.join(OUT, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f'Đã có {name}, bỏ qua.')
            continue
        url = f'https://github.com/{REPO}/releases/download/{TAG}/{name}'
        print(f'Tải {name} ...')
        try:
            urllib.request.urlretrieve(url, dst + '.part', progress)
            os.replace(dst + '.part', dst)
            print(f'\r  xong ({os.path.getsize(dst) / 1e6:.1f} MB)')
        except Exception as e:
            print(f'\n  Lỗi: {e}\n  Kiểm tra REPO/TAG trong file này và Release đã Publish chưa.')
            sys.exit(1)
    print('Hoàn tất. Chạy:  streamlit run app.py')


if __name__ == '__main__':
    main()
