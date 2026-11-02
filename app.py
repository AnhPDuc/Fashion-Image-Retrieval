"""Demo Streamlit – Fashion Image Retrieval (CS406).

Chạy:  streamlit run app.py
Cần: thư mục assets/ (ft_resnet50_triplet.pt, demo_assets.npz, demo_gallery.csv)
     và thư mục images/ chứa ảnh dataset (id.jpg) để hiển thị kết quả.
"""
import os
import numpy as np
import cv2
import streamlit as st
import retrieval as R

# Streamlit mới dùng width='stretch', bản cũ dùng use_container_width=True
_v = tuple(int(x) for x in st.__version__.split('.')[:2] if x.isdigit())
FULL = {'width': 'stretch'} if _v >= (1, 50) else {'use_container_width': True}

st.set_page_config(page_title='Fashion Image Retrieval', page_icon='👗', layout='wide')

# ------------------------------------------------------------------ cấu hình đường dẫn
# Đường dẫn cố định, tính theo vị trí app.py (đổi bằng biến môi trường DEMO_ASSETS / DEMO_IMAGES nếu cần)
HERE = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.environ.get('DEMO_ASSETS', os.path.join(HERE, 'assets'))
images_dir = os.environ.get('DEMO_IMAGES', os.path.join(HERE, 'images'))


@st.cache_resource(show_spinner='Đang nạp embedding gallery...')
def get_assets(d):
    return R.load_assets(d)


@st.cache_resource(show_spinner='Đang nạp mô hình ResNet50 đã fine-tune...')
def get_embedder(d):
    return R.load_embedder(os.path.join(d, 'ft_resnet50_triplet.pt'))


try:
    G_ft, G_hc, meta, cfg = get_assets(assets_dir)
    embedder = get_embedder(assets_dir)
except Exception as e:                                    # thiếu file / sai đường dẫn
    st.error(f'Không nạp được dữ liệu từ "{assets_dir}": {e}')
    st.info('Cần có trong thư mục assets: `ft_resnet50_triplet.pt`, `demo_assets.npz`, `demo_gallery.csv` '
            '(do notebook huấn luyện tạo ra). Xem README.')
    st.stop()

if not os.path.isdir(images_dir):
    st.warning(f'Chưa tìm thấy thư mục ảnh dataset ({images_dir}) → kết quả chỉ hiện dạng chữ.')

# ------------------------------------------------------------------ điều khiển
st.title('👗 Fashion Image Retrieval')
st.caption('Tìm sản phẩm thời trang giống ảnh truy vấn nhất. Mô hình: ResNet50 fine-tune bằng triplet loss '
           f'kết hợp đặc trưng thủ công ({cfg["hc_name"]}) – hệ số α điều chỉnh được.')

st.sidebar.header('🔎 Tìm kiếm')
k = st.sidebar.slider('Số kết quả (Top-K)', 3, 20, 10)
mode = st.sidebar.radio('Ưu tiên', [
    f'Giống về LOẠI sản phẩm (α = {cfg["alpha"]:.2f})',
    f'Giống cả LOẠI và MÀU (α = {cfg["alpha_colour"]:.2f})',
    'Tự chỉnh α', ])
if mode.startswith('Giống về LOẠI'):
    alpha = cfg['alpha']
elif mode.startswith('Giống cả'):
    alpha = cfg['alpha_colour']
else:
    alpha = st.sidebar.slider('α  (0 = chỉ màu/kết cấu, 1 = chỉ mô hình deep)', 0.0, 1.0, 0.5, 0.05)

with st.sidebar.expander('Lọc kết quả theo metadata'):
    types = st.multiselect('Loại sản phẩm', sorted(meta['articleType'].unique()))
    colours = st.multiselect('Màu', sorted(meta['baseColour'].astype(str).unique()))
show_scores = st.sidebar.checkbox('Hiện chi tiết điểm số', value=False)

# ------------------------------------------------------------------ nguồn ảnh truy vấn
tab_up, tab_cam, tab_sample = st.tabs(['📁 Tải ảnh lên', '📷 Chụp ảnh', '🎲 Ảnh mẫu trong gallery'])
# Truy vấn chỉ chạy khi bấm nút "Tìm kiếm"; kết quả được giữ lại để đổi Top-K / α / bộ lọc không phải tìm lại.
with tab_up:
    f = st.file_uploader('Chọn ảnh sản phẩm (jpg/png)', type=['jpg', 'jpeg', 'png', 'webp'])
    if st.button('🔍 Tìm kiếm', key='go_up', type='primary', disabled=f is None):
        img = R.decode_image(f.getvalue())
        if img is None:
            st.error('Không đọc được file ảnh này.')
        else:
            st.session_state['query'] = dict(img=img, exclude=None, source=f.name)
with tab_cam:
    c = st.camera_input('Chụp ảnh sản phẩm (nền đơn giản cho kết quả tốt hơn)')
    if st.button('🔍 Tìm kiếm', key='go_cam', type='primary', disabled=c is None):
        st.session_state['query'] = dict(img=R.decode_image(c.getvalue()), exclude=None, source='camera')
with tab_sample:
    if st.button('🎲 Chọn ngẫu nhiên và tìm kiếm', key='go_sample', type='primary'):
        i = int(np.random.randint(len(meta)))
        p = R.image_path(meta.iloc[i], images_dir)
        if p is None:
            st.warning('Không tìm thấy file ảnh của sản phẩm này – kiểm tra thư mục ảnh dataset.')
        else:
            st.session_state['query'] = dict(
                img=cv2.imread(p), exclude=i,
                source=f'gallery #{meta.iloc[i]["id"]} ({meta.iloc[i]["articleType"]}, {meta.iloc[i]["baseColour"]})')

if 'query' not in st.session_state:
    st.info('👆 Chọn ảnh rồi bấm **Tìm kiếm** để bắt đầu.')
    st.stop()
query_bgr, exclude, source = (st.session_state['query'][k] for k in ('img', 'exclude', 'source'))

# ------------------------------------------------------------------ truy vấn
# Tách nền + trích đặc trưng chỉ tính 1 lần cho mỗi truy vấn; đổi α / Top-K / bộ lọc dùng lại kết quả này.
fkey = id(st.session_state['query'])
if st.session_state.get('feat_key') != fkey:
    with st.spinner('Đang tìm kiếm...'):
        query_proc, bg_removed = R.remove_background(query_bgr)   # luôn tách nền (ảnh nền trắng thì giữ nguyên)
        pad = R.resize_with_padding(query_proc)
        st.session_state['feat'] = dict(pad=pad, bg_removed=bg_removed,
                                        q_ft=embedder(pad[None])[0], q_hc=R.hc_vector(pad, cfg['hc_name']))
        st.session_state['feat_key'] = fkey
pad, bg_removed, q_ft, q_hc = (st.session_state['feat'][k] for k in ('pad', 'bg_removed', 'q_ft', 'q_hc'))

allowed = None
if types or colours:
    allowed = np.ones(len(meta), bool)
    if types:
        allowed &= meta['articleType'].isin(types).values
    if colours:
        allowed &= meta['baseColour'].astype(str).isin(colours).values

idx, score, s_ft, s_hc = R.search(q_ft, q_hc, G_ft, G_hc, alpha, k, allowed, exclude)

left, right = st.columns([1, 4])
with left:
    st.subheader('Ảnh truy vấn')
    st.image(cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB), caption=source, **FULL)
    with st.expander('Ảnh đưa vào mô hình' + (' (đã tách nền)' if bg_removed else '')):
        st.image(cv2.cvtColor(pad, cv2.COLOR_BGR2RGB), **FULL)
with right:
    st.subheader(f'Top-{len(idx)} kết quả')
    if len(idx) == 0:
        st.warning('Không có sản phẩm nào khớp bộ lọc.')
    per_row = 5
    for start in range(0, len(idx), per_row):
        cols = st.columns(per_row)
        for col, j, sc, a, b in zip(cols, idx[start:start + per_row], score[start:start + per_row],
                                    s_ft[start:start + per_row], s_hc[start:start + per_row]):
            row = meta.iloc[int(j)]
            with col:
                p = R.image_path(row, images_dir)
                if p:
                    st.image(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), **FULL)
                else:
                    st.write('🖼️ (không có file ảnh)')
                st.markdown(f'**{row["articleType"]}**  \n{row["baseColour"]} · điểm {sc:.3f}')
                if show_scores:
                    st.caption(f'deep {a:.3f} | thủ công {b:.3f}')

with st.expander('ℹ️ Cách hoạt động'):
    st.markdown(f"""
1. Ảnh được resize giữ tỉ lệ về 128×128 (đệm nền trắng).
2. **Deep:** ResNet50 đã fine-tune bằng *batch-hard triplet loss* → vector 256 chiều.
   **Thủ công:** {cfg['hc_name']} – histogram màu HSV + LBP theo lưới.
3. Điểm giống nhau = **α · cos(deep) + (1 − α) · cos(thủ công)**, xếp hạng trên toàn gallery ({len(meta):,} ảnh).
4. α lớn → ưu tiên *đúng loại sản phẩm*; α nhỏ → ưu tiên *đúng màu và kết cấu*.
""")
