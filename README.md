# Fashion Image Retrieval

Content-based image retrieval (CBIR) for fashion products: given a photo, return the most similar products from a gallery of ~22k images.
The system fine-tunes a **ResNet50** with **batch-hard triplet loss**, fuses it with handcrafted **HSV-histogram + LBP** features, and ships a **Streamlit demo** with automatic background removal.

> Course project – CS406 *Image Processing and Applications*, University of Information Technology (UIT-VNU HCM).

<!-- Add a screenshot or GIF of the demo here: ![demo](docs/demo.gif) -->

## Highlights
- **Leak-free evaluation:** query and gallery are disjoint, 8 whole classes are held out of training ("unseen"), near-duplicate images are removed (RMSE on 16×16 colour thumbnails), α and checkpoints are selected on a validation split and reported on test.
- **Fine-tuning helps a lot:** mAP 0.440 → **0.670** vs. pretrained ResNet50, and 0.282 → **0.425** on classes never seen in training.
- **Fast search:** FAISS IVF (nprobe=4) answers 1000 queries in ~11 ms vs. ~589 ms for brute-force NumPy, with the same P@10 (0.78) and 94% top-10 overlap with exact search.
- **Demo:** upload / camera / random gallery item, automatic background removal, adjustable α, metadata filters.

## Data
[Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) (≈44k images). Classes with <50 images are dropped (70 `articleType` classes remain). Relevance = same `articleType` (strict mode: same `articleType` **and** `baseColour`).
Split: ~50% of the seen classes for training, the rest for validation/test queries and the gallery; 8 classes are unseen. The dataset is **not** included in this repository.

## Method
```
image → resize 128×128 (white padding)
   ├─ ResNet50 (layer4 GAP → Linear 256 → L2)   fine-tuned with batch-hard triplet loss
   └─ HSV hist (8×4×4) + LBP (uniform) per cell of a 3×3 grid → Hellinger → L2
score = α · cos(deep) + (1 − α) · cos(handcrafted)
```
α = 0.95 favours matching the **product type**; α = 0.30 favours **type + colour**.

## Results (test set, 1000 queries, 25% from unseen classes)
| Method | P@1 | P@10 | mAP | mAP seen | mAP unseen |
|---|---|---|---|---|---|
| Random | 0.042 | 0.045 | 0.047 | 0.055 | 0.021 |
| Handcrafted (HSV+LBP, 3×3) | 0.601 | 0.480 | 0.183 | 0.209 | 0.105 |
| VGG16 pretrained (fc7) | 0.719 | 0.647 | 0.349 | 0.395 | 0.213 |
| ResNet50 pretrained (layer4 GAP) | 0.776 | 0.706 | 0.440 | 0.493 | 0.282 |
| ResNet50 pretrained (layer4 GeM) | 0.793 | 0.719 | 0.446 | 0.498 | 0.289 |
| **ResNet50 fine-tuned (triplet)** | 0.824 | 0.781 | 0.670 | 0.752 | 0.425 |
| **Fine-tuned + handcrafted, α=0.95** | 0.821 | 0.782 | **0.671** | 0.752 | **0.428** |
| Fine-tuned + handcrafted, α=0.30 (type + colour) | 0.812 | 0.753 | 0.614 | 0.685 | 0.401 |

Notes (honest reading):
- Fusion adds almost nothing over the fine-tuned network on the type criterion (0.670 → 0.671); its value is the α knob that trades type-accuracy for colour-awareness.
- **Preprocessing did not help** on these clean studio images (e.g. ResNet50 mAP 0.441 raw vs. 0.432 with flood-fill background removal vs. 0.308 with the original Otsu + CLAHE pipeline). This is scoped to the type criterion and clean images.
- **Robustness (mAP, fine-tuned):** 0.670 clean → 0.540 (blur σ=3), 0.609 (Gaussian noise σ=40), 0.562 (rotation 30°), 0.590 (brightness ×1.4). The handcrafted features degrade the most under noise.
- Most failures come from ambiguous/overlapping labels (e.g. similar `articleType`s), not from obviously wrong images.

ANN benchmark (gallery of ~22k × 256-d, 1000 queries):

| Method | Time (ms) | P@10 | Top-10 overlap with exact |
|---|---|---|---|
| NumPy exact | 588.5 | 0.781 | 1.000 |
| FAISS Flat | 71.4 | 0.781 | 0.999 |
| FAISS IVF nprobe=1 | 5.3 | 0.765 | 0.633 |
| **FAISS IVF nprobe=4** | 11.0 | 0.780 | 0.943 |
| FAISS IVF nprobe=16 | 29.3 | 0.780 | 0.998 |
| FAISS PQ (m=16) | 182.5 | 0.735 | 0.211 |

## Repository layout
```
├── notebook/            # training + evaluation notebook (run on Kaggle)
├── app.py               # Streamlit demo
├── retrieval.py         # feature extraction, background removal, search
├── download_assets.py   # downloads model + gallery files from the GitHub Release
├── requirements.txt
└── assets/              # (not in git) created by download_assets.py
```

## Run the demo
```bash
pip install -r requirements.txt
python download_assets.py          # fetches assets/ from the v1.0 Release
streamlit run app.py
```
To see product images in the results, download the dataset from Kaggle and put its `images/` folder next to `app.py`
(or set `DEMO_IMAGES=/path/to/images`). Without it the demo still works but shows text only.
The demo runs on CPU. Photos with a plain background work best (the gallery consists of white-background studio shots); non-white backgrounds are removed automatically with GrabCut, which can fail on cluttered scenes.

## Limitations
- Gallery images are clean studio photos; real-world photos with cluttered backgrounds or people wearing the item are out of distribution.
- "Relevant" means same product type, so visually near-identical but differently-labelled types count as errors.
- Single dataset, single train/test split (confidence intervals were computed by bootstrap over queries in the notebook, not over splits).

## Acknowledgements
Dataset: Param Aggarwal, *Fashion Product Images (Small)*, Kaggle. Models: torchvision ResNet50/VGG16 (ImageNet weights), FAISS.
