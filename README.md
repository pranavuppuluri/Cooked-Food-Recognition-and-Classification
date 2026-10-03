# Indian Food Classifier — ResNet-50

Recognises **37 traditional Indian dishes** from images, using a ResNet-50
pretrained on ImageNet and fine-tuned end to end on `ind_food_37`.

| | |
| --- | --- |
| Architecture | ResNet-50, fine-tuned |
| Trainable parameters | 39,544,363 |
| Epochs | 80 |
| Batch size | 16 |
| Input | 224×224 RGB |
| Loss | Cross-entropy |
| Optimiser | SGD, lr 1e-3, momentum 0.9 |
| LR schedule | `ReduceLROnPlateau` (factor 0.2, patience 3) |
| **Best validation accuracy** | **0.940** (epoch 59) |
| **Mean per-class test accuracy** | **0.937** |

---

## Results

### Training

![Training curves](assets/training_curves.png)

Validation accuracy reaches **0.940 at epoch 59** and then flattens, while
training accuracy keeps climbing toward 0.97. That gap is the honest read on this
run: the last twenty epochs buy nothing on validation, and the model is starting
to memorise. The checkpoint worth keeping is epoch 59, not epoch 79.

### Per class

![Per-class accuracy](assets/per_class_accuracy.png)

Eight classes are perfect on the test split — Vada Pav, Thukpa, Medu Vada,
Khandvi, Khakhra, Kachori, Dosa, Dal Makhni — and **eight sit below 0.90**:

| Class | Accuracy | Likely reason |
| --- | --- | --- |
| Shrikhand | 0.75 | 15/20. A plain bowl of thick white-yellow dairy, visually near Kheer and Rasmalai |
| Dal Bati Churma | 0.78 | A composite plate; the model sees whichever element dominates the crop |
| Aloo Puri | 0.78 | Shares nearly every visual cue with Chhole Bhature |
| Rajma Chawal | 0.86 | Brown curry beside white rice — the generic Indian plate layout |
| Paneer | 0.87 | A label spanning many different dishes |
| Siddu | 0.88 | Only 16 test images, so 14/16 — a noisy estimate |
| Pani Puri | 0.88 | Confused with Kachori and Samosa — all round and fried |
| Bhindi Masala | 0.89 | Dark-green sabzi, close to other dry vegetable dishes |

The pattern is consistent: failures cluster where dishes share **colour and
plating**, not where they are rare. More data alone would not fix Shrikhand; the
fix is harder negatives against the dairy-dessert group.

Raw numbers behind both charts are in [`results/`](results/), and
[`results/make_charts.py`](results/make_charts.py) regenerates the figures.

---

## Dataset

`ind_food_37` — 8,395 images across 37 classes, in `ImageFolder` layout.

| Split | Images |
| --- | --- |
| train | 6,185 |
| val | 1,092 |
| test | 1,118 |

```
ind_food_37/
├── train/<class_name>/*.jpg
├── val/<class_name>/*.jpg
└── test/<class_name>/*.jpg
```

<details>
<summary>All 37 classes</summary>

Aloo Puri · Bhakarwadi · Bhindi Masala · Biryani · Chhole Bhature ·
Dal Bati Churma · Dal Makhni · Dhokla · Dosa · Dum Aloo · Ghevar · Gulab Jamun ·
Idli Sambhar · Jalebi · Kachori · Khakhra · Khandvi · Kheer · Medu Vada ·
Modak · Mushroom · Nan Khatai · Paneer · Pani Puri · Pav Bhaji · Poha ·
Rajma Chawal · Rasgulla · Rasmalai · Samosa · Sarson ka Saag Makki ki Roti ·
Shrikhand · Siddu · Thepla · Thukpa · Uttapam · Vada Pav

</details>

Images are resized to 224×224 and normalised with ImageNet statistics. No
augmentation beyond that — which is one of the clearer things to try next.

---

## Setup

```bash
git clone https://github.com/pranavuppuluri/Cooked-Food-Recognition-and-Classification.git
cd Cooked-Food-Recognition-and-Classification

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install `torch` for your own CUDA version from [pytorch.org](https://pytorch.org)
if you want GPU training. On CPU, 80 epochs is not practical.

Then open the notebook:

```bash
jupyter notebook notebooks/resnet_food.ipynb
```

It reads `ind_food_37/train` and `ind_food_37/val` directly, so it runs against
the dataset in this repository with no path changes.

---

## Repository layout

```
├── ind_food_37/           # the dataset, ImageFolder layout
├── notebooks/
│   └── resnet_food.ipynb  # training, evaluation and video inference
├── assets/                # figures used in this README
├── results/
│   ├── training_history.json     # 80 epochs of train/val loss and accuracy
│   ├── per_class_accuracy.json   # final accuracy for each of the 37 classes
│   └── make_charts.py            # regenerates the figures from those files
├── requirements.txt
└── LICENSE
```

**What is not here:** trained weights (too large for git) and a packaged `src/`
module. The notebook is the implementation — it is exploratory in places, and it
is the honest artefact of how the model was actually built.

---

## Notes

The notebook also contains a video-inference path that reads a video file,
classifies frames, and writes an HTML summary. It depends on OpenCV and is less
polished than the training code.

If you re-run training, the result worth keeping is the **epoch-59 checkpoint**,
not the final one — see the training curves above.

---

## License

MIT — see [LICENSE](LICENSE).
