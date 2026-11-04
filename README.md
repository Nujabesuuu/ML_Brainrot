# Brainrot Detector: is this image Italian brainrot?

Mini-project for *Machine Learning*, NaUKMA. A binary image classifier (TensorFlow/Keras) with a Streamlit demo.

## 1. Goal

Predict whether a single image is an **Italian brainrot** character: the 2025 TikTok meme genre of AI-generated surreal
animal/object hybrids (Tralalero Tralala, Ballerina Cappuccina, ...). The label is 1 for brainrot and 0 for anything else.

The interesting question is not "can a CNN separate these classes?" but **"what does it actually learn?"**.
Every brainrot image is AI-generated. If the negatives are only real photos, a model can score high just by detecting
AI-generated images. So the negative class deliberately includes **ordinary AI-generated images** (hard negatives),
and every result is broken down by image source.

## 2. Data

| class | source tag | images | dataset | licence |
|---|---|---|---|---|
| 1 brainrot | `brainrot` | 5000 | [Largest Italian Brainrot Images Dataset](https://www.kaggle.com/datasets/levcheng/largest-italian-brainrot-images-dataset) (Kaggle, from the Italian Brainrot Wiki) | MIT |
| 0 not brainrot | `ai_normal` | 500 | [midjourney-dalle-sd-dataset](https://huggingface.co/datasets/julienlucas/midjourney-dalle-sd-dataset), test split, label *fake* (Midjourney / DALL-E / SD) | MIT |
| 0 not brainrot | `photo` | 500 | same Hugging Face dataset, label *real* | MIT |
| 0 not brainrot | `photo` | 4000 | [Imagenette 320px](https://github.com/fastai/imagenette): 400 images from each of 10 ImageNet classes | Apache 2.0 |

`src/prepare_data.py` downloads everything (~700 MB raw), then gives **every** image the same treatment: RGB, centre square crop,
256×256, JPEG quality 90. Image size, aspect ratio or compression therefore say nothing about the class.
18 duplicates (identical 64-bit difference hash) are removed **before** splitting. The final set has 9982 images,
split 70/15/15 into train/val/test, stratified by source.

## 3. Method

| step | choice | reason |
|---|---|---|
| baseline | majority class; colour histogram (HSV 8×8×8) + logistic regression | gives a floor, and measures how much "style/colour only" already explains |
| preprocessing | fixed per-image standardisation; `StandardScaler` inside a `Pipeline` | nothing learnt from data is fitted outside a CV fold, so nothing leaks |
| models | (a) frozen MobileNetV2 embeddings + logistic regression; (b) MobileNetV2 fine-tuned in Keras with augmentation | transfer learning works on ~7k images and a CPU; (a) is cheap enough to cross-validate |
| validation | 5-fold stratified CV on train+val for the scikit-learn models; train/val for Keras (early stopping, threshold); test used once | images are independent after deduplication, so stratified splits are valid |
| metrics | ROC-AUC, PR-AUC, F1, balanced accuracy, **error rate per source** | balanced classes, but class 0 mixes easy photos with hard AI negatives |
| error analysis | most confident mistakes, a border-removal test (watermarks), occlusion maps | checks what the model looks at |

Keras training has two stages. First only the classification head is trained (5 epochs, lr 1e-3). Then the top 30 MobileNetV2
layers are trained with lr 1e-5, with early stopping on validation loss. The decision threshold is the one that maximises
balanced accuracy on the validation set.

## 4. Results

**5-fold CV on train+val** (mean ± std):

| model | ROC-AUC | PR-AUC | F1 | balanced acc. |
|---|---|---|---|---|
| majority class | 0.500 ± 0.000 | 0.499 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 |
| colour histogram + logreg | 0.775 ± 0.011 | 0.732 ± 0.010 | 0.723 ± 0.009 | 0.722 ± 0.009 |
| MobileNetV2 embeddings + logreg | 0.978 ± 0.002 | 0.976 ± 0.002 | 0.931 ± 0.003 | 0.931 ± 0.003 |

**Held-out test set** (1498 images, used once):

| model | ROC-AUC | PR-AUC | F1 | bal. acc. | error on brainrot | error on photo | error on ai_normal |
|---|---|---|---|---|---|---|---|
| majority class | 0.500 | 0.499 | 0.000 | 0.500 | 100 % | 0 % | 0 % |
| colour histogram + logreg | 0.773 | 0.718 | 0.725 | 0.717 | 25.3 % | 28.9 % | 53.3 % |
| embeddings + logreg | 0.972 | 0.969 | 0.931 | 0.931 | 7.2 % | 3.0 % | 38.7 % |
| **fine-tuned MobileNetV2** | **0.986** | **0.982** | **0.951** | **0.951** | **4.7 %** | **1.5 %** | **38.7 %** |

The fine-tuned model uses threshold 0.35, the value that maximised balanced accuracy on the validation set.
Figures are in `reports/figures/`; all numbers are also in `reports/metrics.json`.

**Hard-negative curve** (embeddings model, 5-fold CV). The error on ordinary AI images is 57.6 % with none of them in
training, 47.3 % with 25 % of them, 37.2 % with 50 % and 31.8 % with all ~340.

**Border test.** Keeping only the central 60 % of each image changes the error on brainrot from 4.7 % to 9.1 % and on
photos from 1.5 % to 3.4 %.

## 5. Conclusions and limitations

**Supported.** The model reliably separates brainrot from everyday photos (ROC-AUC 0.986, 1.5 % false alarms on photos).
Each model clearly beats the previous one, and most of the gain comes from pretrained ImageNet features.
Brainrot also has a recognisable colour style: colour alone gives 0.77. Cropping away the borders hurts brainrot and
photos about equally, so there is no sign that the model relies on TikTok watermarks or captions.

**Not supported.** That the model understands "brainrot" as a concept. **29 of 75 ordinary AI-generated test images
(38.7 %) are flagged as brainrot.** These are mostly cartoon-like creatures with big eyes, so
the model has partly learnt "AI-generated character ⇒ brainrot". The hard-negative curve shows this error falls
steadily as more ordinary AI images are added to training and has not levelled off. Collecting more hard negatives is
the most promising next step.

**Limitations.**
- Only 75 hard negatives are in the test set, so their error rate is uncertain.
- The data covers one meme wave (2025 Italian brainrot, one wiki).
- "Brainrot" has no formal definition, and the labels come from the source datasets.
- The photos come from only 10 ImageNet classes.
- The Keras model was trained once, so its seed-to-seed variance was not measured.
- Occlusion maps are only qualitative.

**Bottom line:** usable as a "brainrot vs photos" filter, but not yet as a "brainrot vs any AI picture" filter.

## 6. Work done

1. **Question and data.** Defined the task and chose the data sources, including ordinary AI-generated images as
   hard negatives so the model cannot win by just detecting AI images.
2. **Data preparation** (`prepare_data.py`). Downloaded everything, gave every image the same standardisation
   (RGB, centre crop, 256×256, JPEG 90), removed duplicates and split 70/15/15 stratified by source.
3. **EDA.** Checked class and source balance and looked through samples of every source.
4. **Baselines.** Majority class and colour histogram + logistic regression, to measure how much colour/style alone explains.
5. **Embeddings model and CV.** Frozen MobileNetV2 embeddings + logistic regression, compared with the baselines via
   5-fold stratified CV using ROC-AUC, PR-AUC, F1 and balanced accuracy.
6. **Keras model.** MobileNetV2 fine-tuned in two stages (head first, then top 30 layers with frozen BatchNorm),
   with augmentation and early stopping; the decision threshold was chosen on the validation set.
7. **Test evaluation.** All models evaluated once on the held-out test set, with error rates per source.
8. **Error analysis.** Most confident mistakes, the border-removal test, occlusion maps and the hard-negative
   learning curve, to see what the model really learnt and where it fails.
9. **Streamlit app.** A demo that classifies uploaded images with the fine-tuned model.

## 7. Use of AI tools

*Draft. The authors must edit this section so it describes exactly what they did.*

- **Tools and what for.** Claude Code (Anthropic), used as a coding assistant in the terminal. It helped find a
  public source of ordinary AI-generated images, wrote the first versions of `src/`, the notebook and `app.py`,
  helped debug the code and drafted this README.
- **Influence on methodology.** The assistant suggested AI-generated images as hard negatives and per-source error
  rates as the main diagnostic. It also suggested the border-removal test, the hard-negative learning curve and
  freezing BatchNorm during fine-tuning. We kept these after checking the arguments and the outputs ourselves.
  Choices we made ourselves: model selection, cross-validation method, metrics
- **Verified independently.** We re-ran the notebook from scratch and looked through samples of every source by hand
  (the hard negatives really are ordinary AI images). We also checked that the duplicate removal makes the split
  leak-free, and that the test set is used once. All code was manually checked verified

## 8. Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Kaggle token: kaggle.com -> Settings -> API -> "Create New Token" -> ~/.kaggle/kaggle.json
python -m src.prepare_data                     # ~1 min after downloads, builds data/index.csv
jupyter nbconvert --to notebook --execute notebook.ipynb --inplace   # ~15 min on CPU, no GPU needed

streamlit run app.py                           # demo at http://localhost:8501
```

Seed 42 is fixed everywhere and TensorFlow runs with deterministic ops. Library versions are in `requirements.txt`
(tested on Python 3.13, macOS arm64, CPU only).

**Streamlit Community Cloud:** push the repo to GitHub (the 21 MB model in `models/` is committed), then on share.streamlit.io
choose *New app*, pick the repo, set main file `app.py`, and pick Python 3.12+ under *Advanced settings*.

## Repository

```
src/config.py         paths, seed, image size
src/prepare_data.py   download → standardise → deduplicate → split → data/index.csv
src/model.py          tf.data loading, MobileNetV2 model, embeddings (shared by notebook and app)
notebook.ipynb        the full study, steps 1–11 of the assignment
app.py                Streamlit demo
models/               trained model (.keras)
reports/              metrics.json and figures/
```
