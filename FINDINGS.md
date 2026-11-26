# Findings

Every number here comes from a file in `results/`. Dataset: MVTec AD (Bergmann et al.,
"A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", CVPR 2019),
licence CC BY-NC-SA 4.0.

## Stage 0: the data

Source: `notebooks/00_data.ipynb` (run on 2026-11-23). Result files: `results/stage0_counts.csv`,
`stage0_image_sizes.csv`, `stage0_defect_types.csv`, `stage0_defect_sizes.csv`,
`stage0_defect_size_summary.csv`.

### The dataset is complete

- **5,354 images** in 15 categories (10 objects, 5 textures), the same as MVTec's published
  total: 3,629 good training images, 467 good test images, 1,258 defective test images.
- Every defective image has a mask, and every mask has an image. The training folders contain only
  good images. Masks have the same size as their images and contain only the values 0 and 255.
- 73 defect types in total. Toothbrush has 1 type, cable has 8.

| category | kind | train (good) | test good | test defect | defect types | defect share of test | size (px) | colour | median defect area % | defects < 64 px | cut by a 224 crop (part) | (fully) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bottle | object | 209 | 20 | 63 | 3 | 0.76 | 900×900 | RGB | 5.72 | 0 | 4 | 0 |
| cable | object | 224 | 58 | 92 | 8 | 0.61 | 1024×1024 | RGB | 3.68 | 0 | 1 | 0 |
| capsule | object | 219 | 23 | 109 | 5 | 0.83 | 1000×1000 | RGB | 0.43 | 5 | 3 | 0 |
| hazelnut | object | 391 | 40 | 70 | 4 | 0.64 | 1024×1024 | RGB | 1.62 | 0 | 0 | 0 |
| metal_nut | object | 220 | 22 | 93 | 4 | 0.81 | 700×700 | RGB | 3.67 | 0 | 35 | 0 |
| pill | object | 267 | 26 | 141 | 7 | 0.84 | 800×800 | RGB | 1.03 | 7 | 5 | 0 |
| screw | object | 320 | 41 | 119 | 5 | 0.74 | 1024×1024 | L | 0.29 | 0 | 12 | 0 |
| toothbrush | object | 60 | 12 | 30 | 1 | 0.71 | 1024×1024 | RGB | 1.08 | 0 | 5 | 0 |
| transistor | object | 213 | 60 | 40 | 4 | 0.40 | 1024×1024 | RGB | 2.03 | 0 | 9 | 0 |
| zipper | object | 240 | 32 | 119 | 7 | 0.79 | 1024×1024 | L | 2.35 | 0 | 40 | 0 |
| carpet | texture | 280 | 28 | 89 | 5 | 0.76 | 1024×1024 | RGB | 1.48 | 0 | 11 | 0 |
| grid | texture | 264 | 21 | 57 | 5 | 0.73 | 1024×1024 | L | 0.60 | 0 | 22 | 1 |
| leather | texture | 245 | 32 | 92 | 5 | 0.74 | 1024×1024 | RGB | 0.46 | 1 | 8 | 0 |
| tile | texture | 230 | 33 | 84 | 5 | 0.72 | 840×840 | RGB | 7.93 | 0 | 51 | 0 |
| wood | texture | 247 | 19 | 60 | 5 | 0.76 | 1024×1024 | RGB | 2.90 | 0 | 26 | 0 |

"Colour" L means grayscale. "Defects < 64 px" counts defects that cover fewer than 64 pixels
(an 8 × 8 square) of the 256 × 256 model input. "Cut by a 224 crop" counts defects that a
224 × 224 centre crop of that input would cut partly or fully; we decided not to crop (below).

### What the data tells us

1. **Image format.** All images are square, 700 to 1024 px per side. Screw, zipper and grid are
   grayscale; they must be copied into 3 channels for pretrained networks.
2. **Defects are small.** The median defect covers 1.54% of the image. The smallest one
   (capsule/test/crack/021.png) covers 0.037%, about 24 pixels of the 256 × 256 model input. 13
   of 1,258 defects cover fewer than 64 pixels (pill `color` 7, capsule `crack` 4, capsule `poke`
   1, leather `color` 1). Screw has the smallest median (0.29%), tile the largest (7.93%).
3. **Some defects are logical, not local.** metal_nut `flip` (23 images, 47.8–48.7% of the image),
   transistor `misplaced` (10 images, 37.7–49.3%) and pill `pill_type` (9 images, 40.5–41.0%)
   have masks that cover the whole object. Each small part looks normal; only the object as a
   whole is wrong. We expect patch-based methods to have more trouble with these.
4. **A centre crop would remove defect pixels.** Keeping only the centre 224 × 224 of the
   256 × 256 image (as the PatchCore paper does) drops a 6.25% band on each edge. 232 of 1,258
   defects lie partly in that band, 7 would lose more than half their area, and 1
   (grid/test/glue/008.png) would be removed completely. Most affected: tile 51, zipper 40,
   metal_nut 35, wood 26, grid 22.
5. **Test sets are mostly defective** (defect share 0.40 for transistor to 0.84 for pill), the
   opposite of a real production line. Image AUROC does not depend on this ratio; thresholds
   (Stage 4) will be set on good training images only.

### Decisions (2026-11-23)

- **Starting categories: bottle, screw, transistor, carpet, grid** (3 objects, 2 textures).
  - bottle: easy object, the largest median defect among objects (5.72%).
  - screw: hard object, the smallest median defect of all categories (0.29%), grayscale.
  - transistor: `misplaced` defects cover the whole object (finding 3); the only test set where
    good images outnumber defective ones (defect share 0.40).
  - carpet: irregular colour texture (median defect 1.48%).
  - grid: regular grayscale texture; 22 defects lie partly in the border a crop would remove.
- **Preprocessing: resize to 256 × 256, no centre crop** (finding 4). Every labelled defect stays
  visible to the model and every mask pixel counts in the pixel metrics. The cost is about 30%
  more compute than 224 × 224 (256² / 224² ≈ 1.31), and our numbers cannot be compared
  one-to-one with papers that crop. Set in `config.yaml` (`image.crop: null`).

## Stage 1: neural network fundamentals (Fashion-MNIST)

Source: `notebooks/01_fundamentals.ipynb` (run on 2026-11-26). A learning stage on Fashion-MNIST
(Xiao, Rasul and Vollgraf, Zalando Research, 2017; MIT licence), not on MVTec AD. Result files:
`results/stage1_fashion_split.csv`, `stage1_fc_history.csv`, `stage1_fc_confusion.csv`,
`stage1_regularisation.csv`, `stage1_fc_full_data.csv`, `stage1_fc_vs_cnn.csv`,
`stage1_per_class_accuracy.csv`, `stage1_augmentation.csv`, `stage1_augmentation_robustness.csv`,
`stage1_augmentation_full_data.csv`, `stage1_test.csv`.

### Setup

- The 60,000 training images were split once, with a fixed seed, into 50,000 for training and
  10,000 for validation (957 to 1,061 per class). Every choice was made on the validation set.
  The 10,000 official test images (1,000 per class) were used once, at the end.
- Device: Apple GPU (`mps`). Every run is repeatable: rerunning the notebook gives the same
  accuracies and losses; only the training times change a little.

### Final results

| model | weights | validation accuracy | test accuracy | validation loss | test loss |
|---|---|---|---|---|---|
| fully connected (dropout 0.3, weight decay 0.1, early stopping) | 269,322 | 89.24% | 88.62% | 0.298 | 0.321 |
| small CNN (same settings) | 50,186 | 92.69% | 92.25% | 0.208 | 0.219 |

Both models score a little lower on the test set, as expected after choosing them on
validation. The CNN's lead holds: 3.6 points on test, 3.5 on validation.

### What we found

1. **Overfitting makes a model overconfident before it makes it less accurate.** Trained on
   1,000 images for 100 epochs, the fully connected network reached 100% training accuracy and
   81.2% validation accuracy. Its validation loss was lowest at epoch 13 (0.551) and doubled to
   1.079 by epoch 100.
2. **The fixes cured overconfidence, not accuracy.** Over 3 seeds, the validation loss was
   1.027–1.105 without regularisation, 0.547–0.551 with early stopping, and 0.525–0.530 with
   early stopping + dropout 0.5 + weight decay 1.0. Validation accuracy stayed between 80.2% and
   81.4% in all 9 runs. Early stopping did most of the work.
3. **Too much regularisation underfits, and the right amount depends on the data size.** Weight
   decay 10 on 1,000 images: 92.0% training and 78.8% validation accuracy. On 50,000 images,
   dropout 0.5 + weight decay 1.0 (the best setting on 1,000) gave 86.7%; the lighter dropout 0.3
   + weight decay 0.1 gave 89.24% and the lowest validation loss (0.298).
4. **The CNN beats the fully connected network with about 5 times fewer weights:** 92.69%
   against 89.24% validation accuracy. It is better in every class; the largest gains are shirt
   (67.7% → 77.8%), coat (85.9% → 91.5%) and pullover (81.1% → 86.3%). Shirt remains the hardest
   class. The price is compute: about 3.0 s per epoch against 0.7 s.
5. **Data augmentation buys robustness, not accuracy, on tidy data.** On 1,000 images (small
   CNN, 3 seeds), validation accuracy was 83.5–84.2% without augmentation, 82.9–83.5% with a
   left-right flip, 82.8–83.4% with flip + rotation up to 10°, and 69.3–71.3% with upside-down
   flips + rotation up to 180°. On all 50,000 images: 92.69% without and 91.51% with flip +
   rotation. But on mirrored validation images, the CNN without augmentation fell to 59.5%
   (1,000 images) and 69.6% (50,000 images), while the augmented ones kept 83.1% and 91.7%.
   Fashion-MNIST photos are all centred and upright, with almost every shoe pointing left, so
   that robustness was never needed. Filling the corners of rotated images with the wrong colour
   made no measurable difference to classification (82.8–83.4%).

### What this means for MVTec AD

- **Early stopping needs a validation set.** The autoencoder in Stage 2 will need one, made
  only from good training images, never from the test set. How to split it is part of the
  evaluation design, so we agree on it at the start of Stage 2.
- **A difference smaller than the spread between seeds is not a finding.** Compare methods over
  several seeds where training is random.
- **Augment only with variation the good training images already show.** A mirrored good
  metal nut has the shape of the `flip` defect, and a rotated good transistor looks like the
  `misplaced` defect: augmenting them would teach the model that these defects are normal.
  Screws already lie at every angle, so rotating them adds nothing new. Rotation by angles other
  than multiples of 90° uncovers corners, which no good image has.
- **Padding and empty corners create edges that are not in the image.** Both matter for
  defects near the border: 232 of the 1,258 MVTec defects lie at least partly in the outer 6.25%
  band of the image (Stage 0).
