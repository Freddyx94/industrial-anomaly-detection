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
