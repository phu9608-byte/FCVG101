# FCVG101 – Task 2: Data Pipeline & Deep-Learning Model (FruitSort)

3-class fruit classification (**apple, banana, orange**) on the Kaggle dataset
*Fruit Images for Object Detection* (Muhammed Buyukkinaci, licence CC0). The whole workflow is one notebook,
`Task2_pipeline.ipynb`; `src/` holds the three modules that the Task 3 app imports.

## Structure

```
task2/
├── Task2_pipeline.ipynb   # §1 ingestion → §2 cleaning + grouped split → §3 near-duplicate audit (pHash)
│                          # → §4 preprocessing/augmentation → §5 FruitCNN → §6 training → §7 evaluation → §8 inference
├── src/
│   ├── model.py           # FruitCNN: 3 × (Conv3×3 + ReLU [+ MaxPool]) → global average pooling → Linear (23,779 parameters)
│   ├── data_utils.py      # make_transform(): one preprocessing for train / val / test / inference (+ train-only augmentation)
│   └── inference.py       # load_model(), predict_pil()  – used by Task 3
├── artifacts/             # model.pth (trained model), classes.json, import_log.csv, data_manifest.csv,
│                          # rejected_images.csv, near_duplicates.csv
├── results/               # train_history.csv, learning_curves.png, test_metrics.json, test_predictions.csv,
│                          # confusion_matrix.png, near_duplicates.png
└── runs/                  # earlier 30-epoch baseline runs (GPU and CPU) + training log of the final model
```

The images are in `../datasets/fruits/`: `raw/<class>/` (255 images after cleaning) and
`splits/{train,val,test}/<class>/` (179 / 38 / 38).

## Run

Open `Task2_pipeline.ipynb` with the working directory set to `task2/` and run all cells.

- `KAGGLE_ARCHIVE = None` (default): the imported images in `../datasets/fruits/raw/` are reused.
  Set it to the extracted Kaggle archive to rebuild `raw/` from scratch (empty `raw/` first).
- `RUN_TRAINING = False` (default): the submitted `artifacts/model.pth` is evaluated, which reproduces the report
  (test accuracy 37/38 = 97.4 %, macro-F1 0.973). `True` retrains (Adam 1e-3, batch 32, up to 200 epochs,
  early stopping patience 20) and overwrites the model.

The split is deterministic (sorted groups, seed 42), so re-running §2 recreates exactly the same files.

## Notes

- Augmentation runs only in the training loader; validation/test use the plain transform.
- The test set is evaluated once, after the checkpoint has been chosen by validation loss.
- The near-duplicate list is a review aid: pairs at Hamming distance ≤ 14 are checked by eye; nothing is deleted.
- The softmax output is a model score, not a calibrated probability.
