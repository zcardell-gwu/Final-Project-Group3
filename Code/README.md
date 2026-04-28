## Execution Order

**Step 1: Prepare Data**
Download CelebA dataset from Kaggle:
https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/data

**Step 2: Run Classifier Training**
python "train script.py"

- First run automatically pre-caches all images (CLAHE + resize) — takes ~15 min, runs once only
- Trains DenseNet169 + BiLSTM for up to 20 epochs with early stopping
- Saves best model to Models/model_Group3.pt

### Output Files
- results_Group3.xlsx — predictions for all 19,962 test images
- training_log_Group3.csv — per-epoch train/val loss and F1
- per_attr_f1_Group3.csv / .txt — per-attribute F1 ranked report
- summary_Group3.txt — model architecture and hyperparameters
- model_Group3.pt — trained model weights (276MB, not uploaded to GitHub)


### Step 3: Train CVAE
```bash
python cVAE_train_revised.py
```

### Step 4: Evaluate Generated Images
```bash
python evaluate_generated.py
```

### Step 5: Run Demo
```bash
streamlit run streamlit_demo.py
```
