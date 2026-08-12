# Attribute-Controlled Face Generation and Multi-Label Classification with CelebA

A PyTorch deep-learning pipeline for classifying 40 facial attributes and conditionally generating 128×128 face images from the CelebA dataset. The project combines a DenseNet169–BiLSTM multi-label classifier, a conditional variational autoencoder (cVAE), model evaluation pipelines, and an interactive Streamlit interface.

## Project Highlights

* Built a multi-label classification pipeline covering all 40 CelebA facial attributes
* Achieved approximately **0.75 validation macro-F1** with the DenseNet169–BiLSTM classifier
* Applied class-weighted loss and per-attribute threshold tuning to address label imbalance
* Evaluated conditional face generation on **10,000 generated images**
* Compared generated and real images using per-attribute F1, FID, and CLIP-based evaluation
* Developed an interactive Streamlit interface for face generation and attribute classification

## Key Results

| Component                     | Evaluation                 | Result |
| ----------------------------- | -------------------------- | -----: |
| DenseNet169–BiLSTM classifier | Validation macro-F1        | ≈ 0.75 |
| Conditional VAE               | Generated-image macro-F1   | 0.4300 |
| Conditional VAE               | FID                        | 104.41 |
| CLIP evaluation               | Macro-F1                   | 0.3420 |
| Evaluation dataset            | Generated images evaluated | 10,000 |

The conditional model performed best on broad facial attributes such as `Male`, `Young`, `Smiling`, `No_Beard`, and `Mouth_Slightly_Open`. Performance was weaker on subtle or localized attributes such as earrings, necklaces, rosy cheeks, pale skin, and double chin.

## Project Workflow

1. Download and organize the CelebA image and attribute data.
2. Preprocess images using resizing, CLAHE enhancement, normalization, and cached tensors.
3. Train a DenseNet169–BiLSTM classifier for 40-label facial-attribute prediction.
4. Tune decision thresholds separately for each attribute.
5. Train conditional VAE variants for attribute-controlled face generation.
6. Evaluate generated images using classifier-based F1, FID, and CLIP.
7. Present the trained models through an interactive Streamlit interface.

## Contributors and Responsibilities

| Contributor         | Main contributions                                                                                                                                                                                                                                |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Irene Xia**       | CelebA preprocessing and image caching; development and training of the DenseNet169–BiLSTM multi-label classifier; class-imbalance handling; per-attribute threshold tuning and evaluation; results documentation, final report, and presentation |
| **Zachary Cardell** | Conditional VAE architecture and training; LPIPS experiments; generation pipeline; FID and CLIP evaluation                                                                                                                                        |
| **Jiatong Peng**    | Streamlit interface development; model-selection functionality; interactive face-generation and classification workflow                                                                                                                           |

## Irene Xia’s Contribution

I developed the project’s multi-label facial-attribute classification pipeline. My work included preparing and caching the CelebA images, implementing the DenseNet169–BiLSTM architecture, addressing attribute imbalance with positive-class weighting, tuning attribute-specific classification thresholds, and producing per-attribute evaluation results.

The classifier generated predictions for **19,962 test images** and achieved approximately **0.75 validation macro-F1** across 40 facial attributes. I also contributed to the project documentation, final report, and group presentation.

* [Irene’s individual code and report](./irene-xia-individual-project)
* [Classifier training code](./Code/classifier_train.py)
* [Per-attribute F1 results](./Code/per_attr_f1_Group3.csv)
* [Training log](./Code/training_log_Group3.csv)

## Repository Contents

| Directory                                                                    | Description                                                           |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| [`Code`](./Code)                                                             | Classifier, cVAE, evaluation, data-download, and Streamlit scripts    |
| [`Experiments`](./Experiments)                                               | Alternative cVAE architectures, training logs, and evaluation results |
| [`irene-xia-individual-project`](./irene-xia-individual-project)             | Irene Xia’s classifier code and individual project report             |
| [`jiatong-peng-individual-project`](./jiatong-peng-individual-project)       | Jiatong Peng’s Streamlit code and individual project report           |
| [`zachary-cardell-individual-project`](./zachary-cardell-individual-project) | Zachary Cardell’s cVAE code and individual project report             |
| [`Final-Group-Project-Report`](./Final-Group-Project-Report)                 | Final written report                                                  |
| [`Final-Group-Presentation`](./Final-Group-Presentation)                     | Final project presentation                                            |
| [`Group-Proposal`](./Group-Proposal)                                         | Original group proposal                                               |

## Technologies

* Python
* PyTorch and Torchvision
* DenseNet169
* Bidirectional LSTM
* Conditional Variational Autoencoder
* OpenCV
* pandas and NumPy
* scikit-learn
* LPIPS
* CLIP
* Streamlit

## Data and Model Files

This project uses the [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Raw images, cached tensors, and trained model weights are not stored in this repository because of their size.

Expected local directories:

```text
Data/
├── img_align_celeba/
├── list_attr_celeba.csv
└── list_eval_partition.csv

Models/
├── model_Group3.pt
└── model_cVAE_Group3_final.pt
```

## Project Resources

* [Final Group Report](./Final-Group-Project-Report/Group3_Final_Report.pdf)
* [Final Group Presentation](./Final-Group-Presentation/Group3_Presentation.pptx)
* [Source Code](./Code)
* [Experiment Logs](./Experiments)

## Limitations and Responsible Use

The cVAE captured broad structural attributes more successfully than subtle accessories and localized facial features. Generated-image evaluation was also limited by differences between the cVAE output resolution and the classifier input resolution.

CelebA attributes are simplified dataset annotations and may contain social or demographic biases. This project was developed as an academic demonstration and should not be used for identity verification, clinical decisions, employment screening, or other high-stakes applications.
