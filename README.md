# ECG-Latent-Space-Analysis-for-Sports-Eligibility

This repository contains a notebook that:
- Takes 12-lead ECGs;
- Compresses them into a **latent space** using an **Autoencoder (AE)**;
- Adds a **classification head** on the latent space;
- Uses **classical ML models** on the latent (and aggregated features) to predict **sport eligibility** (target `sport_ability`).

> **Note:** Data is not included (standard practice for privacy/licensing reasons).

---

## ⚡ The Idea in 30 Seconds

1.  **ECG Preprocessing**
    - Mean removal per lead
    - Bandpass **0.5–40 Hz**
    - Notch **50 Hz**
    - Windowing with overlap

2.  **“Split-Lead” AE**
    - Split the 12 leads into **2 groups of 6**
    - 2 separate encoders → `z1` and `z2`
    - Concatenation → `z = [z1, z2]` (default `LATENT_DIM = 32`)
    - Separate decoders to reconstruct the two groups
    - Classification head on `z` for `sport_ability`

3.  **2-Phase Training**
    - **Phase 1 (pretrain):** Reconstruction only (early stopping on `val_rec`)
    - **Phase 2 (fine-tune):** Reconstruction + classification (`ALPHA * rec + BETA * cls`, early stopping on F1)

4.  **Patient Features + ML**
    - Extract `z` per window
    - Aggregate per patient via **mean + std** of each latent dimension
    - (Optional) Merge with tabular features
    - Removal of highly correlated features
    - Scaling + training of models like **LogReg / SVM / RandomForest / GradientBoosting / XGBoost**
    - Evaluation via ROC-AUC, ROC curve, etc.

---

## 📂 Repository Content

- `Code.ipynb` → Main notebook (complete pipeline)

**Typical outputs saved by the notebook:**
- `best_rec_only.pth` → Best weights from Phase 1 (reconstruction only)
- `best_ae_split_leads.pth` → Best weights after fine-tuning

---

## 💾 Expected Data

### ECG
- `.mat` files
- Each file contains a variable `val`
- Expected shape: **[5000, 12]** (Sampling rate `FS_SIGNAL = 500 Hz`, approx. 10s)

The notebook converts this to a tensor `[N, 12, 5000]` and creates windows:
- `WINDOW_SIZE = 1000` (~2s)
- `STRIDE = 250` (75% overlap)

### Excel (Metadata)
Two Excel files (batch 1 and batch 2), containing at least:
- `ECG_patient_id`
- `sport_ability` (target, binary in the notebook)

