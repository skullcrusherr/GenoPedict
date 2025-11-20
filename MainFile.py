# ====================== IMPORT PACKAGES ==============

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle

import matplotlib
matplotlib.use("Agg")  # use non-GUI backend so we can save figures
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
#  1. CONFIG
# =====================================================

DATA_PATH = "dna_sequence_dataset.csv"  # your prepared DNA dataset

DNA_COLUMNS = [
    "Seq_Window_1",
    "Seq_Window_2",
    "Promoter_Seq",
    "Exon_1_Seq",
    "Mutation_Site_Seq",
    "Enhancer_Seq",
    "Intron_1_Seq",
    "CpG_Island_Seq",
    "Repeat_Seq",
    "Motif_1",
    "Motif_2",
    "DNA_Segment_1",
    "DNA_Segment_2",
    "DNA_Segment_3",
    "DNA_Segment_4",
]

DINUCLEOTIDES = [a + b for a in "ACGT" for b in "ACGT"]


# =====================================================
#  2. FEATURE ENGINEERING (DNA → NUMERIC)
# =====================================================

def featurize_sequence(seq: str) -> dict:
    """
    Turn one DNA string into numeric features:
      - length
      - GC content
      - A/C/G/T proportions
      - all 16 dinucleotide frequencies
    Works for any A/C/G/T string (other characters are ignored).
    """
    if seq is None:
        seq = ""
    seq = seq.strip().upper()
    seq = "".join([c for c in seq if c in "ACGT"])

    length = len(seq)
    if length == 0:
        # avoid divide-by-zero; treat as empty but set length to 1 for proportions
        length = 1
        seq_clean = ""
    else:
        seq_clean = seq

    # Mono-nucleotide counts
    counts = {base: seq_clean.count(base) for base in "ACGT"}
    mono_props = {f"prop_{b}": counts[b] / length for b in "ACGT"}

    # GC content
    gc_content = (counts["G"] + counts["C"]) / length

    # Di-nucleotide counts & frequencies
    d_counts = {d: 0 for d in DINUCLEOTIDES}
    for i in range(len(seq_clean) - 1):
        pair = seq_clean[i:i+2]
        if pair in d_counts:
            d_counts[pair] += 1

    total_pairs = max(len(seq_clean) - 1, 1)
    d_freqs = {f"di_{d}": d_counts[d] / total_pairs for d in DINUCLEOTIDES}

    features = {
        "length": float(length),
        "gc_content": gc_content,
    }
    features.update(mono_props)
    features.update(d_freqs)
    return features


def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row, featurize all 15 DNA columns and return a numeric DataFrame.
    Column names look like:
      Seq_Window_1_length, Seq_Window_1_prop_A, ..., DNA_Segment_4_di_TT
    """
    rows = []
    for _, row in df.iterrows():
        feat_row = {}
        for col in DNA_COLUMNS:
            seq_val = row[col]
            seq_feats = featurize_sequence(seq_val)
            for k, v in seq_feats.items():
                feat_row[f"{col}_{k}"] = v
        rows.append(feat_row)

    return pd.DataFrame(rows)


# =====================================================
#  3. MAIN PIPELINE
# =====================================================

def main():
    # ---------- Load dataset ----------
    print("--------------------------------")
    print("Loading DNA sequence dataset")
    print("--------------------------------")

    df = pd.read_csv(DATA_PATH)

    print("First 10 rows:")
    print(df.head(10), "\n")

    print("Data Info:")
    print(df.info(), "\n")

    print("Summary statistics (including non-numeric):")
    print(df.describe(include="all"), "\n")

    print("Target Variable Distribution (Disease):")
    print(df["Disease"].value_counts(), "\n")

    # ---------- Plot & save class distribution ----------
    plt.figure(figsize=(8, 4))
    sns.countplot(x="Disease", data=df)
    plt.xticks(rotation=45, ha="right")
    plt.title("Disease Class Distribution")
    plt.tight_layout()
    plt.savefig("disease_class_distribution.png", dpi=300)
    plt.close()
    print("Saved class distribution plot as 'disease_class_distribution.png'")

    # ---------- Build numeric features from DNA ----------
    print("Building numeric features from 15 DNA columns...")
    X_features = build_feature_dataframe(df)
    print("Feature matrix shape:", X_features.shape)
    feature_columns = X_features.columns.tolist()
    print("Number of numeric features:", len(feature_columns), "\n")

    # ---------- Encode target ----------
    y_raw = df["Disease"].astype(str)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # ---------- Train / test split ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X_features,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("---------------------------------------------")
    print("             Data Splitting                  ")
    print("---------------------------------------------")
    print("Total samples :", df.shape[0])
    print("Train samples :", X_train.shape[0])
    print("Test samples  :", X_test.shape[0], "\n")

    # ---------- Scale features ----------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------- Train model ----------
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train_scaled, y_train)

    # ---------- Train performance ----------
    train_preds = model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_preds) * 100

    y_train_labels = label_encoder.inverse_transform(y_train)
    train_pred_labels = label_encoder.inverse_transform(train_preds)

    print("---------------------------------------------")
    print("   RandomForest Classifier - DNA Features    ")
    print("---------------------------------------------")
    print(f"Train Accuracy = {train_acc:.2f}%\n")
    print("Classification Report (Train):")
    print(classification_report(y_train_labels, train_pred_labels))

    # ---------- Test performance ----------
    test_preds = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_preds) * 100

    y_test_labels = label_encoder.inverse_transform(y_test)
    test_pred_labels = label_encoder.inverse_transform(test_preds)

    print(f"Test Accuracy = {test_acc:.2f}%\n")
    print("Classification Report (Test):")
    print(classification_report(y_test_labels, test_pred_labels))

    # ---------- Confusion Matrix (Test) ----------
    cm = confusion_matrix(
        y_test_labels,
        test_pred_labels,
        labels=label_encoder.classes_,
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Test Data)")
    plt.tight_layout()
    plt.savefig("confusion_matrix_test.png", dpi=300)
    plt.close()
    print("Saved confusion matrix plot as 'confusion_matrix_test.png'")

    # ---------- Accuracy Comparison Plot ----------
    plt.figure(figsize=(5, 4))
    plt.bar(["Train", "Test"], [train_acc, test_acc], color=["#3498db", "#e74c3c"])
    plt.ylim(0, 100)
    for i, v in enumerate([train_acc, test_acc]):
        plt.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)
    plt.ylabel("Accuracy (%)")
    plt.title("Train vs Test Accuracy")
    plt.tight_layout()
    plt.savefig("accuracy_comparison.png", dpi=300)
    plt.close()
    print("Saved accuracy comparison plot as 'accuracy_comparison.png'")

    # ---------- Save model artifacts ----------
    with open("dna_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("dna_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open("dna_label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    with open("dna_feature_columns.pkl", "wb") as f:
        pickle.dump(feature_columns, f)

    print("---------------------------------------------")
    print("Saved artifacts:")
    print("  dna_model.pkl")
    print("  dna_scaler.pkl")
    print("  dna_label_encoder.pkl")
    print("  dna_feature_columns.pkl")
    print("And plots:")
    print("  disease_class_distribution.png")
    print("  confusion_matrix_test.png")
    print("  accuracy_comparison.png")
    print("---------------------------------------------")


if __name__ == "__main__":
    main()
