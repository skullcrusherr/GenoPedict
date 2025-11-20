import requests
import pandas as pd
import random
from collections import Counter

BASE_URL = "http://127.0.0.1:5000"

USERNAME = "csvtestuser"
PASSWORD = "csvtest123"

# These must match DNA_COLUMNS in app.py
DNA_COLUMNS = [
    'Seq_Window_1', 'Seq_Window_2', 'Promoter_Seq', 'Exon_1_Seq',
    'Mutation_Site_Seq', 'Enhancer_Seq', 'Intron_1_Seq', 'CpG_Island_Seq',
    'Repeat_Seq', 'Motif_1', 'Motif_2', 'DNA_Segment_1',
    'DNA_Segment_2', 'DNA_Segment_3', 'DNA_Segment_4'
]

def extract_prediction(html: str) -> str | None:
    marker = "Predicted Disease:"
    idx = html.find(marker)
    if idx == -1:
        return None

    strong_start = html.find("<strong>", idx)
    strong_end = html.find("</strong>", strong_start)
    if strong_start == -1 or strong_end == -1:
        return None

    strong_start += len("<strong>")
    return html[strong_start:strong_end].strip()

def main():
    # Load your original dataset
    df = pd.read_csv("dna_sequence_dataset.csv")
    print("[*] Loaded dataset with", len(df), "rows")

    # Choose up to 100 random rows
    N = 300
    indices = random.sample(range(len(df)), min(N, len(df)))

    s = requests.Session()

    # 1) Register test user (ignore already exists)
    print("[*] Registering csvtest user...")
    reg_data = {"username": USERNAME, "password": PASSWORD}
    r = s.post(f"{BASE_URL}/register", data=reg_data)
    print("  Register status:", r.status_code)

    # 2) Login
    print("[*] Logging in...")
    login_data = {"username": USERNAME, "password": PASSWORD}
    r = s.post(f"{BASE_URL}/login", data=login_data, allow_redirects=True)
    if "/login" in r.url:
        print("  [!] Login may have failed, still on /login.")
        return
    print("  Logged in, redirected to:", r.url)

    disease_counts = Counter()
    failures = 0

    print(f"[*] Sending {len(indices)} real samples from CSV to the app...")

    for i, idx in enumerate(indices, start=1):
        row = df.loc[idx]
        # Build the form dict from real DNA sequences
        sample = {col: str(row[col]) for col in DNA_COLUMNS}

        r = s.post(f"{BASE_URL}/", data=sample)
        if r.status_code != 200:
            print(f"  [{i}] HTTP {r.status_code} – request failed")
            failures += 1
            continue

        pred = extract_prediction(r.text)
        if pred is None:
            print(f"  [{i}] Could not find prediction in HTML")
            failures += 1
        else:
            disease_counts[pred] += 1
            print(f"  [{i}] Predicted: {pred}")

    print("\n=== SUMMARY (REAL CSV SAMPLES) ===")
    print("Total requests:", len(indices))
    print("Failures:", failures)
    print("\nPredictions count per disease:")
    for disease, count in disease_counts.most_common():
        print(f"  {disease:15s}: {count}")

if __name__ == "__main__":
    main()
