import requests

BASE_URL = "http://127.0.0.1:5000"

USERNAME = "testuser"
PASSWORD = "test123"

# 15 DNA features - you can change these to anything
DNA_INPUT = {
    "Seq_Window_1":  "ACGTGACCTA",
    "Seq_Window_2":  "TTGACCGTAA",
    "Promoter_Seq":  "GCTTAGGCTAACGTTACGGA",
    "Exon_1_Seq":    "ATGGCTAACCTGAATCGTGA",
    "Mutation_Site_Seq": "CGTACGATCGTAACGTCGAT",
    "Enhancer_Seq":  "TTAACCGGTACGATCCGTAA",
    "Intron_1_Seq":  "GGAATTCGACCTTAGGACTA",
    "CpG_Island_Seq": "CGCGATTCGCGATTCGGCTA",
    "Repeat_Seq":    "ATATATATATATATATATAT",
    "Motif_1":       "GGCGTACGATCCGATGCAAA",
    "Motif_2":       "TTCGATGGTACGCGATACCA",
    "DNA_Segment_1": "CGATGCTAACGGTACCTGAA",
    "DNA_Segment_2": "AAGTCGGATACCGTACGCTA",
    "DNA_Segment_3": "TTGACGATGCCGTTAACGAT",
    "DNA_Segment_4": "GGTACCGATTCGATGACCAT",
}

def main():
    s = requests.Session()

    # 1) Try to register (ignore error if user already exists)
    print("[*] Registering user...")
    reg_data = {"username": USERNAME, "password": PASSWORD}
    r = s.post(f"{BASE_URL}/register", data=reg_data)
    print(f"  Register status: {r.status_code} (OK if 200 or redirect)")

    # 2) Login
    print("[*] Logging in...")
    login_data = {"username": USERNAME, "password": PASSWORD}
    r = s.post(f"{BASE_URL}/login", data=login_data)
    if r.url.endswith("/login"):
        print("  [!] Login may have failed (still on /login). Check username/password.")
    else:
        print("  Login OK, redirected to:", r.url)

    # 3) Send DNA input to home ('/')
    print("[*] Sending DNA sequences for prediction...")
    r = s.post(f"{BASE_URL}/", data=DNA_INPUT)

    print(f"  Response status: {r.status_code}")
    # Just print a small snippet of HTML to confirm
    text = r.text

    # Try to extract the line containing "Predicted Disease"
    marker = "Predicted Disease:"
    if marker in text:
        start = text.index(marker)
        snippet = text[start:start+200]
        print("\n=== Prediction Snippet from HTML ===")
        print(snippet)
        print("====================================")
    else:
        print("\n[!] Could not find 'Predicted Disease' in response HTML.")
        print("Here is the first 500 characters of the response:\n")
        print(text[:500])


if __name__ == "__main__":
    main()
