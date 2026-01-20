import pandas as pd
import csv

# --- CONFIGURATION ---
# using forward slashes works on Windows too and is safer
SCORE_FILE = "wmt-mqm-human-evaluation-main/newstest2021/ende/mqm_newstest2021_ende.avg_seg_scores.tsv"
TEXT_FILE = "wmt-mqm-human-evaluation-main/newstest2021/ende/mqm_newstest2021_ende.tsv"
OUTPUT_FILE = "metricx_wmt21_mqm_processed_full.jsonl"
"""
def load_scores(filepath):
    print(f"Loading scores from {filepath}...")

    # sep=r'\s+' splits on ANY whitespace (tab or space)
    # engine='python' is explicitly set to avoid warnings with regex separators
    df = pd.read_csv(filepath, sep=r'\s+', engine='python', quoting=csv.QUOTE_NONE)

    #print(f"Detected columns: {list(df.columns)}")
    #print(df.head())

    # WMT21 MQM scores: 0=Perfect, Negative=Bad
    # MetricX Target: 0=Perfect, Positive=Bad
    # Flip the sign.
    df["score"] = -1 * df["mqm_avg_score"]
    #print(df)
    return df[["system", "seg_id", "score"]]

"""
import pandas as pd
import csv


def load_scores(filepath):
    print(f"Loading scores from {filepath}...")

    # Read with regex separator to handle mixed tabs/spaces
    df = pd.read_csv(filepath, sep=r'\s+', engine='python', quoting=csv.QUOTE_NONE)

    # 1. Force column to numeric.
    # This converts "None", "NA", or empty strings to NaN (Not a Number).
    df["mqm_avg_score"] = pd.to_numeric(df["mqm_avg_score"], errors='coerce')

    # 2. Check how many are missing before dropping
    initial_count = len(df)
    df = df.dropna(subset=["mqm_avg_score"])
    dropped_count = initial_count - len(df)

    if dropped_count > 0:
        print(f"⚠️ Dropped {dropped_count} rows with invalid or missing scores.")

    # 3. Flip the sign (now safe because we know they are all numbers)
    df["score"] = -1 * df["mqm_avg_score"]

    return df[["system", "seg_id", "score"]]

def load_text_data(filepath,langs):
    print(f"Loading text from {filepath}...")

    # 1. Find the header row dynamically
    skip_rows = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.startswith("system\t"):
                skip_rows = i
                break

    print(f"Detected header at row {skip_rows}. Reading...")

    # 2. Read dataframe
    df = pd.read_csv(filepath, sep="\t", skiprows=skip_rows, quoting=csv.QUOTE_NONE, on_bad_lines='warn')
    print(df.columns)
    # 3. Fix System Names
    df["system"] = df["system"].str.replace("hyp.", "", regex=False)
    df["system"] = df["system"].str.replace("ref.", "ref-",regex=False)

    # 4. Deduplicate
    text_df = df.drop_duplicates(subset=["system", "seg_id"])

    # 5. Select Columns
    cols = ["system", "seg_id", "source", "target"]
    if "reference" in df.columns:
        cols.append("reference")

    text_df = text_df[cols].copy()

    # Rename
    #rename_map = {"reference": "src", "target": "mt", "source": langs}
    rename_map = {"source": "src", "target": "mt", "reference": "ref"}
    text_df.rename(columns=rename_map, inplace=True)

    return text_df


def format_input_deterministic(row,lang):
    """
    NON-HYBRID MODE:
    Always provides Source + Candidate + Reference (if available).
    This trains a standard reference-based metric that also utilizes the source.
    """
    src = row["src"]
    mt = row["mt"]
    ref = row.get("ref", "")

    # Standard MetricX Input (Full Context)
    return f"candidate: {mt} reference: {src}  source: {lang}"


# --- MAIN EXECUTION ---
if __name__ == "__main__":

    #SCORE_FILE = "wmt-mqm-human-evaluation-main/newstest2021/ende/mqm_newstest2021_ende.avg_seg_scores.tsv"
    #TEXT_FILE = "wmt-mqm-human-evaluation-main/newstest2021/ende/mqm_newstest2021_ende.tsv"
    #OUTPUT_FILE = "wmt21_mqm_ende.jsonl"
    #lang = "en-de"
    SCORE_FILE = "wmt-mqm-human-evaluation-main/newstest2021/zhen/mqm_newstest2021_zhen.avg_seg_scores.tsv"
    TEXT_FILE = "wmt-mqm-human-evaluation-main/newstest2021/zhen/mqm_newstest2021_zhen.tsv"
    OUTPUT_FILE = "wmt21_mqm_zhen.jsonl"
    lang = "zh-en"


    # 1. Load
    df_scores = load_scores(SCORE_FILE)
    df_text = load_text_data(TEXT_FILE,langs=lang)


    print(df_text)

    # 2. Merge
    print("Merging text and scores...")
    merged_df = pd.merge(df_text, df_scores, on=["system", "seg_id"])
    print(f"Successfully matched {len(merged_df)} segments.")

    # 3. Format
    print("Formatting inputs (Deterministic/Full Mode)...")
    merged_df["input_text"] = merged_df.apply(
        format_input_deterministic,
        lang="en-de",
        axis=1
    )

    # 4. Save
    final_df = merged_df[["input_text", "score"]]
    print(f"Saving to {OUTPUT_FILE}...")
    final_df.to_json(OUTPUT_FILE, orient="records", lines=True)

    print("Done.")
    print(final_df.head(2))