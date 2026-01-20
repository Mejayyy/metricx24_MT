import pandas as pd


def convert_csv_to_jsonl(csv_path, output_path):
    print(f"Reading {csv_path}...")
    # Load the CSV
    df = pd.read_csv(csv_path)

    # 1. Format the 'input_text' strictly according to your template:
    # "candidate: {mt} reference: {src} source:{lp}"
    print("Formatting input strings...")
    df["input_text"] = df.apply(
        lambda row: f"candidate: {row['mt']} reference: {row['src']} source:{row['lp']}",
        axis=1
    )

    # 2. Invert the score
    # Your template said: "score": -{score}
    # If the CSV has -2.5, the JSONL will have 2.5
    print("Formatting scores...")
    df["score"] = -1 * df["score"]

    # 3. Select only the columns we need for training
    final_df = df[["input_text", "score"]]

    # 4. Save to JSONL
    print(f"Saving to {output_path}...")
    final_df.to_json(output_path, orient='records', lines=True)
    print("Done.")

    # Preview
    print("\nPreview of the first line:")
    print(final_df.iloc[0].to_dict())

if __name__ == "__main__":
    # Run the conversion
    #convert_csv_to_jsonl("WMT_20_21_22.csv", "WMT_20_21_22.jsonl")
