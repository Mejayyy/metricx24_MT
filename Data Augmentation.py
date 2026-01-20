import json
import random
import re
import string
import math


def generate_gibberish(length, vocab):
    """Generates a random sequence of words from the vocab."""
    if not vocab:
        return "gibberish error"
    return " ".join(random.choices(vocab, k=length))


def make_undertranslation(text):
    """
    Removes a sentence or 20-80% of words from the end.
    Returns (new_text, score).
    Score: 5 (minor cut) to 25 (major cut).
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)

    if len(sentences) > 1:
        # Remove an arbitrary sentence -> Bad score
        sentences.pop(random.randint(0, len(sentences) - 1))
        return " ".join(sentences), 25.0
    else:
        words = text.split()
        if len(words) <= 1:
            return text, 25.0

        percent_to_remove = random.uniform(0.2, 0.8)
        num_to_keep = int(len(words) * (1 - percent_to_remove))
        num_to_keep = max(1, num_to_keep)

        new_text = " ".join(words[:num_to_keep])

        # Map 20% cut -> 5.0 score, 80% cut -> 25.0 score
        score = 5.0 + (percent_to_remove - 0.2) * 33.33
        score = min(25.0, max(5.0, score))
        return new_text, score


def augment_data_smart_ratio(input_jsonl, output_jsonl):
    print(f"Reading {input_jsonl}...")
    data = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    total_original = len(data)
    target_count = int(total_original * 0.01)  # 1% of total
    print(f"Total Original Rows: {total_original}")
    print(f"Target per synthetic category (1%): {target_count} rows")

    # --- 1. Parse Data & Build Resources ---
    print("Building resources...")
    all_refs = []
    vocab = []
    perfect_indices = []  # Track rows with score 0.0

    # Regex to extract parts: "candidate: ... reference: ... source: ..."
    pattern = re.compile(r"candidate:\s*(.*?)\s*reference:\s*(.*?)\s*source:\s*(.*)")

    parsed_data = []  # Store parsed versions to avoid re-regexing

    for i, row in enumerate(data):
        match = pattern.search(row['input_text'])
        if match:
            cand, ref, src_lp = match.groups()
            parsed_data.append({
                "cand": cand,
                "ref": ref,
                "src_lp": src_lp,
                "orig_score": row['score']
            })
            all_refs.append(ref)
            vocab.extend(ref.split())

            # Check for perfect score (handling float tolerance)
            if abs(row['score']) < 1e-6:
                perfect_indices.append(i)
        else:
            parsed_data.append(None)  # Keep index alignment

    vocab = list(set(vocab))

    # --- 2. Synthetic Generation Functions ---
    synthetic_rows = []

    # Scores (Positive = Bad)
    SCORE_BAD = 25.0
    SCORE_PERFECT = 0.0
    SCORE_MINOR = 1.0

    # Helper to create input string
    def make_input(cand, ref, src):
        return f"candidate: {cand} reference: {ref} source: {src}"

    # A. Reference-Matching (Strict Logic: Only from Perfect Examples)
    print("Generating: Reference-Matching (from perfect examples only)...")
    if perfect_indices:
        # Sample with replacement if we don't have enough perfect rows
        selected_indices = random.choices(perfect_indices, k=target_count)
        for idx in selected_indices:
            item = parsed_data[idx]
            # Candidate becomes the Reference
            row = {
                "input_text": make_input(item['ref'], item['ref'], item['src_lp']),
                "score": SCORE_PERFECT
            }
            synthetic_rows.append(row)
    else:
        print("Warning: No perfect scores (0.0) found in data. Skipping 'Reference-Matching' category.")

    # B. Generate other categories (Random sampling from ALL data)
    # We sample indices for each category independently

    valid_indices = [i for i, x in enumerate(parsed_data) if x is not None]

    def sample_rows(count):
        return [parsed_data[i] for i in random.sample(valid_indices, min(count, len(valid_indices)))]

    # 1. Empty
    for item in sample_rows(target_count):
        synthetic_rows.append({
            "input_text": make_input("", item['ref'], item['src_lp']),
            "score": SCORE_BAD
        })

    # 2. Gibberish
    for item in sample_rows(target_count):
        gibberish = generate_gibberish(len(item['ref'].split()), vocab)
        synthetic_rows.append({
            "input_text": make_input(gibberish, item['ref'], item['src_lp']),
            "score": SCORE_BAD
        })

    # 3. Fluent Unrelated
    for item in sample_rows(target_count):
        random_ref = random.choice(all_refs)
        synthetic_rows.append({
            "input_text": make_input(random_ref, item['ref'], item['src_lp']),
            "score": SCORE_BAD
        })

    # 4. Undertranslation
    for item in sample_rows(target_count):
        new_cand, new_score = make_undertranslation(item['cand'])
        synthetic_rows.append({
            "input_text": make_input(new_cand, item['ref'], item['src_lp']),
            "score": new_score
        })

    # 5. Duplication
    for item in sample_rows(target_count):
        dup = f"{item['cand']} {item['cand']}"
        synthetic_rows.append({
            "input_text": make_input(dup, item['ref'], item['src_lp']),
            "score": SCORE_BAD
        })

    # 6. Missing Punctuation
    # We filter for rows that actually HAVE punctuation to remove
    punc_indices = [i for i in valid_indices if
                    parsed_data[i]['ref'] and parsed_data[i]['ref'][-1] in string.punctuation]
    if punc_indices:
        selected_punc = random.choices(punc_indices, k=target_count)
        for idx in selected_punc:
            item = parsed_data[idx]
            no_punc = item['ref'][:-1]
            synthetic_rows.append({
                "input_text": make_input(no_punc, item['ref'], item['src_lp']),
                "score": SCORE_MINOR
            })

    # --- 3. Save ---
    print(f"Generated {len(synthetic_rows)} synthetic rows.")
    final_data = data + synthetic_rows

    # Shuffle to mix synthetic data in
    random.shuffle(final_data)

    print(f"Saving to {output_jsonl}...")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for entry in final_data:
            f.write(json.dumps(entry) + "\n")
    print("Done.")


# Run it
augment_data_smart_ratio("WMT_20_21_22.jsonl", "WMT_20_21_22_augmented.jsonl")
