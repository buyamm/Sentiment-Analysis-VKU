# =============================================================================
# Extract Hotel Domain — VI, EN, FR, DE, ZH, ES, PT  (raw, no mapping)
# Dataset: Multilingual-NLP/M-ABSA
# Output: id, text, sentiment, aspect  — aspect = category gốc từ dataset
# =============================================================================

import ast
import os
import pandas as pd
import langid
from datasets import load_dataset

# ── 1. Import & load ──────────────────────────────────────────────────────────
DATASET_LANGS = ["ar", "da", "de", "en", "es", "fr", "hi", "hr",
                 "id", "ja", "ko", "nl", "pt", "ru", "sk", "sv",
                 "sw", "th", "tr", "vi", "zh"]
langid.set_languages(DATASET_LANGS)

print("Loading M-ABSA dataset...")
ds = load_dataset("Multilingual-NLP/M-ABSA")
print("Done!", {k: len(v) for k, v in ds.items()})

# ── 2. Gộp tất cả splits ──────────────────────────────────────────────────────
all_dfs = []
for split_name in ds.keys():
    df_tmp = ds[split_name].to_pandas()
    df_tmp["split"] = split_name
    all_dfs.append(df_tmp)

df_all = pd.concat(all_dfs, ignore_index=True)
print(f"Total rows: {len(df_all):,}")

# ── 3. Parse sentence & labels ────────────────────────────────────────────────
def parse_row(raw_text):
    raw_text = str(raw_text)
    if "####" in raw_text:
        sentence, label_str = raw_text.split("####", 1)
        try:
            labels = ast.literal_eval(label_str.strip())
        except Exception:
            labels = []
    else:
        sentence = raw_text
        labels = []
    return sentence.strip(), labels

df_all[["sentence", "labels"]] = df_all["text"].apply(
    lambda x: pd.Series(parse_row(x))
)
print(f"Parsed xong! Shape: {df_all.shape}")

# ── 4. Whitelist category thuộc domain Hotel ──────────────────────────────────
HOTEL_CATEGORIES_LOWER = {
    # hotel overall
    "hotel cleanliness",          "hotel comfort",           "hotel design_features",
    "hotel general",              "hotel miscellaneous",     "hotel prices",
    "hotel quality",
    # rooms
    "rooms cleanliness",          "rooms comfort",           "rooms design_features",
    "rooms general",              "rooms miscellaneous",     "rooms prices",
    "rooms quality",
    # room amenities
    "room_amenities cleanliness", "room_amenities comfort",  "room_amenities design_features",
    "room_amenities general",     "room_amenities prices",   "room_amenities quality",
    # service
    "service general",
    # location
    "location general",
    # food & drinks
    "food general",               "food prices",             "food quality",
    "food recommendation",        "food style_options",
    "drinks prices",              "drinks quality",          "drinks style_options",
    "food_drinks miscellaneous",  "food_drinks prices",      "food_drinks quality",
    "food_drinks style_options",
    # restaurant
    "restaurant general",         "restaurant miscellaneous","restaurant prices",
    # ambience
    "ambience general",
}

def is_hotel(labels):
    if not labels:
        return False
    for triplet in labels:
        if len(triplet) >= 2 and str(triplet[1]).lower() in HOTEL_CATEGORIES_LOWER:
            return True
    return False

df_all["is_hotel"] = df_all["labels"].apply(is_hotel)
df_hotel = df_all[df_all["is_hotel"]].copy()
print(f"Rows sau khi lọc domain hotel: {len(df_hotel):,}")
print(df_hotel["split"].value_counts().to_string())

# ── 5. Detect ngôn ngữ ────────────────────────────────────────────────────────
def has_script(text, start, end):
    return any(start <= ord(c) <= end for c in text)

def detect_by_script(text):
    t = str(text)
    if has_script(t, 0xAC00, 0xD7A3) or has_script(t, 0x1100, 0x11FF):
        return "ko"
    if has_script(t, 0x3040, 0x309F) or has_script(t, 0x30A0, 0x30FF):
        return "ja"
    if has_script(t, 0x4E00, 0x9FFF) or has_script(t, 0x3400, 0x4DBF):
        return "zh"
    if has_script(t, 0x0600, 0x06FF):
        return "ar"
    if has_script(t, 0x0E00, 0x0E7F):
        return "th"
    if has_script(t, 0x0900, 0x097F):
        return "hi"
    if has_script(t, 0x0400, 0x04FF):
        return "ru"
    return None

def detect_language(text):
    script = detect_by_script(text)
    if script is not None:
        return script
    lang, _ = langid.classify(str(text))
    return lang

print(f"Detecting language cho {len(df_hotel):,} rows...")
df_hotel = df_hotel.copy()
df_hotel["language"] = df_hotel["sentence"].apply(detect_language)
print("Done!")
print("\nPhân phối ngôn ngữ (tất cả):")
print(df_hotel["language"].value_counts().to_string())

# ── 6. Lọc ngôn ngữ: VI, EN, FR, DE, ZH, ES, PT ─────────────────────────────
TARGET_LANGS = ["vi", "en", "fr", "de", "zh", "es", "pt"]
df_filtered = df_hotel[df_hotel["language"].isin(TARGET_LANGS)].copy()
print(f"Rows sau khi lọc ngôn ngữ: {len(df_filtered):,}")

# ── 7. Normalize sentiment labels ─────────────────────────────────────────────
SENTIMENT_MAP = {
    "pos": "positive",
    "neg": "negative",
    "neu": "neutral",
}

def normalize_labels(labels):
    result = []
    for triplet in labels:
        if len(triplet) == 3:
            entity, category, sentiment = triplet
            sentiment_norm = SENTIMENT_MAP.get(str(sentiment).lower(), str(sentiment).lower())
            result.append([entity, category, sentiment_norm])
        else:
            result.append(triplet)
    return result

df_filtered["labels"] = df_filtered["labels"].apply(normalize_labels)

# ── 8. Transform: mỗi triplet → 1 row, câu nhiều aspect → chung id ───────────
# aspect = category gốc (lowercase, strip), không mapping
#
# Output columns: id | text | sentiment | aspect | language | split

rows = []
sentence_id = 0

for _, row in df_filtered.iterrows():
    sentence  = row["sentence"]
    language  = row["language"]
    split_val = row["split"]

    valid_triplets = []
    for triplet in row["labels"]:
        if len(triplet) != 3:
            continue
        entity, category, sentiment = triplet
        cat_lower = str(category).lower().strip()
        if cat_lower not in HOTEL_CATEGORIES_LOWER:
            continue  # bỏ triplet không thuộc hotel
        valid_triplets.append((cat_lower, sentiment))

    if not valid_triplets:
        continue

    for aspect, sentiment in valid_triplets:
        rows.append({
            "id":        sentence_id,
            "text":      sentence,
            "sentiment": sentiment,
            "aspect":    aspect,
            "language":  language,
            "split":     split_val,
        })

    sentence_id += 1

df_output = pd.DataFrame(rows)

# ── 9. Dedup: giữ 1 row trong mỗi nhóm (text, sentiment, aspect) trùng ───────
n_before = len(df_output)
df_output = df_output.drop_duplicates(subset=["text", "sentiment", "aspect"], keep="first")
n_after = len(df_output)
print(f"\nDedup: {n_before:,} → {n_after:,} rows (xóa {n_before - n_after:,} duplicates)")

# ── 10. Thống kê ──────────────────────────────────────────────────────────────
print(f"\nTotal rows : {len(df_output):,}")
print(f"Unique IDs : {df_output['id'].nunique():,}")

print("\nPhân phối aspect (category gốc):")
for asp, cnt in df_output["aspect"].value_counts().items():
    print(f"  {asp:<35s}: {cnt:,}")

print("\nPhân phối sentiment:")
print(df_output["sentiment"].value_counts().to_string())

print("\nPhân phối ngôn ngữ:")
print(df_output["language"].value_counts().to_string())

print("\nSample output:")
print(df_output[["id", "text", "sentiment", "aspect"]].head(10).to_string())

# ── 11. Lưu kết quả ───────────────────────────────────────────────────────────
output_dir = "output_hotel"
os.makedirs(output_dir, exist_ok=True)

save_cols    = ["id", "text", "sentiment", "aspect"]
save_cols_ex = ["id", "text", "sentiment", "aspect", "language", "split"]

# 1. Toàn bộ
path = os.path.join(output_dir, "hotel_all.csv")
df_output[save_cols_ex].to_csv(path, index=False, encoding="utf-8-sig")
print(f"\n[ALL]        {path}  ({len(df_output):,} rows)")

# 2. Theo ngôn ngữ
for lang in TARGET_LANGS:
    df_lang = df_output[df_output["language"] == lang]
    if len(df_lang) == 0:
        continue
    path = os.path.join(output_dir, f"hotel_{lang}.csv")
    df_lang[save_cols].to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[{lang.upper():5s}]       {path}  ({len(df_lang):,} rows)")

# 3. Theo split
for sp in ["train", "validation", "test"]:
    df_sp = df_output[df_output["split"] == sp]
    if len(df_sp) == 0:
        continue
    path = os.path.join(output_dir, f"hotel_{sp}.csv")
    df_sp[save_cols].to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[{sp:12s}] {path}  ({len(df_sp):,} rows)")

print("\nDone! Tất cả file đã lưu vào thư mục 'output_hotel/'")
