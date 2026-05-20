# =============================================================================
# Extract Hotel Domain — VI, EN, FR, DE, ZH, ES, PT  (v2)
# Dataset: Multilingual-NLP/M-ABSA
# Mapping: 11 aspect labels (tiếng Việt)
# Output: id, text, sentiment, aspect  (1 câu nhiều aspect → chung id)
# =============================================================================

# ── 1. Import ─────────────────────────────────────────────────────────────────
import ast
import os
import pandas as pd
import langid
from datasets import load_dataset

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
    """Tách sentence và labels từ: sentence####[[entity, category, sentiment]]"""
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

# ── 4. Lọc domain Hotel theo aspect category ──────────────────────────────────
HOTEL_CATEGORIES_LOWER = {
    # hotel overall
    "hotel cleanliness",        "hotel comfort",         "hotel design_features",
    "hotel general",            "hotel miscellaneous",   "hotel prices",
    "hotel quality",
    # rooms
    "rooms cleanliness",        "rooms comfort",         "rooms design_features",
    "rooms general",            "rooms miscellaneous",   "rooms prices",
    "rooms quality",
    # room amenities
    "room_amenities cleanliness", "room_amenities comfort", "room_amenities design_features",
    "room_amenities general",     "room_amenities prices",  "room_amenities quality",
    # service
    "service general",
    # location
    "location general",
    # food & drinks
    "food general",             "food prices",           "food quality",
    "food recommendation",      "food style_options",
    "drinks prices",            "drinks quality",        "drinks style_options",
    "food_drinks miscellaneous","food_drinks prices",    "food_drinks quality",
    "food_drinks style_options",
    # restaurant
    "restaurant general",       "restaurant miscellaneous", "restaurant prices",
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
        return "ko"   # Hangul
    if has_script(t, 0x3040, 0x309F) or has_script(t, 0x30A0, 0x30FF):
        return "ja"   # Hiragana / Katakana
    if has_script(t, 0x4E00, 0x9FFF) or has_script(t, 0x3400, 0x4DBF):
        return "zh"   # CJK Unified Ideographs
    if has_script(t, 0x0600, 0x06FF):
        return "ar"   # Arabic
    if has_script(t, 0x0E00, 0x0E7F):
        return "th"   # Thai
    if has_script(t, 0x0900, 0x097F):
        return "hi"   # Devanagari
    if has_script(t, 0x0400, 0x04FF):
        return "ru"   # Cyrillic
    return None       # Latin → dùng langid

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

# Kiểm tra các giá trị sentiment sau normalize
all_sentiments = set()
df_filtered["labels"].apply(
    lambda lbls: [all_sentiments.add(t[2]) for t in lbls if len(t) == 3]
)
print("Unique sentiment values sau normalize:", sorted(all_sentiments))

# ── 8. Mapping: 11 aspect labels ─────────────────────────────────────────────
#
#  1. chat_luong_phong      (room_quality)      — phòng ốc, tiện nghi
#  2. ve_sinh               (cleanliness)       — vệ sinh
#  3. dich_vu               (service)           — dịch vụ nhân viên
#  4. vi_tri                (location)          — vị trí, địa điểm
#  5. am_thuc               (food_drinks)       — ẩm thực, đồ ăn uống
#  6. nha_hang              (restaurant)        — nhà hàng trong khách sạn
#  7. khong_gian            (ambience)          — không gian, bầu không khí
#  8. gia_ca                (price_value)       — giá cả, value for money
#  9. de_xuat               (suggestion)        — đề xuất, gợi ý
# 10. tong_the              (overall_hotel)     — đánh giá tổng thể khách sạn
# 11. noi_chung             (general)           — chung chung, không rõ khía cạnh

ASPECT_MAPPING = {
    # ── 1. chat_luong_phong ───────────────────────────────────
    "rooms general":                  "chat_luong_phong",
    "rooms quality":                  "chat_luong_phong",
    "rooms comfort":                  "chat_luong_phong",
    "rooms design_features":          "chat_luong_phong",
    "rooms miscellaneous":            "chat_luong_phong",
    "room_amenities general":         "chat_luong_phong",
    "room_amenities quality":         "chat_luong_phong",
    "room_amenities comfort":         "chat_luong_phong",
    "room_amenities design_features": "chat_luong_phong",
    # ── 2. ve_sinh ────────────────────────────────────────────
    "rooms cleanliness":              "ve_sinh",
    "room_amenities cleanliness":     "ve_sinh",
    "hotel cleanliness":              "ve_sinh",
    # ── 3. dich_vu ────────────────────────────────────────────
    "service general":                "dich_vu",
    # ── 4. vi_tri ─────────────────────────────────────────────
    "location general":               "vi_tri",
    # ── 5. am_thuc ────────────────────────────────────────────
    "food general":                   "am_thuc",
    "food quality":                   "am_thuc",
    "food style_options":             "am_thuc",
    "drinks quality":                 "am_thuc",
    "drinks style_options":           "am_thuc",
    "food_drinks quality":            "am_thuc",
    "food_drinks style_options":      "am_thuc",
    "food_drinks miscellaneous":      "am_thuc",
    # ── 6. nha_hang ───────────────────────────────────────────
    "restaurant general":             "nha_hang",
    "restaurant miscellaneous":       "nha_hang",
    # ── 7. khong_gian ─────────────────────────────────────────
    "ambience general":               "khong_gian",
    "hotel design_features":          "khong_gian",
    "hotel comfort":                  "khong_gian",
    # ── 8. gia_ca ─────────────────────────────────────────────
    "hotel prices":                   "gia_ca",
    "rooms prices":                   "gia_ca",
    "room_amenities prices":          "gia_ca",
    "food prices":                    "gia_ca",
    "food_drinks prices":             "gia_ca",
    "drinks prices":                  "gia_ca",
    "restaurant prices":              "gia_ca",
    # ── 9. de_xuat — keyword heuristic (xem bên dưới) ─────────
    # ── 10. tong_the ──────────────────────────────────────────
    "hotel general":                  "tong_the",
    "hotel quality":                  "tong_the",
    "hotel miscellaneous":            "tong_the",
    # ── 11. noi_chung ─────────────────────────────────────────
    "food recommendation":            "noi_chung",
}

# Keyword heuristic → de_xuat
SUGGEST_KEYWORDS = [
    "recommend", "suggest", "worth", "should stay", "must stay", "would go back",
    "giới thiệu", "nên ở", "đáng ở", "đáng tiền",
    "recommande", "conseille",
    "empfehle", "empfehlen",
    "recomiend", "recomiendo",
    "recomend", "recomendo",
    "推荐",
]
SUGGEST_ELIGIBLE = {
    "hotel general", "hotel quality", "hotel miscellaneous",
    "rooms general", "service general",
}

def map_category(category, sentence=""):
    """Trả về tên aspect (string) hoặc None nếu không thuộc domain hotel."""
    cat_lower  = str(category).lower().strip()
    sent_lower = str(sentence).lower()

    # Heuristic: de_xuat
    if cat_lower in SUGGEST_ELIGIBLE:
        if any(kw in sent_lower for kw in SUGGEST_KEYWORDS):
            return "de_xuat"

    return ASPECT_MAPPING.get(cat_lower)  # None nếu không thuộc hotel

# ── Test mapping ──────────────────────────────────────────────────────────────
test_cases = [
    ("rooms general",             "",                                    "chat_luong_phong"),
    ("room_amenities quality",    "",                                    "chat_luong_phong"),
    ("rooms cleanliness",         "",                                    "ve_sinh"),
    ("hotel cleanliness",         "",                                    "ve_sinh"),
    ("service general",           "",                                    "dich_vu"),
    ("location general",          "",                                    "vi_tri"),
    ("food quality",              "",                                    "am_thuc"),
    ("food_drinks quality",       "",                                    "am_thuc"),
    ("restaurant general",        "",                                    "nha_hang"),
    ("ambience general",          "",                                    "khong_gian"),
    ("hotel design_features",     "",                                    "khong_gian"),
    ("hotel prices",              "",                                    "gia_ca"),
    ("rooms prices",              "",                                    "gia_ca"),
    ("hotel general",             "",                                    "tong_the"),
    ("hotel quality",             "",                                    "tong_the"),
    ("hotel general",             "I would recommend this hotel",       "de_xuat"),
    ("hotel general",             "Tôi muốn giới thiệu khách sạn này",  "de_xuat"),
    ("food recommendation",       "",                                    "noi_chung"),
]
print("\nTest mapping:")
all_ok = True
for cat, sent, expected in test_cases:
    result = map_category(cat, sent)
    ok = (result == expected)
    if not ok:
        all_ok = False
    status = "✓" if ok else f"✗ (expected {expected})"
    print(f"  {status} {str(result):<22} ← '{cat}'")
print("\nAll tests passed!" if all_ok else "\nCó lỗi mapping, kiểm tra lại!")

# ── 9. Transform: mỗi triplet → 1 row, câu nhiều aspect → chung id ───────────
#
# Output columns: id | text | sentiment | aspect
# id: gán theo câu (sentence-level), các triplet cùng câu → cùng id

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
        entity, attribute, sentiment = triplet
        aspect = map_category(attribute, sentence)
        if aspect is None:
            continue
        valid_triplets.append((aspect, sentiment))

    if not valid_triplets:
        continue  # câu không có aspect nào thuộc hotel → bỏ qua

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

# Dedup: giữ 1 row trong mỗi nhóm (text, sentiment, aspect) trùng nhau
n_before = len(df_output)
df_output = df_output.drop_duplicates(subset=["text", "sentiment", "aspect"], keep="first")
n_after = len(df_output)
print(f"\nDedup: {n_before:,} → {n_after:,} rows (xóa {n_before - n_after:,} duplicates)")

print(f"\nTotal rows (1 triplet = 1 row): {len(df_output):,}")
print(f"Total unique sentences (id):    {df_output['id'].nunique():,}")

print("\nPhân phối aspect:")
for asp, cnt in df_output["aspect"].value_counts().items():
    print(f"  {asp:<25s}: {cnt:,}")

print("\nPhân phối sentiment:")
print(df_output["sentiment"].value_counts().to_string())

print("\nPhân phối ngôn ngữ:")
print(df_output["language"].value_counts().to_string())

print("\nSample output (id, text, sentiment, aspect):")
print(df_output[["id", "text", "sentiment", "aspect"]].head(15).to_string())

# ── 10. Lưu kết quả ───────────────────────────────────────────────────────────
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

save_cols    = ["id", "text", "sentiment", "aspect"]
save_cols_ex = ["id", "text", "sentiment", "aspect", "language", "split"]

# 1. Toàn bộ
path = os.path.join(output_dir, "hotel_v2_all.csv")
df_output[save_cols_ex].to_csv(path, index=False, encoding="utf-8-sig")
print(f"\n[ALL]        {path}  ({len(df_output):,} rows)")

# 2. Theo ngôn ngữ
for lang in TARGET_LANGS:
    df_lang = df_output[df_output["language"] == lang]
    if len(df_lang) == 0:
        continue
    path = os.path.join(output_dir, f"hotel_v2_{lang}.csv")
    df_lang[save_cols].to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[{lang.upper():5s}]       {path}  ({len(df_lang):,} rows)")

# 3. Theo split
for sp in ["train", "validation", "test"]:
    df_sp = df_output[df_output["split"] == sp]
    if len(df_sp) == 0:
        continue
    path = os.path.join(output_dir, f"hotel_v2_{sp}.csv")
    df_sp[save_cols].to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[{sp:12s}] {path}  ({len(df_sp):,} rows)")

print("\nDone! Tất cả file đã lưu vào thư mục 'output/'")
