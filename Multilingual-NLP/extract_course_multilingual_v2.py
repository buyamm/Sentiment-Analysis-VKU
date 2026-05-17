# =============================================================================
# Extract Course Domain — VI, EN, FR, DE, ZH, ES, PT  (v2)
# Dataset: Multilingual-NLP/M-ABSA
# Mapping mới: 11 aspect labels (tiếng Việt)
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
    """Tách sentence và labels từ: sentence####[[aspect, category, sentiment]]"""
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

# ── 4. Lọc domain Coursera theo aspect category ───────────────────────────────
COURSERA_CATEGORIES_LOWER = {
    "assignments comprehensiveness", "assignments quality", "assignments quantity",
    "assignments relatability",      "assignments workload",
    "course comprehensiveness",      "course general",      "course quality",
    "course relatability",           "course value",        "course workload",
    "faculty comprehensiveness",     "faculty general",     "faculty relatability",
    "faculty response",              "faculty value",
    "grades general",
    "material comprehensiveness",    "material quality",    "material quantity",
    "material relatability",         "material workload",
    "presentation comprehensiveness","presentation quality","presentation quantity",
    "presentation relatability",     "presentation workload",
    "course_general_feedback",
    "instructor",
    "mathematical_related_concept",
    "teaching_setup",
    "facilities general",   "facilities quality",  "facilities cleanliness",
    "facilities comfort",   "facilities design_features",
    "facilities miscellaneous", "facilities prices",
    "polarity positive",    "polarity negative",   "polarity neutral",
    "proyecto final",
}

def is_coursera(labels):
    if not labels:
        return False
    for triplet in labels:
        if len(triplet) >= 2 and str(triplet[1]).lower() in COURSERA_CATEGORIES_LOWER:
            return True
    return False

df_all["is_coursera"] = df_all["labels"].apply(is_coursera)
df_course = df_all[df_all["is_coursera"]].copy()
print(f"Rows sau khi lọc domain coursera: {len(df_course):,}")
print(df_course["split"].value_counts().to_string())

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

print(f"Detecting language cho {len(df_course):,} rows...")
df_course = df_course.copy()
df_course["language"] = df_course["sentence"].apply(detect_language)
print("Done!")
print("\nPhân phối ngôn ngữ (tất cả):")
print(df_course["language"].value_counts().to_string())

# ── 6. Lọc ngôn ngữ: VI, EN, FR, DE, ZH, ES, PT ─────────────────────────────
TARGET_LANGS = ["vi", "en", "fr", "de", "zh", "es", "pt"]
df_filtered = df_course[df_course["language"].isin(TARGET_LANGS)].copy()
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
            aspect, category, sentiment = triplet
            sentiment_norm = SENTIMENT_MAP.get(str(sentiment).lower(), str(sentiment).lower())
            result.append([aspect, category, sentiment_norm])
        else:
            result.append(triplet)
    return result

df_filtered["labels"] = df_filtered["labels"].apply(normalize_labels)

# ── 8. Mapping mới: 11 aspect labels ─────────────────────────────────────────
#
#  1. ky_nang_giang_day   (teaching_skill)
#  2. kinh_nghiem         (experience)
#  3. hanh_vi             (behavior)
#  4. bai_tap             (assignments)
#  5. ham_diem            (grading)
#  6. cung_cap_tai_lieu   (materials)
#  7. kien_thuc           (knowledge)
#  8. chuong_trinh_hoc    (curriculum)
#  9. thiet_bi_day_hoc    (facilities)
# 10. de_xuat             (suggestion)
# 11. noi_chung           (general)

ASPECT_MAPPING = {
    # ── 1. ky_nang_giang_day ──────────────────────────────────
    "presentation quality":           "ky_nang_giang_day",
    "presentation comprehensiveness": "ky_nang_giang_day",
    "presentation quantity":          "ky_nang_giang_day",
    "presentation relatability":      "ky_nang_giang_day",
    "presentation workload":          "ky_nang_giang_day",
    "faculty value":                  "ky_nang_giang_day",
    "faculty general":                "ky_nang_giang_day",
    "instructor":                     "ky_nang_giang_day",
    "faculty comprehensiveness":      "ky_nang_giang_day",
    # ── 2. kinh_nghiem ────────────────────────────────────────
    "faculty relatability":           "kinh_nghiem",
    # ── 3. hanh_vi ────────────────────────────────────────────
    "faculty response":               "hanh_vi",
    # ── 4. bai_tap ────────────────────────────────────────────
    "assignments quality":            "bai_tap",
    "assignments quantity":           "bai_tap",
    "assignments workload":           "bai_tap",
    "assignments comprehensiveness":  "bai_tap",
    "proyecto final":                 "bai_tap",
    "assignments relatability":       "bai_tap",
    # ── 5. ham_diem ───────────────────────────────────────────
    "grades general":                 "ham_diem",
    # ── 6. cung_cap_tai_lieu ──────────────────────────────────
    "material comprehensiveness":     "cung_cap_tai_lieu",
    "material quality":               "cung_cap_tai_lieu",
    "material quantity":              "cung_cap_tai_lieu",
    "material relatability":          "cung_cap_tai_lieu",
    "material workload":              "cung_cap_tai_lieu",
    # ── 7. kien_thuc ──────────────────────────────────────────
    "mathematical_related_concept":   "kien_thuc",
    # ── 8. chuong_trinh_hoc ───────────────────────────────────
    "course quality":                 "chuong_trinh_hoc",
    "course relatability":            "chuong_trinh_hoc",
    "course workload":                "chuong_trinh_hoc",
    "course value":                   "chuong_trinh_hoc",
    "course_general_feedback":        "chuong_trinh_hoc",
    "course comprehensiveness":       "chuong_trinh_hoc",
    # ── 9. thiet_bi_day_hoc ───────────────────────────────────
    "teaching_setup":                 "thiet_bi_day_hoc",
    "facilities general":             "thiet_bi_day_hoc",
    "facilities quality":             "thiet_bi_day_hoc",
    "facilities cleanliness":         "thiet_bi_day_hoc",
    "facilities comfort":             "thiet_bi_day_hoc",
    "facilities design_features":     "thiet_bi_day_hoc",
    "facilities miscellaneous":       "thiet_bi_day_hoc",
    "facilities prices":              "thiet_bi_day_hoc",
    # ── 10. de_xuat (suggestion) — keyword heuristic ──────────
    # "course general" → de_xuat nếu có từ khóa recommend
    # ── 11. noi_chung ─────────────────────────────────────────
    "course general":                 "noi_chung",
    "null":                           "noi_chung",
    "other":                          "noi_chung",
    "polarity positive":              "noi_chung",
    "polarity negative":              "noi_chung",
    "polarity neutral":               "noi_chung",
}

# Keyword heuristic → de_xuat (suggestion)
SUGGEST_KEYWORDS = [
    "recommend", "suggest", "worth", "should take", "must take",
    "giới thiệu", "nên học", "đáng học",
    "recommande", "conseille",
    "empfehle", "empfehlen",
    "recomiend", "recomiendo",
    "recomend", "recomendo",
    "推荐",
]
SUGGEST_ELIGIBLE = {
    "course general", "course quality", "course_general_feedback", "null", "other"
}

def map_category(category, sentence=""):
    """Trả về tên aspect (string) hoặc None nếu không thuộc domain."""
    cat_lower  = str(category).lower().strip()
    sent_lower = str(sentence).lower()

    # Heuristic: de_xuat
    if cat_lower in SUGGEST_ELIGIBLE:
        if any(kw in sent_lower for kw in SUGGEST_KEYWORDS):
            return "de_xuat"

    return ASPECT_MAPPING.get(cat_lower)  # None nếu không có

# ── Test mapping ──────────────────────────────────────────────────────────────
test_cases = [
    ("presentation quality",         "",                                  "ky_nang_giang_day"),
    ("faculty general",              "",                                  "ky_nang_giang_day"),
    ("Instructor",                   "",                                  "ky_nang_giang_day"),
    ("faculty relatability",         "",                                  "kinh_nghiem"),
    ("faculty response",             "",                                  "hanh_vi"),
    ("assignments quality",          "",                                  "bai_tap"),
    ("proyecto final",               "",                                  "bai_tap"),
    ("grades general",               "",                                  "ham_diem"),
    ("material quality",             "",                                  "cung_cap_tai_lieu"),
    ("Mathematical_Related_Concept", "",                                  "kien_thuc"),
    ("course quality",               "",                                  "chuong_trinh_hoc"),
    ("course workload",              "",                                  "chuong_trinh_hoc"),
    ("Teaching_Setup",               "",                                  "thiet_bi_day_hoc"),
    ("facilities general",           "",                                  "thiet_bi_day_hoc"),
    ("course general",               "I would recommend this course",    "de_xuat"),
    ("course general",               "Tôi muốn giới thiệu khóa học này", "de_xuat"),
    ("course general",               "Great course overall.",            "noi_chung"),
    ("NULL",                         "",                                  "noi_chung"),
    ("polarity positive",            "",                                  "noi_chung"),
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

    # Lấy tất cả aspect hợp lệ của câu này
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
        continue  # câu không có aspect nào thuộc domain → bỏ qua

    # Tất cả triplet hợp lệ của câu này dùng chung sentence_id
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

# 1. Toàn bộ (có language + split để tiện lọc sau)
path = os.path.join(output_dir, "course_v2_all.csv")
df_output[save_cols_ex].to_csv(path, index=False, encoding="utf-8-sig")
print(f"\n[ALL]        {path}  ({len(df_output):,} rows)")

# 2. Theo ngôn ngữ
for lang in TARGET_LANGS:
    df_lang = df_output[df_output["language"] == lang]
    if len(df_lang) == 0:
        continue
    path = os.path.join(output_dir, f"course_v2_{lang}.csv")
    df_lang[save_cols].to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[{lang.upper():5s}]       {path}  ({len(df_lang):,} rows)")

# 3. Theo split
for sp in ["train", "validation", "test"]:
    df_sp = df_output[df_output["split"] == sp]
    if len(df_sp) == 0:
        continue
    path = os.path.join(output_dir, f"course_v2_{sp}.csv")
    df_sp[save_cols].to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[{sp:12s}] {path}  ({len(df_sp):,} rows)")

print("\nDone! Tất cả file đã lưu vào thư mục 'output/'")
