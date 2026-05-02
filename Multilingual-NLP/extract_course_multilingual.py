# =============================================================================
# Extract Course Domain — VI, EN, FR, DE, ZH, ES, PT
# Dataset: Multilingual-NLP/M-ABSA
# =============================================================================
# Chạy: python extract_course_multilingual.py
# Hoặc copy từng section vào Jupyter notebook
# =============================================================================

# %% [markdown]
# ## 1. Cài đặt & Import

# %%
# datasets và pandas đã có sẵn, chỉ cài thêm langid (~1MB, nhanh)
# !pip install -q langid

import ast
import os
import pandas as pd
import langid
from datasets import load_dataset

# Giới hạn langid trong 21 ngôn ngữ của dataset → chính xác hơn
DATASET_LANGS = ["ar", "da", "de", "en", "es", "fr", "hi", "hr",
                 "id", "ja", "ko", "nl", "pt", "ru", "sk", "sv",
                 "sw", "th", "tr", "vi", "zh"]
langid.set_languages(DATASET_LANGS)

print("Loading M-ABSA dataset...")
ds = load_dataset("Multilingual-NLP/M-ABSA")
print("Done!", {k: len(v) for k, v in ds.items()})


# %% [markdown]
# ## 2. Gộp tất cả splits

# %%
all_dfs = []
for split_name in ds.keys():
    df_tmp = ds[split_name].to_pandas()
    df_tmp["split"] = split_name
    all_dfs.append(df_tmp)

df_all = pd.concat(all_dfs, ignore_index=True)
print(f"Total rows: {len(df_all):,}")


# %% [markdown]
# ## 3. Parse sentence & labels

# %%
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


# %% [markdown]
# ## 4. Khám phá tất cả aspect categories trong dataset

# %%
def extract_categories(text):
    if "####" not in str(text):
        return []
    try:
        labels = ast.literal_eval(text.split("####", 1)[1].strip())
        return [triplet[1] for triplet in labels if len(triplet) >= 2]
    except:
        return []

df_all["categories"] = df_all["text"].apply(extract_categories)

all_cats = set()
df_all["categories"].apply(lambda cats: all_cats.update(cats))
print(f"Total unique categories: {len(all_cats)}")
print("\nAll categories:")
for c in sorted(all_cats):
    print(f"  - {c}")


# %% [markdown]
# ## 5. Lọc domain Coursera theo aspect category

# %%
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


# %% [markdown]
# ## 6. Detect ngôn ngữ
#
# Bước 1 — Unicode script (chính xác 100%, không cần model):
#   Hiragana/Katakana → ja | CJK only → zh | Hangul → ko
#   Arabic → ar | Thai → th | Devanagari → hi | Cyrillic → ru
#
# Bước 2 — langid cho các ngôn ngữ Latin (VI/EN/FR/DE/ES/PT/...)

# %%
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

# Kiểm tra nhanh
tests = [
    ("Python の非常に詳細な入門学習。",       "ja"),
    ("这门课程非常好。",                       "zh"),
    ("Nó cung cấp lời khuyên hữu ích.",      "vi"),
    ("Excellent course.",                     "en"),
    ("Ansonsten war es super.",               "de"),
    ("Tak Coursera og Michigan University.", "da"),
    ("Un cours absolument fantastique.",      "fr"),
    ("Un curso absolutamente fantástico.",    "es"),
    ("Um curso absolutamente fantástico.",    "pt"),
]
print("Test detect_language:")
for text, expected in tests:
    result = detect_language(text)
    status = "✓" if result == expected else "✗"
    print(f"  {status} [{result:3s}] (expected {expected:3s}) {text}")

# %%
print(f"Detecting language cho {len(df_course):,} rows...")
df_course = df_course.copy()
df_course["language"] = df_course["sentence"].apply(detect_language)
print("Done!")

print("\nPhân phối ngôn ngữ (tất cả):")
print(df_course["language"].value_counts().to_string())


# %% [markdown]
# ## 7. Lọc ngôn ngữ: VI, EN, FR, DE, ZH, ES, PT

# %%
TARGET_LANGS = ["vi", "en", "fr", "de", "zh", "es", "pt"]

df_filtered = df_course[df_course["language"].isin(TARGET_LANGS)].copy()
print(f"Rows sau khi lọc ngôn ngữ: {len(df_filtered):,}")


# %% [markdown]
# ## 8. Thống kê & kiểm tra chất lượng

# %%
lang_names = {
    "vi": "Tiếng Việt",        "en": "Tiếng Anh",
    "fr": "Tiếng Pháp",        "de": "Tiếng Đức",
    "zh": "Tiếng Trung",       "es": "Tiếng Tây Ban Nha",
    "pt": "Tiếng Bồ Đào Nha",
}

print("=" * 55)
print("KẾT QUẢ SAU KHI LỌC")
print("=" * 55)
print(f"Tổng số câu: {len(df_filtered):,}")

print("\nPhân phối theo ngôn ngữ:")
for lang, cnt in df_filtered["language"].value_counts().items():
    print(f"  {lang.upper():5s} ({lang_names.get(lang, lang)}): {cnt:,} câu")

print("\nPhân phối theo split:")
print(df_filtered["split"].value_counts().to_string())

print("\nNgôn ngữ × Split:")
print(df_filtered.groupby(["language", "split"]).size().unstack(fill_value=0).to_string())

# %%
# Xem 3 ví dụ cho mỗi ngôn ngữ
for lang in TARGET_LANGS:
    subset = df_filtered[df_filtered["language"] == lang]
    if len(subset) == 0:
        print(f"[{lang.upper()}] Không có câu nào.")
        continue
    print(f"\n{'='*55}")
    print(f"[{lang.upper()}] {lang_names.get(lang, lang)} — {len(subset):,} câu")
    print(f"{'='*55}")
    for _, row in subset.sample(min(3, len(subset)), random_state=42).iterrows():
        print(f"  Sentence : {row['sentence'][:150]}")
        print(f"  Labels   : {row['labels']}")
        print()


# %% [markdown]
# ## 9. Normalize sentiment labels (POS/NEG/NEU → positive/negative/neutral)

# %%
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

# Kiểm tra các giá trị sentiment sau normalize
all_sentiments = set()
df_filtered["labels"].apply(lambda lbls: [all_sentiments.add(t[2]) for t in lbls if len(t) == 3])
print("Unique sentiment values sau normalize:", sorted(all_sentiments))

# Xem ví dụ ZH và ES sau normalize
for lang in ["zh", "es"]:
    sample = df_filtered[df_filtered["language"] == lang].head(2)
    print(f"\n[{lang.upper()}] sample:")
    for _, row in sample.iterrows():
        print(f"  {row['labels']}")


# %% [markdown]
# ## 10. Map sang 15 aspect labels & tạo output format
#
# LECTURER:   0=Teaching_Skill  1=Knowledge  2=Experience  3=Behavior  4=Support
# COURSE:     5=Curriculum      6=Materials  7=Workload    8=Assignments
# ASSESSMENT: 9=Grading        10=Exams
# FACILITIES: 11=Classroom     12=Platforms
# OTHERS:     13=General       14=Recommendation

# %%
ASPECT_LABELS = [
    "Teaching_Skill",  # 0  - LECTURER
    "Knowledge",       # 1
    "Experience",      # 2
    "Behavior",        # 3
    "Support",         # 4
    "Curriculum",      # 5  - COURSE
    "Materials",       # 6
    "Workload",        # 7
    "Assignments",     # 8
    "Grading",         # 9  - ASSESSMENT
    "Exams",           # 10
    "Classroom",       # 11 - FACILITIES
    "Platforms",       # 12
    "General",         # 13 - OTHERS
    "Recommendation",  # 14
]

CATEGORY_TO_ASPECT = {
    # ── LECTURER: Teaching_Skill (0) ──────────────────────────
    "faculty general":                0,
    "faculty comprehensiveness":      0,
    "faculty relatability":           0,
    "faculty value":                  0,
    "teaching_setup":                 0,
    "presentation quality":           0,
    "presentation quantity":          0,
    "presentation comprehensiveness": 0,
    "presentation relatability":      0,
    "presentation workload":          0,
    # ── LECTURER: Knowledge (1) ───────────────────────────────
    "mathematical_related_concept":   1,
    # ── LECTURER: Experience (2) ──────────────────────────────
    "instructor":                     2,
    # ── LECTURER: Support (4) ─────────────────────────────────
    "faculty response":               4,
    # ── COURSE: Curriculum (5) ────────────────────────────────
    "course general":                 5,
    "course quality":                 5,
    "course comprehensiveness":       5,
    "course relatability":            5,
    "course value":                   5,
    "course_general_feedback":        5,
    # ── COURSE: Materials (6) ─────────────────────────────────
    "material quality":               6,
    "material quantity":              6,
    "material comprehensiveness":     6,
    "material relatability":          6,
    "material workload":              6,
    # ── COURSE: Workload (7) ──────────────────────────────────
    "course workload":                7,
    # ── COURSE: Assignments (8) ───────────────────────────────
    "assignments comprehensiveness":  8,
    "assignments quality":            8,
    "assignments quantity":           8,
    "assignments relatability":       8,
    "assignments workload":           8,
    "proyecto final":                 8,
    # ── ASSESSMENT: Grading (9) ───────────────────────────────
    "grades general":                 9,
    "polarity positive":              9,
    "polarity negative":              9,
    "polarity neutral":               9,
    # ── FACILITIES: Classroom (11) ────────────────────────────
    "facilities general":             11,
    "facilities quality":             11,
    "facilities cleanliness":         11,
    "facilities comfort":             11,
    "facilities design_features":     11,
    "facilities miscellaneous":       11,
    "facilities prices":              11,
    # ── OTHERS: General (13) ──────────────────────────────────
    "null":                           13,
    "other":                          13,
}

# Keyword heuristic → Recommendation (14)
RECOMMEND_KEYWORDS = [
    "recommend", "suggest", "worth", "should take", "must take",
    "giới thiệu", "nên học", "đáng học",
    "recommande", "conseille",
    "empfehle", "empfehlen",
    "recomiend", "recomiendo",
    "recomend", "recomendo",
    "推荐",
]
RECOMMEND_ELIGIBLE = {
    "course general", "course quality", "course_general_feedback", "null", "other"
}

def map_category_to_aspect(category, sentence=""):
    """Trả về (aspect_name, aspect_label_index) hoặc (None, None)."""
    cat_lower  = str(category).lower().strip()
    sent_lower = str(sentence).lower()

    # Heuristic Recommendation
    if cat_lower in RECOMMEND_ELIGIBLE:
        if any(kw in sent_lower for kw in RECOMMEND_KEYWORDS):
            return ASPECT_LABELS[14], 14

    idx = CATEGORY_TO_ASPECT.get(cat_lower)
    if idx is None:
        return None, None
    return ASPECT_LABELS[idx], idx

# ── Test mapping ──────────────────────────────────────────────
test_cases = [
    ("faculty general",              "",                                   0),
    ("Mathematical_Related_Concept", "",                                   1),
    ("Instructor",                   "",                                   2),
    ("faculty response",             "",                                   4),
    ("course general",               "",                                   5),
    ("material quality",             "",                                   6),
    ("course workload",              "",                                   7),
    ("assignments quality",          "",                                   8),
    ("grades general",               "",                                   9),
    ("facilities general",           "",                                  11),
    ("NULL",                         "",                                  13),
    ("Teaching_Setup",               "",                                   0),
    ("presentation quality",         "",                                   0),
    ("course general",               "I would recommend this course",     14),
    ("course general",               "Tôi muốn giới thiệu khóa học này",  14),
]
print("Test mapping:")
all_ok = True
for cat, sent, expected_idx in test_cases:
    name, idx = map_category_to_aspect(cat, sent)
    ok = (idx == expected_idx)
    if not ok:
        all_ok = False
    status = "✓" if ok else f"✗ (expected {expected_idx})"
    print(f"  {status} [{str(idx):>2}] {str(name):<20} ← '{cat}'")
print("\nAll tests passed!" if all_ok else "\nCó lỗi mapping, kiểm tra lại!")

# %%
# TRANSFORM: mỗi triplet → 1 row
# Columns: text | aspect | entity | attribute | aspect_label | sentiment | language | split

rows = []
for _, row in df_filtered.iterrows():
    sentence = row["sentence"]
    language = row["language"]
    split    = row["split"]

    for triplet in row["labels"]:
        if len(triplet) != 3:
            continue
        entity, attribute, sentiment = triplet

        aspect_name, aspect_idx = map_category_to_aspect(attribute, sentence)
        if aspect_name is None:
            continue  # bỏ qua category không thuộc giáo dục

        rows.append({
            "text":         sentence,
            "aspect":       aspect_name,
            "entity":       entity,
            "attribute":    attribute,
            "aspect_label": aspect_idx,
            "sentiment":    sentiment,
            "language":     language,
            "split":        split,
        })

df_output = pd.DataFrame(rows)
print(f"Total rows (1 triplet = 1 row): {len(df_output):,}")

print("\nPhân phối aspect:")
aspect_dist = (
    df_output.groupby(["aspect_label", "aspect"])
    .size().reset_index(name="count")
    .sort_values("aspect_label")
)
for _, r in aspect_dist.iterrows():
    print(f"  [{int(r.aspect_label):2d}] {r.aspect:<20s}: {r['count']:,}")

print("\nPhân phối sentiment:")
print(df_output["sentiment"].value_counts().to_string())

print("\nPhân phối ngôn ngữ:")
print(df_output["language"].value_counts().to_string())

print("\nSample output:")
print(df_output.head(10).to_string())


# %% [markdown]
# ## 11. Lưu kết quả

# %%
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

save_cols = ["text", "aspect", "entity", "attribute", "aspect_label", "sentiment", "language", "split"]

# 1. Toàn bộ
path = os.path.join(output_dir, "course_vi_en_fr_de_zh_es_pt_all.csv")
df_output[save_cols].to_csv(path, index=False, encoding="utf-8-sig")
print(f"[ALL]        {path}  ({len(df_output):,} rows)")

# 2. Theo ngôn ngữ
for lang in TARGET_LANGS:
    df_lang = df_output[df_output["language"] == lang]
    if len(df_lang) == 0:
        continue
    path = os.path.join(output_dir, f"course_{lang}.csv")
    df_lang[save_cols].to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[{lang.upper():5s}]       {path}  ({len(df_lang):,} rows)")

# 3. Theo split
for sp in ["train", "validation", "test"]:
    df_sp = df_output[df_output["split"] == sp]
    if len(df_sp) == 0:
        continue
    path = os.path.join(output_dir, f"course_multilingual_{sp}.csv")
    df_sp[save_cols].to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[{sp:12s}] {path}  ({len(df_sp):,} rows)")

print("\nDone! Tất cả file đã lưu vào thư mục 'output/'")
