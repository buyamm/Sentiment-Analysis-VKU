# =============================================================================
# Thống kê dataset v2 để điền vào mục 3.8 báo cáo kỹ thuật
# Chạy sau khi đã chạy extract_course_multilingual_v2.py
# =============================================================================

import os
import pandas as pd

OUTPUT_DIR = "output_v2"
ALL_FILE   = os.path.join(OUTPUT_DIR, "course_v2_all.csv")

if not os.path.exists(ALL_FILE):
    print(f"[ERROR] Không tìm thấy {ALL_FILE}")
    print("Hãy chạy extract_course_multilingual_v2.py trước.")
    exit(1)

df = pd.read_csv(ALL_FILE, encoding="utf-8-sig")
print(f"Loaded: {len(df):,} rows | {df['id'].nunique():,} unique sentences\n")

LANG_NAMES = {
    "vi": "Vietnamese", "en": "English", "fr": "French",
    "de": "German",     "zh": "Chinese", "es": "Spanish", "pt": "Portuguese",
}
ASPECT_EN = {
    "ky_nang_giang_day": "Teaching Skill",
    "kinh_nghiem":       "Experience",
    "hanh_vi":           "Behavior",
    "bai_tap":           "Assignments",
    "ham_diem":          "Grading",
    "cung_cap_tai_lieu": "Materials",
    "kien_thuc":         "Knowledge",
    "chuong_trinh_hoc":  "Curriculum",
    "thiet_bi_day_hoc":  "Facilities",
    "de_xuat":           "Suggestion",
    "noi_chung":         "General",
}

# ── Table 1: Overall summary ──────────────────────────────────────────────────
print("=" * 55)
print("TABLE 1 — Overall Summary")
print("=" * 55)
print(f"  Total aspect-level samples : {len(df):,}")
print(f"  Total unique sentences     : {df['id'].nunique():,}")
print(f"  Languages                  : {df['language'].nunique()}")
print(f"  Aspect categories          : {df['aspect'].nunique()}")
print(f"  Splits                     : {sorted(df['split'].unique())}")
print()

# ── Table 2: Distribution by Language × Split ─────────────────────────────────
print("=" * 65)
print("TABLE 2 — Samples by Language × Split (aspect-level rows)")
print("=" * 65)
pivot = df.groupby(["language", "split"]).size().unstack(fill_value=0)
# Ensure column order
for col in ["train", "validation", "test"]:
    if col not in pivot.columns:
        pivot[col] = 0
pivot = pivot[["train", "validation", "test"]]
pivot["TOTAL"] = pivot.sum(axis=1)
pivot.index = [f"{LANG_NAMES.get(l, l)} ({l.upper()})" for l in pivot.index]

# Header
print(f"  {'Language':<25} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for lang, row in pivot.iterrows():
    print(f"  {lang:<25} {int(row['train']):>8,} {int(row['validation']):>8,} {int(row['test']):>8,} {int(row['TOTAL']):>8,}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
totals = pivot[["train","validation","test","TOTAL"]].sum()
print(f"  {'TOTAL':<25} {int(totals['train']):>8,} {int(totals['validation']):>8,} {int(totals['test']):>8,} {int(totals['TOTAL']):>8,}")
print()

# ── Table 3: Distribution by Aspect ──────────────────────────────────────────
print("=" * 55)
print("TABLE 3 — Samples by Aspect Category")
print("=" * 55)
asp_counts = df["aspect"].value_counts().sort_index()
total_asp  = asp_counts.sum()
print(f"  {'Aspect (VI)':<22} {'Aspect (EN)':<20} {'Count':>7} {'%':>6}")
print(f"  {'-'*22} {'-'*20} {'-'*7} {'-'*6}")
for asp, cnt in asp_counts.items():
    en = ASPECT_EN.get(asp, asp)
    print(f"  {asp:<22} {en:<20} {cnt:>7,} {cnt/total_asp*100:>5.1f}%")
print(f"  {'-'*22} {'-'*20} {'-'*7} {'-'*6}")
print(f"  {'TOTAL':<22} {'':<20} {total_asp:>7,} {'100.0%':>6}")
print()

# ── Table 4: Distribution by Sentiment ───────────────────────────────────────
print("=" * 40)
print("TABLE 4 — Samples by Sentiment")
print("=" * 40)
sent_counts = df["sentiment"].value_counts()
total_sent  = sent_counts.sum()
print(f"  {'Sentiment':<12} {'Count':>7} {'%':>6}")
print(f"  {'-'*12} {'-'*7} {'-'*6}")
for sent, cnt in sent_counts.items():
    print(f"  {sent:<12} {cnt:>7,} {cnt/total_sent*100:>5.1f}%")
print(f"  {'-'*12} {'-'*7} {'-'*6}")
print(f"  {'TOTAL':<12} {total_sent:>7,} {'100.0%':>6}")
print()

# ── Table 5: Aspect × Sentiment heatmap ──────────────────────────────────────
print("=" * 75)
print("TABLE 5 — Aspect × Sentiment")
print("=" * 75)
heat = df.groupby(["aspect", "sentiment"]).size().unstack(fill_value=0)
for col in ["positive", "neutral", "negative", "conflict"]:
    if col not in heat.columns:
        heat[col] = 0
heat = heat[["positive", "neutral", "negative", "conflict"]]
heat["total"] = heat.sum(axis=1)
print(f"  {'Aspect':<22} {'Positive':>9} {'Neutral':>9} {'Negative':>9} {'Conflict':>9} {'Total':>7}")
print(f"  {'-'*22} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*7}")
for asp, row in heat.iterrows():
    print(f"  {asp:<22} {int(row['positive']):>9,} {int(row['neutral']):>9,} {int(row['negative']):>9,} {int(row['conflict']):>9,} {int(row['total']):>7,}")
print(f"  {'-'*22} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*7}")
h_tot = heat[["positive","neutral","negative","conflict","total"]].sum()
print(f"  {'TOTAL':<22} {int(h_tot['positive']):>9,} {int(h_tot['neutral']):>9,} {int(h_tot['negative']):>9,} {int(h_tot['conflict']):>9,} {int(h_tot['total']):>7,}")
print()

print("Done! Copy các bảng trên vào mục 3.8 báo cáo.")
