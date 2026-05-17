# =============================================================================
# Check & remove duplicates: (text, sentiment, aspect) trong tất cả CSV output
# Giữ lại 1 row trong mỗi nhóm trùng, ghi đè file gốc.
# =============================================================================

import os
import pandas as pd

OUTPUT_DIR = "output"
CHECK_COLS = ["text", "sentiment", "aspect"]

# Lấy tất cả file CSV trong output/ (bỏ qua lock file)
csv_files = sorted([
    f for f in os.listdir(OUTPUT_DIR)
    if f.endswith(".csv") and not f.startswith(".~")
])

print("=" * 70)
print(f"  {'FILE':<42} {'ROWS':>7} {'DUPS':>7} {'AFTER':>7}")
print("=" * 70)

total_removed = 0

for fname in csv_files:
    fpath = os.path.join(OUTPUT_DIR, fname)
    df = pd.read_csv(fpath, encoding="utf-8-sig")

    # Chỉ check các cột tồn tại trong file
    cols = [c for c in CHECK_COLS if c in df.columns]
    if not cols:
        print(f"  [SKIP] {fname} — không có cột {CHECK_COLS}")
        continue

    n_before = len(df)
    n_dups   = df.duplicated(subset=cols).sum()

    if n_dups > 0:
        df_clean = df.drop_duplicates(subset=cols, keep="first")
        df_clean.to_csv(fpath, index=False, encoding="utf-8-sig")
        n_after = len(df_clean)
        flag = " ✓ đã xóa"
    else:
        n_after = n_before
        flag = ""

    total_removed += n_dups
    print(f"  {fname:<42} {n_before:>7,} {n_dups:>7,} {n_after:>7,}{flag}")

print("=" * 70)
print(f"  Tổng số duplicate đã xóa: {total_removed:,}")
print("\nDone!")
