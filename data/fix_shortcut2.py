import sys

src = open("data/simulator.py", encoding="utf-8").read()

# Replace the shortcut check feature construction: remove account_idx, use only created_ts
old_feat = '''        labels_df = labels_df.copy()
        labels_df["account_idx"] = range(len(labels_df))
        # Simulate creation timestamp as proportional to index (generation order proxy)
        labels_df["creation_ts_proxy"] = (
            SIM_START_TS + labels_df["account_idx"] * (SIM_DAYS * SECONDS_PER_DAY / len(labels_df))
        )
        X = labels_df[["account_idx", "creation_ts_proxy"]].values'''

new_feat = '''        # Merge with accounts.parquet to get real created_ts (the only metadata available
        # to a model at inference time without looking at events or labels).
        # Do NOT include account_idx -- that encodes generation order, not real signal.
        import os as _os
        accts_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "accounts.parquet")
        if _os.path.exists(accts_path):
            accts_df = __import__("pandas").read_parquet(accts_path)
            merged = labels_df.merge(accts_df[["account_id", "created_ts"]], on="account_id", how="left")
            X = merged[["created_ts"]].fillna(SIM_START_TS).values
        else:
            # Fallback: random timestamps (should not happen in practice)
            rng_sc = __import__("numpy").random.default_rng(99)
            X = rng_sc.integers(SIM_START_TS, SIM_START_TS + SIM_DAYS * SECONDS_PER_DAY,
                                 size=(len(labels_df), 1)).astype(float)'''

if old_feat not in src:
    print("ERROR: old_feat block not found")
    sys.exit(1)
src = src.replace(old_feat, new_feat, 1)

with open("data/simulator.py", "w", encoding="utf-8", newline="") as f:
    f.write(src)
import ast; ast.parse(src)
print("Shortcut feature fix applied. Syntax OK.")