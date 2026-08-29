import sys

src = open("graph/temporal_graph.py", encoding="utf-8").read()

# Replace the slow per-account loop with a vectorized groupby approach
old_behav = '''def extract_account_behavioral_features(events_df, as_of_ts, account_ids, accounts_df):
    past = events_df[events_df["timestamp"] <= as_of_ts].copy()
    acc_created = accounts_df.set_index("account_id")["created_ts"]
    SECONDS_PER_DAY = 86400
    rows = []
    for acc in account_ids:
        acc_evts = past[past["account_id"] == acc]
        orders = acc_evts[acc_evts["event_type"] == "order_placed"]
        returns = acc_evts[acc_evts["event_type"] == "order_returned"]
        sessions = acc_evts[acc_evts["event_type"] == "session_start"]
        refs_sent = past[(past["event_type"] == "referral") & (past["referrer_id"] == acc)]
        refs_recv = acc_evts[acc_evts["event_type"] == "referral"]
        n_orders = len(orders)
        n_returns = len(returns)
        n_sessions = sessions["session_id"].nunique()
        amounts = orders["amount"].dropna().values
        promo_orders = orders["promo_code"].notna().sum()
        created_ts = acc_created.get(acc, as_of_ts)
        age_days = max(0, (as_of_ts - created_ts) / SECONDS_PER_DAY)
        first_order_age = age_days
        if n_orders > 0:
            first_order_ts = orders["timestamp"].min()
            first_order_age = max(0, (first_order_ts - created_ts) / SECONDS_PER_DAY)
        order_days = 0
        if n_orders > 0:
            order_days = orders["timestamp"].apply(lambda t: (t - created_ts) // SECONDS_PER_DAY).nunique()
        burst_score = 0
        if n_orders >= 2:
            ts_sorted = sorted(orders["timestamp"].values)
            window_sec = 48 * 3600
            lo = 0
            for hi in range(len(ts_sorted)):
                while ts_sorted[hi] - ts_sorted[lo] > window_sec:
                    lo += 1
                burst_score = max(burst_score, hi - lo + 1)
        rows.append({"account_id": acc, "n_orders": n_orders, "n_sessions": n_sessions,
                     "n_returns": n_returns, "return_rate": n_returns / max(n_orders, 1),
                     "promo_rate": promo_orders / max(n_orders, 1), "has_promo": int(promo_orders > 0),
                     "mean_order_amount": float(np.mean(amounts)) if len(amounts) > 0 else 0.0,
                     "std_order_amount": float(np.std(amounts)) if len(amounts) > 1 else 0.0,
                     "max_order_amount": float(np.max(amounts)) if len(amounts) > 0 else 0.0,
                     "order_days_active": order_days, "mean_daily_orders": n_orders / max(order_days, 1),
                     "account_age_days": age_days, "first_order_age_days": first_order_age,
                     "burst_score": burst_score, "n_referrals_sent": len(refs_sent),
                     "n_referrals_received": len(refs_recv)})
    return pd.DataFrame(rows)'''

new_behav = '''def extract_account_behavioral_features(events_df, as_of_ts, account_ids, accounts_df):
    """
    Vectorized behavioral feature extraction.
    Replaces per-account pandas filter loop (O(n_accs * n_events)) with
    groupby aggregations (O(n_events)) -- critical for 5K accounts / 40K events.
    """
    past = events_df[events_df["timestamp"] <= as_of_ts].copy()
    acc_created = accounts_df.set_index("account_id")["created_ts"]
    SECONDS_PER_DAY = 86400
    acc_set = set(account_ids)

    # Filter to relevant accounts only
    past_acc = past[past["account_id"].isin(acc_set)]

    orders   = past_acc[past_acc["event_type"] == "order_placed"].copy()
    returns  = past_acc[past_acc["event_type"] == "order_returned"]
    sessions = past_acc[past_acc["event_type"] == "session_start"]

    # refs_sent: referrer_id in account_ids
    refs_sent_all = past[past["event_type"] == "referral"].dropna(subset=["referrer_id"])
    refs_sent_all = refs_sent_all[refs_sent_all["referrer_id"].isin(acc_set)]
    refs_recv     = past_acc[past_acc["event_type"] == "referral"]

    # Aggregate per account
    order_grp   = orders.groupby("account_id")
    n_orders_s  = order_grp.size().rename("n_orders")
    amounts_agg = order_grp["amount"].agg(["mean", "std", "max"]).rename(
        columns={"mean": "mean_order_amount", "std": "std_order_amount", "max": "max_order_amount"})
    promo_agg   = order_grp["promo_code"].apply(lambda x: x.notna().sum()).rename("promo_orders")
    first_ts    = order_grp["timestamp"].min().rename("first_order_ts")

    n_returns_s  = returns.groupby("account_id").size().rename("n_returns")
    n_sessions_s = sessions.groupby("account_id")["session_id"].nunique().rename("n_sessions")
    n_refs_sent  = refs_sent_all.groupby("referrer_id").size().rename("n_referrals_sent")
    n_refs_recv  = refs_recv.groupby("account_id").size().rename("n_referrals_received")

    # Per-account burst score (still needs a loop but only over order timestamps)
    def _burst(ts_series):
        ts = sorted(ts_series.values)
        if len(ts) < 2:
            return len(ts)
        window_sec = 48 * 3600; lo = 0; best = 1
        for hi in range(len(ts)):
            while ts[hi] - ts[lo] > window_sec:
                lo += 1
            best = max(best, hi - lo + 1)
        return best
    burst_s = order_grp["timestamp"].apply(_burst).rename("burst_score")

    # Order-days active (distinct days with an order)
    def _order_days(grp):
        created = acc_created.get(grp.name, as_of_ts)
        return grp["timestamp"].apply(lambda t: (t - created) // SECONDS_PER_DAY).nunique()
    order_days_s = order_grp.apply(_order_days).rename("order_days_active")

    # Assemble final DataFrame
    base = pd.DataFrame({"account_id": account_ids}).set_index("account_id")
    for s in [n_orders_s, amounts_agg, promo_agg, first_ts, burst_s, order_days_s,
               n_returns_s, n_sessions_s, n_refs_recv]:
        base = base.join(s, how="left")
    base = base.join(n_refs_sent, how="left")

    base["n_orders"]             = base["n_orders"].fillna(0).astype(int)
    base["n_returns"]            = base["n_returns"].fillna(0).astype(int)
    base["n_sessions"]           = base["n_sessions"].fillna(0).astype(int)
    base["promo_orders"]         = base["promo_orders"].fillna(0).astype(int)
    base["n_referrals_sent"]     = base["n_referrals_sent"].fillna(0).astype(int)
    base["n_referrals_received"] = base["n_referrals_received"].fillna(0).astype(int)
    base["burst_score"]          = base["burst_score"].fillna(0).astype(int)
    base["order_days_active"]    = base["order_days_active"].fillna(0).astype(int)
    base["mean_order_amount"]    = base["mean_order_amount"].fillna(0.0)
    base["std_order_amount"]     = base["std_order_amount"].fillna(0.0)
    base["max_order_amount"]     = base["max_order_amount"].fillna(0.0)

    # Derived features
    base["return_rate"]  = base["n_returns"] / base["n_orders"].clip(lower=1)
    base["promo_rate"]   = base["promo_orders"] / base["n_orders"].clip(lower=1)
    base["has_promo"]    = (base["promo_orders"] > 0).astype(int)
    base["mean_daily_orders"] = base["n_orders"] / base["order_days_active"].clip(lower=1)

    # Age features (vectorized)
    created_series = pd.Series(
        [acc_created.get(acc, as_of_ts) for acc in base.index],
        index=base.index, name="created_ts"
    )
    base["account_age_days"]    = ((as_of_ts - created_series) / SECONDS_PER_DAY).clip(lower=0)
    base["first_order_age_days"] = base["account_age_days"]  # default if no orders
    has_orders = base["first_order_ts"].notna()
    base.loc[has_orders, "first_order_age_days"] = (
        (base.loc[has_orders, "first_order_ts"] - created_series[has_orders]) / SECONDS_PER_DAY
    ).clip(lower=0)

    base = base.drop(columns=["promo_orders", "first_order_ts"], errors="ignore")
    return base.reset_index()'''

if old_behav not in src:
    print("ERROR: old_behav block not found -- check indentation/content")
    sys.exit(1)
src = src.replace(old_behav, new_behav, 1)
with open("graph/temporal_graph.py", "w", encoding="utf-8", newline="") as f:
    f.write(src)
import ast; ast.parse(src)
print("Vectorized behavioral extractor written. Syntax OK.")