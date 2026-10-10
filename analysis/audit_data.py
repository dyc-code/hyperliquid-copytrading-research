"""Account snapshot source-data audit. No address ranking or strategy backtest.

Run from any directory with the bundled Python. Input CSV is read-only.
All monetary thresholds use the source file's unspecified monetary unit.
"""
from pathlib import Path
import hashlib
import json
import platform
import io

from source_binding import digest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "account_daily.csv"
OUTPUT = ROOT / "output" / "audit"
PARAMETERS = {
    "absolute_tolerance": 0.01,
    "illustrative_min_start_equity": 1000.0,
    "illustrative_max_net_flow_ratio": 0.10,
    "illustrative_interval_hours": [18.0, 30.0],
}


def verify_frozen_source(source_sha256):
    protocol = OUTPUT.parent / "selection" / "protocol_record.json"
    if (protocol.exists()
            and json.loads(protocol.read_text(encoding="utf-8")).get("source_sha256") != source_sha256):
        raise RuntimeError('Frozen source changed; create a separately named experiment')


def run_audit():
    source_bytes = SOURCE.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    verify_frozen_source(source_sha256)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(io.BytesIO(source_bytes))
    d = raw.copy()
    d["source_row"] = np.arange(2, len(d) + 2)
    d["date"] = pd.to_datetime(d["date"], errors="raise")
    d["timestamp"] = pd.to_datetime(d["timestamp"], format="mixed", errors="raise")
    d = d.sort_values(["user", "date"]).reset_index(drop=True)
    g = d.groupby("user", sort=False)
    tol = PARAMETERS["absolute_tolerance"]
    d["previous_date"] = g["date"].shift()
    d["gap_days"] = (d["date"] - d["previous_date"]).dt.days
    d["interval_hours"] = g["timestamp"].diff().dt.total_seconds() / 3600
    for col in ["equity", "net_deposit", "pnl"]:
        d["previous_" + col] = g[col].shift()
        d["delta_" + col] = g[col].diff()
    d["residual"] = d["equity"] - d["net_deposit"] - d["pnl"]
    d["previous_residual"] = d.groupby("user")["residual"].shift()
    d["delta_residual"] = d.groupby("user")["residual"].diff()
    d["profit_equity_flow_hypothesis"] = d["delta_equity"] - d["delta_net_deposit"]
    d["profit_pnl_hypothesis"] = d["delta_pnl"]
    d["net_flow_ratio"] = d["delta_net_deposit"].abs() / d["previous_equity"].where(d["previous_equity"] > 0)
    d["return_naive_equity"] = d["delta_equity"] / d["previous_equity"].where(d["previous_equity"] > 0)
    for label, fraction in [("end", 0.0), ("mid", 0.5), ("start", 1.0)]:
        denom = d["previous_equity"] + fraction * d["delta_net_deposit"]
        d["denominator_" + label] = denom
        d["return_flow_" + label] = d["profit_equity_flow_hypothesis"] / denom.where(denom > 0)
    d["return_pnl_change"] = d["delta_pnl"] / d["previous_equity"].where(d["previous_equity"] > 0)
    d["timing_scenario_spread"] = d[["return_flow_end", "return_flow_mid", "return_flow_start"]].max(axis=1) - d[["return_flow_end", "return_flow_mid", "return_flow_start"]].min(axis=1)
    d["all_scenario_denominators_positive"] = (d[["denominator_end", "denominator_mid", "denominator_start"]] > 0).all(axis=1)
    d.loc[~d["all_scenario_denominators_positive"], "timing_scenario_spread"] = np.nan
    d["flag_first_observation"] = d["previous_date"].isna()
    d["flag_date_gap"] = d["gap_days"].gt(1)
    d["flag_nonpositive_start_equity"] = d["previous_equity"].le(0)
    d["flag_zero_end_equity"] = d["equity"].eq(0)
    d["flag_level_mismatch"] = d["residual"].abs().gt(tol)
    d["flag_either_endpoint_mismatch"] = d["flag_level_mismatch"] | d["previous_residual"].abs().gt(tol)
    d["flag_profit_mismatch"] = d["delta_residual"].abs().gt(tol)
    d["flag_small_start_equity"] = d["previous_equity"].gt(0) & d["previous_equity"].lt(PARAMETERS["illustrative_min_start_equity"])
    d["flag_large_net_flow"] = d["net_flow_ratio"].gt(PARAMETERS["illustrative_max_net_flow_ratio"])
    d["flag_irregular_daily_interval"] = d["gap_days"].eq(1) & ~d["interval_hours"].between(*PARAMETERS["illustrative_interval_hours"])
    d["flag_return_below_minus_100pct"] = d["return_flow_end"].lt(-1)
    # Research mask: explicitly conditional, not an investment eligibility flag.
    stages = [
        ("全部相邻观测区间", ~d["flag_first_observation"]),
        ("日期连续一天", d["gap_days"].eq(1)),
        ("期初权益大于零", d["previous_equity"].gt(0)),
        ("实际间隔为18至30小时（示例）", d["interval_hours"].between(*PARAMETERS["illustrative_interval_hours"])),
        ("两个端点的金额关系均一致", ~d["flag_either_endpoint_mismatch"]),
        ("期初权益至少1000（示例）", d["previous_equity"].ge(PARAMETERS["illustrative_min_start_equity"])),
        ("净现金流不超过期初权益10%（示例）", ~d["flag_large_net_flow"]),
    ]
    mask = pd.Series(True, index=d.index)
    waterfall_rows = []
    for label, condition in stages:
        mask &= condition
        waterfall_rows.append({"stage": label, "intervals": int(mask.sum()), "users": int(d.loc[mask, "user"].nunique())})
    d["research_consistent_interval"] = mask
    # Do not bridge gaps or concatenate filtered returns into a full NAV.
    previous_valid = d.groupby("user")["research_consistent_interval"].shift(fill_value=False)
    d["research_segment_start"] = mask & ~previous_valid
    d["research_segment_id"] = d.groupby("user")["research_segment_start"].cumsum().where(mask)

    addresses = d.groupby("user").agg(
        first_date=("date", "min"), last_date=("date", "max"), observations=("date", "size"),
        zero_equity_rows=("flag_zero_end_equity", "sum"), level_mismatch_rows=("flag_level_mismatch", "sum"),
        profit_mismatch_intervals=("flag_profit_mismatch", "sum"), gaps=("flag_date_gap", "sum"),
        consistent_intervals=("research_consistent_interval", "sum"),
        net_deposit_changes=("delta_net_deposit", lambda x: int(x.abs().gt(tol).sum())),
    )
    addresses["calendar_span_days"] = (addresses.last_date - addresses.first_date).dt.days + 1
    addresses["missing_days_inside_span"] = addresses.calendar_span_days - addresses.observations
    addresses["observation_coverage_inside_span"] = addresses.observations / addresses.calendar_span_days
    addresses["observed_at_cutoff"] = addresses.last_date.eq(d.date.max())
    addresses["consistent_share_of_all_intervals"] = addresses.consistent_intervals / (addresses.observations - 1).replace(0, np.nan)
    runs = d.loc[mask].groupby(["user", "research_segment_id"]).size()
    addresses["longest_consistent_run_intervals"] = runs.groupby(level=0).max().reindex(addresses.index).fillna(0).astype(int)
    addresses["zero_only_before_first_positive"] = (~(d.equity.eq(0) & d.groupby("user").equity.cummax().gt(0))).groupby(d.user).all()

    daily = d.groupby("date").agg(observations=("user", "size"), zero_equity_rows=("flag_zero_end_equity", "sum"), level_mismatch_rows=("flag_level_mismatch", "sum"), consistent_intervals=("research_consistent_interval", "sum"))
    daily["mismatch_fraction"] = daily.level_mismatch_rows / daily.observations
    daily["observation_count_change"] = daily.observations.diff()
    daily["snapshot_hour_p10"] = d.assign(hour=d.timestamp.dt.hour + d.timestamp.dt.minute / 60).groupby("date").hour.quantile(.1)
    daily["snapshot_hour_p90"] = d.assign(hour=d.timestamp.dt.hour + d.timestamp.dt.minute / 60).groupby("date").hour.quantile(.9)
    calendar = pd.date_range(d.date.min(), d.date.max())
    missing_global_dates = calendar.difference(d.date.unique())
    zero = d.equity.eq(0)
    pair = d.previous_date.notna()
    daypair = d.gap_days.eq(1) & d.previous_equity.gt(0)
    noflow = d.delta_net_deposit.abs().le(tol)
    varying_residual = d.groupby("user").residual.agg(lambda x: x.max() - x.min()).gt(tol)
    summary = {
        "source_file": SOURCE.name, "sha256": source_sha256,
        "rows": len(d), "users": len(addresses), "columns": list(raw.columns),
        "date_min": str(d.date.min().date()), "date_max": str(d.date.max().date()),
        "calendar_days": len(calendar), "distinct_dates": int(d.date.nunique()),
        "globally_missing_dates": [str(x.date()) for x in missing_global_dates],
        "null_cells": int(raw.isna().sum().sum()), "duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_user_date": int(raw.duplicated(["user", "date"]).sum()),
        "invalid_address_rows": int((~raw.user.str.fullmatch(r"0x[0-9a-fA-F]{40}")).sum()),
        "user_equals_trader": bool(raw.user.eq(raw.trader).all()),
        "equity_account_value_max_abs_difference": float((raw.equity - raw.cross_margin_summary_account_value).abs().max()),
        "date_timestamp_mismatch": int(d.date.ne(d.timestamp.dt.normalize()).sum()),
        "nonpositive_timestamp_steps": int(d.interval_hours.le(0).sum()),
        "timestamp_timezone": str(d.timestamp.dt.tz),
        "observed_days_median": float(addresses.observations.median()),
        "observed_days_q25": float(addresses.observations.quantile(.25)),
        "users_at_cutoff": int(addresses.observed_at_cutoff.sum()),
        "users_with_gaps": int(addresses.gaps.gt(0).sum()), "gap_intervals": int(d.flag_date_gap.sum()),
        "missing_days_inside_spans": int(addresses.missing_days_inside_span.sum()),
        "largest_gap_days": int(d.gap_days.max()),
        "shared_gap_nov22_to_nov28_users": int((d.previous_date.eq(pd.Timestamp("2025-11-22")) & d.date.eq(pd.Timestamp("2025-11-28"))).sum()),
        "zero_rows": int(zero.sum()), "zero_users": int(d.loc[zero, "user"].nunique()),
        "zero_rows_before_first_positive": int((zero & d.groupby("user").equity.cummax().eq(0)).sum()),
        "positive_to_zero_transitions": int((zero & d.previous_equity.gt(0)).sum()),
        "zero_to_positive_transitions": int((d.equity.gt(0) & d.previous_equity.eq(0)).sum()),
        "zero_rows_with_nonzero_pnl": int((zero & d.pnl.abs().gt(tol)).sum()),
        "level_mismatch_rows": int(d.flag_level_mismatch.sum()),
        "level_mismatch_users": int(addresses.level_mismatch_rows.gt(0).sum()),
        "negative_residual_rows": int(d.residual.lt(-tol).sum()), "positive_residual_rows": int(d.residual.gt(tol).sum()),
        "changing_residual_users": int(varying_residual.sum()),
        "intervals": int(pair.sum()), "profit_mismatch_intervals": int(d.flag_profit_mismatch.sum()),
        "constant_net_deposit_intervals": int((pair & noflow).sum()),
        "positive_net_deposit_changes": int(d.delta_net_deposit.gt(tol).sum()),
        "negative_net_deposit_changes": int(d.delta_net_deposit.lt(-tol).sum()),
        "negative_net_deposit_rows": int(d.net_deposit.lt(0).sum()),
        "cumulative_deposit_hypothesis_consistent_intervals": int((d.profit_equity_flow_hypothesis - d.delta_pnl).abs().le(tol).sum()),
        "period_deposit_hypothesis_consistent_intervals": int((d.delta_equity - d.net_deposit - d.delta_pnl).abs().le(tol).sum()),
        "daily_intervals": int(d.gap_days.eq(1).sum()),
        "irregular_daily_intervals": int(d.flag_irregular_daily_interval.sum()),
        "daily_interval_hours_min": float(d.loc[d.gap_days.eq(1), "interval_hours"].min()),
        "daily_interval_hours_max": float(d.loc[d.gap_days.eq(1), "interval_hours"].max()),
        "usable_research_intervals": int(mask.sum()), "users_with_research_intervals": int(d.loc[mask, "user"].nunique()),
        "daily_positive_start_intervals": int(daypair.sum()),
        "naive_vs_flow_return_difference_gt_1pp": int((daypair & (d.return_naive_equity - d.return_flow_end).abs().gt(.01)).sum()),
        "flow_vs_pnl_return_difference_gt_1pp": int((daypair & (d.return_flow_end - d.return_pnl_change).abs().gt(.01)).sum()),
        "timing_spread_gt_1pp_daily_positive_start": int((daypair & d.timing_scenario_spread.gt(.01)).sum()),
        "net_flow_zero_but_profit_mismatch_intervals": int((pair & noflow & d.flag_profit_mismatch).sum()),
        "negative_or_zero_start_scenario_denominator_daily_positive_start": int((daypair & d.denominator_start.le(0)).sum()),
    }
    coverage = pd.DataFrame([{
        "minimum_observations": n,
        "users_with_observations": int(addresses.observations.ge(n).sum()),
        "users_with_consistent_intervals": int(addresses.consistent_intervals.ge(n).sum()),
        "users_with_consecutive_consistent_intervals": int(addresses.longest_consistent_run_intervals.ge(n).sum()),
    } for n in [30, 60, 90, 120, 150]])
    sensitivity = []
    structural = d.gap_days.eq(1) & d.previous_equity.gt(0) & ~d.flag_either_endpoint_mismatch & d.interval_hours.between(*PARAMETERS["illustrative_interval_hours"])
    for floor in [0, 100, 1000, 10000]:
        for flow_limit in [0, .1, .25, 1.0, np.inf]:
            fmask = noflow if flow_limit == 0 else d.net_flow_ratio.le(flow_limit)
            selected = structural & d.previous_equity.ge(floor) & fmask
            n = d.loc[selected].groupby("user").size()
            sensitivity.append({"min_start_equity": floor, "max_net_flow_ratio": "zero_within_tolerance" if flow_limit == 0 else str(flow_limit), "intervals": int(selected.sum()), "users": int(d.loc[selected, "user"].nunique()), "users_with_at_least_60_intervals": int(n.ge(60).sum())})
    return_diagnostics = []
    for cohort, cohort_mask in [("日期连续且期初权益为正", daypair), ("条件一致区间（示例阈值）", mask)]:
        for col in ["return_naive_equity", "return_flow_end", "return_flow_mid", "return_flow_start", "return_pnl_change"]:
            s = d.loc[cohort_mask, col].replace([np.inf, -np.inf], np.nan).dropna()
            return_diagnostics.append({"cohort": cohort, "method": col, "finite_intervals": len(s), "p01": s.quantile(.01), "median": s.median(), "p99": s.quantile(.99), "minimum": s.min(), "maximum": s.max(), "below_minus_100pct": int(s.lt(-1).sum()), "above_100pct": int(s.gt(1).sum())})
    # Case examples are selected for audit discrepancies, never for investment merit.
    cases_mask = daypair & d.previous_equity.ge(1000) & noflow
    case_columns = ["source_row", "user", "previous_date", "date", "previous_equity", "equity", "delta_net_deposit", "delta_pnl", "profit_equity_flow_hypothesis", "previous_residual", "residual", "return_flow_end", "return_pnl_change"]
    examples = d.loc[cases_mask].assign(impact=lambda x: (x.return_flow_end - x.return_pnl_change).abs()).nlargest(8, "impact")[case_columns]
    flags = [c for c in d if c.startswith("flag_")]
    flag_counts = pd.DataFrame([{"flag": c, "rows": int(d[c].sum()), "users": int(d.loc[d[c], "user"].nunique())} for c in flags])
    gap_clusters = d.loc[d.flag_date_gap].groupby(["previous_date", "date", "gap_days"]).size().rename("users").reset_index().sort_values("users", ascending=False)
    tables = {
        "address_quality": addresses.reset_index(), "daily_quality": daily.reset_index(),
        "research_interval_waterfall": pd.DataFrame(waterfall_rows), "coverage_thresholds": coverage,
        "threshold_sensitivity": pd.DataFrame(sensitivity), "return_diagnostics": pd.DataFrame(return_diagnostics),
        "discrepancy_examples": examples, "quality_flags": flag_counts,
        "shared_gap_clusters": gap_clusters,
    }
    if digest(SOURCE) != source_sha256:
        raise RuntimeError('Source changed during audit; no output may be published')
    verify_frozen_source(source_sha256)
    for name, table in tables.items():
        table.to_csv(OUTPUT / (name + ".csv"), index=False, encoding="utf-8-sig")
    d.to_csv(OUTPUT / "audited_intervals.csv.gz", index=False, compression="gzip")
    manifest = {"summary": summary, "parameters": PARAMETERS, "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__}, "limitations": ["字段定义、货币单位、时区和数据生成过程未获原始说明确认。", "阈值是审计情景，不是最终地址选择规则。", "掩码仅标记可进一步研究的观测区间；禁止跳过无效区间拼接完整净值。", "无持仓或成交数据，不能计算实际跟单收益。"]}
    (OUTPUT / "audit_summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    # Independent algebraic reconciliation; no clipping or winsorization.
    algebra_error = (d.profit_equity_flow_hypothesis - d.profit_pnl_hypothesis - d.delta_residual).abs().max()
    if algebra_error > 1e-5:
        raise AssertionError(f"Increment identity error: {algebra_error}")
    if len(d) != summary["users"] + summary["intervals"]:
        raise AssertionError("First observations plus intervals do not cover input")
    provenance = {'version': 1, 'source_sha256': source_sha256,
                  'cache_file': 'audited_intervals.csv.gz',
                  'cache_sha256': digest(OUTPUT / 'audited_intervals.csv.gz'),
                  'summary_file': 'audit_summary.json',
                  'summary_sha256': digest(OUTPUT / 'audit_summary.json')}
    (OUTPUT / 'audit_cache_provenance.json').write_text(
        json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    return d, tables, manifest


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    _, _, manifest = run_audit()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
