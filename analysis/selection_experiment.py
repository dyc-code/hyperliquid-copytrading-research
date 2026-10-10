"""Past-only selection and forward evaluation of conditional account returns.

Inputs: previous audit with source-row traceability. Unknown outcomes are NA.
No execution, actual capital allocation or copying orders takes place.
"""
from pathlib import Path
import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
import sys

import numpy as np
import pandas as pd
from source_binding import verify_source_binding

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "selection"
CONFIG_FILE = ROOT / "analysis" / "selection_config.json"
CONFIG = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
AUDIT_FILE = ROOT / "output" / "audit" / "audited_intervals.csv.gz"
SOURCE = ROOT / "data" / "account_daily.csv"


def verify_inputs():
    return verify_source_binding(SOURCE, AUDIT_FILE,
                                 AUDIT_FILE.with_name('audit_summary.json'),
                                 CONFIG_FILE, OUT / 'protocol_record.json')


def load_data():
    verify_inputs()
    cols = ["user", "date", "source_row", "previous_date", "gap_days", "interval_hours", "equity", "previous_equity", "residual", "previous_residual", "delta_net_deposit", "net_flow_ratio", "return_flow_end", "return_flow_mid", "return_flow_start"]
    return pd.read_csv(AUDIT_FILE, usecols=cols, parse_dates=["date", "previous_date"]).sort_values(["user", "date"]).reset_index(drop=True)


def source_valid_mask(d, cfg):
    """Does not censor large losses or high net flows in forward outcomes."""
    return (
        d.gap_days.eq(1) & d.previous_equity.gt(0)
        & d.residual.abs().le(cfg["reconciliation_tolerance"])
        & d.previous_residual.abs().le(cfg["reconciliation_tolerance"])
        & np.isfinite(d.return_flow_mid) & d.return_flow_mid.ge(-1)
        & (d.previous_equity + d.delta_net_deposit).gt(0)
    )


def metrics_of_returns(values, cfg):
    r = np.asarray(values, dtype=float)
    if not np.isfinite(r).all() or (r < -1).any():
        return None
    with np.errstate(divide="ignore"):
        log_r = np.log1p(r)
    nav = np.r_[1.0, np.cumprod(1 + r)]
    block_returns = [float(np.expm1(x.sum())) for x in np.array_split(log_r, 3)]
    positive = np.maximum(log_r, 0)
    best = np.argsort(log_r)[-3:]
    without_best = log_r.copy()
    without_best[best] = 0  # Counterfactual diagnostic, never used to create test NAV.
    downside = float(np.sqrt(np.mean(np.minimum(log_r, 0) ** 2)))
    return {
        "training_return": float(nav[-1] - 1),
        "max_drawdown": float((1 - nav / np.maximum.accumulate(nav)).max()),
        "worst_day": float(r.min()),
        "daily_volatility": float(r.std(ddof=1)),
        "downside_log_rms": downside,
        "score": float(log_r.mean() / max(downside, cfg["daily_downside_floor"])) if np.isfinite(log_r).all() else -np.inf,
        "profitable_blocks": int(sum(v > 0 for v in block_returns)),
        "block_1_return": block_returns[0], "block_2_return": block_returns[1], "block_3_return": block_returns[2],
        "nonflat_days": int((np.abs(r) > cfg["nonflat_return_threshold"]).sum()),
        "top3_positive_log_share": float(np.sort(positive)[-3:].sum() / positive.sum()) if positive.sum() > 0 else 1.0,
        "return_without_best3_days": float(np.expm1(without_best.sum())),
    }


def training_features(d, cutoff, cfg):
    cutoff = pd.Timestamp(cutoff)
    first_end = cutoff - pd.Timedelta(days=cfg["lookback_days"] - 1)
    x = d.loc[d.date.between(first_end, cutoff)].copy()
    required_dates = pd.date_range(first_end, cutoff)
    x["valid_structural"] = source_valid_mask(x, cfg)
    x["valid_timing"] = x.interval_hours.between(cfg["snapshot_hours_min"], cfg["snapshot_hours_max"])
    x["valid_equity_size"] = x.previous_equity.ge(cfg["min_start_equity"])
    x["valid_cashflow"] = x.net_flow_ratio.le(cfg["max_training_net_flow_ratio"])
    x["bad_structural"] = ~x.valid_structural
    x["bad_timing"] = ~x.valid_timing
    x["bad_size"] = ~x.valid_equity_size
    x["bad_cashflow"] = ~x.valid_cashflow
    f = x.groupby("user", sort=True).agg(
        observed_intervals=("date", "size"),
        structural_invalid_intervals=("bad_structural", "sum"),
        irregular_timing_intervals=("bad_timing", "sum"),
        small_equity_intervals=("bad_size", "sum"),
        large_net_flow_intervals=("bad_cashflow", "sum"),
        last_observation=("date", "max"), last_equity=("equity", "last"),
    ).reset_index()
    if not len(f):
        raise ValueError("No training records at cutoff")
    f["cutoff"] = cutoff
    f["training_start_snapshot"] = first_end - pd.Timedelta(days=1)
    f["required_intervals"] = len(required_dates)
    f["missing_intervals"] = len(required_dates) - f.observed_intervals
    f.loc[f.last_observation.ne(cutoff), "last_equity"] = np.nan
    f["data_eligible"] = f.missing_intervals.eq(0) & f[["structural_invalid_intervals", "irregular_timing_intervals", "small_equity_intervals", "large_net_flow_intervals"]].eq(0).all(axis=1)
    issue_columns = {
        "missing_intervals": "研究窗口有缺失", "structural_invalid_intervals": "收益口径或分母待核实",
        "irregular_timing_intervals": "日快照间隔不规则", "small_equity_intervals": "期初权益小于示例下限",
        "large_net_flow_intervals": "净入金变化超过示例比例",
    }
    f["data_issues"] = f.apply(lambda row: "；".join(text for col, text in issue_columns.items() if row[col] > 0), axis=1)
    histories = {}
    metric_rows = []
    eligible_users = set(f.loc[f.data_eligible, "user"])
    for user, q in x.loc[x.user.isin(eligible_users)].groupby("user", sort=True):
        q = q.set_index("date").reindex(required_dates)
        vals = q.return_flow_mid.to_numpy(dtype=float)
        metrics = metrics_of_returns(vals, cfg)
        if metrics is None:
            raise AssertionError("Structurally eligible training history has invalid returns")
        metric_rows.append({"user": user, **metrics})
        histories[user] = vals
    if metric_rows:
        f = f.merge(pd.DataFrame(metric_rows), on="user", how="left", validate="one_to_one")
    for column in ["training_return", "max_drawdown", "worst_day", "profitable_blocks", "nonflat_days", "top3_positive_log_share", "return_without_best3_days", "score"]:
        if column not in f:
            f[column] = np.nan
    checks = {
        "positive_total": f.training_return.gt(0),
        "drawdown_limit": f.max_drawdown.le(cfg["max_drawdown"]),
        "daily_loss_limit": f.worst_day.ge(-cfg["max_single_day_loss"]),
        "profitable_blocks_limit": f.profitable_blocks.ge(cfg["minimum_profitable_blocks"]),
        "nonflat_days_limit": f.nonflat_days.ge(cfg["minimum_nonflat_days"]),
        "profit_concentration_limit": f.top3_positive_log_share.le(cfg["maximum_top3_positive_log_return_share"]),
        "remove_best3_limit": f.return_without_best3_days.gt(0),
    }
    for label, check in checks.items():
        f["pass_" + label] = check
    f["performance_eligible"] = f.data_eligible & pd.concat(checks.values(), axis=1).all(axis=1)
    descriptions = {
        "positive_total": "研究窗口整体未盈利", "drawdown_limit": "历史回撤超过门槛", "daily_loss_limit": "最差日亏损超过门槛",
        "profitable_blocks_limit": "盈利阶段不足", "nonflat_days_limit": "有效变动天数不足", "profit_concentration_limit": "盈利集中在少数日期", "remove_best3_limit": "拿掉最好三天后未盈利",
    }
    f["performance_issues"] = f.apply(lambda row: "；".join(text for name, text in descriptions.items() if not row["pass_" + name]) if row.data_eligible else "", axis=1)
    return f, histories


def diversified_selection(f, histories, cfg):
    candidates = f.loc[f.performance_eligible].sort_values(["score", "user"], ascending=[False, True])
    selected, blocked = [], {}
    for user in candidates.user:
        if len(selected) >= cfg["maximum_addresses"]:
            blocked[user] = "主池名额已满"
            continue
        correlations = [(other, float(np.corrcoef(histories[user], histories[other])[0, 1])) for other in selected]
        too_similar = [(other, c) for other, c in correlations if np.isfinite(c) and c > cfg["maximum_pairwise_positive_correlation"]]
        if too_similar:
            other, c = max(too_similar, key=lambda item: item[1])
            blocked[user] = f"与已选{other}的历史收益相关系数{c:.3f}超过上限"
        else:
            selected.append(user)
    return selected, blocked


def strategy_members(f, histories, cfg):
    selected, blocked = diversified_selection(f, histories, cfg)
    profit_rank = f.loc[f.data_eligible].sort_values(["training_return", "user"], ascending=[False, True]).user.tolist()
    robust_rank = f.loc[f.performance_eligible].sort_values(["score", "user"], ascending=[False, True]).user.tolist()
    target = cfg["maximum_addresses"]
    k = len(selected)
    weight = cfg["capital_weight_per_address"]
    all_eligible = sorted(f.loc[f.data_eligible, "user"].tolist())
    members = {
        "robust_v1": {u: weight for u in selected},
        "profit_top10": {u: weight for u in profit_rank[:target]},
        "profit_matched_budget": {u: weight for u in profit_rank[:k]},
        "eligible_equal_matched_budget": {u: k * weight / len(all_eligible) for u in all_eligible} if all_eligible else {},
        "robust_without_correlation": {u: weight for u in robust_rank[:target]},
    }
    return members, blocked


def forward_evaluation(d, cutoff, end, weights, cfg, name):
    dates = pd.date_range(pd.Timestamp(cutoff) + pd.Timedelta(days=1), pd.Timestamp(end))
    rows = []
    individuals = []
    matrices = {}
    masks = {}
    weight_total = sum(weights.values())
    for user, weight in weights.items():
        q = d.loc[d.user.eq(user) & d.date.isin(dates)].set_index("date").reindex(dates)
        q["user"] = user
        valid = source_valid_mask(q, cfg).fillna(False)
        # Source validity includes observed fields: NaN comparisons are false.
        vals = q.return_flow_mid.where(valid)
        matrices[user] = q[["return_flow_end", "return_flow_mid", "return_flow_start"]].where(valid, np.nan)
        # A timing scenario may imply a return below -100% even when the
        # primary midpoint scenario is interpretable. Do not compound such a
        # scenario or discard the primary outcome: mark that scenario unknown.
        for label in ["end", "start"]:
            col = "return_flow_" + label
            matrices[user][col] = matrices[user][col].where(np.isfinite(matrices[user][col]) & matrices[user][col].ge(-1))
        masks[user] = valid
        complete = bool(valid.all())
        individuals.append({
            "strategy": name, "cutoff": pd.Timestamp(cutoff), "test_end": pd.Timestamp(end), "user": user, "weight": weight,
            "test_days": len(dates), "observable_days": int(valid.sum()), "complete": complete,
            "forward_return": float(np.prod(1 + vals.to_numpy()) - 1) if complete else np.nan,
            "cashflow_sensitive_days": int((valid & q.net_flow_ratio.gt(cfg["max_training_net_flow_ratio"])).sum()),
            "snapshot_irregular_days": int((valid & ~q.interval_hours.between(cfg["snapshot_hours_min"], cfg["snapshot_hours_max"])).sum()),
        })
    for date in dates:
        unknown_weight = sum(w for u, w in weights.items() if not bool(masks[u].loc[date]))
        known = {label: sum(w * float(matrices[u].loc[date, "return_flow_" + label]) for u, w in weights.items() if bool(masks[u].loc[date])) for label in ["end", "mid", "start"]}
        scenario_unknown = {label: sum(w for u, w in weights.items() if not np.isfinite(matrices[u].loc[date, "return_flow_" + label])) for label in ["end", "start"]}
        complete = unknown_weight <= 1e-12
        rows.append({
            "strategy": name, "cutoff": pd.Timestamp(cutoff), "date": date,
            "invested_weight": weight_total, "cash_weight": 1 - weight_total,
            "selected_count": len(weights), "unknown_weight": unknown_weight,
            "unknown_weight_end_flow": scenario_unknown["end"], "unknown_weight_start_flow": scenario_unknown["start"],
            "complete_day": complete,
            # Known component is disclosed as a component, never labelled full return.
            "known_component_mid": known["mid"],
            "portfolio_return_mid": known["mid"] if complete else np.nan,
            "portfolio_return_end": known["end"] if scenario_unknown["end"] <= 1e-12 else np.nan,
            "portfolio_return_start": known["start"] if scenario_unknown["start"] <= 1e-12 else np.nan,
        })
    daily = pd.DataFrame(rows)
    full = bool(daily.complete_day.all())
    result = {
        "strategy": name, "cutoff": pd.Timestamp(cutoff), "test_start": dates.min(), "test_end": dates.max(),
        "test_days": len(dates), "selected_count": len(weights), "invested_weight": weight_total,
        "observable_days": int(daily.complete_day.sum()), "observable_day_share": float(daily.complete_day.mean()),
        "unknown_weight_day_sum": float(daily.unknown_weight.sum()), "complete_fold": full,
        "fold_return": np.nan, "fold_max_drawdown": np.nan, "fold_worst_day": np.nan,
        "fold_return_end_flow": np.nan, "fold_return_start_flow": np.nan,
    }
    if full:
        r = daily.portfolio_return_mid.to_numpy()
        nav = np.r_[1.0, np.cumprod(1 + r)]
        result.update(fold_return=float(nav[-1] - 1), fold_max_drawdown=float((1 - nav / np.maximum.accumulate(nav)).max()), fold_worst_day=float(r.min()))
        for label in ["end", "start"]:
            scenario = daily["portfolio_return_" + label]
            if scenario.notna().all():
                result["fold_return_" + label + "_flow"] = float(np.prod(1 + scenario.to_numpy()) - 1)
    return result, daily, pd.DataFrame(individuals)


def summarize_folds(folds):
    rows = []
    for strategy, q in folds.groupby("strategy"):
        full = q.loc[q.complete_fold]
        rows.append({
            "strategy": strategy, "scheduled_full_folds": len(q), "fully_observable_folds": len(full),
            "observable_days": int(q.observable_days.sum()), "scheduled_days": int(q.test_days.sum()),
            "mean_selected_count": float(q.selected_count.mean()), "mean_invested_weight": float(q.invested_weight.mean()),
            "mean_complete_fold_return": float(full.fold_return.mean()) if len(full) else np.nan,
            "median_complete_fold_return": float(full.fold_return.median()) if len(full) else np.nan,
            "positive_complete_folds": int(full.fold_return.gt(0).sum()),
            "worst_complete_fold_return": float(full.fold_return.min()) if len(full) else np.nan,
            "max_complete_fold_drawdown": float(full.fold_max_drawdown.max()) if len(full) else np.nan,
        })
    return pd.DataFrame(rows)


def paired_comparisons(folds):
    robust = folds.loc[folds.strategy.eq("robust_v1")].set_index("cutoff")
    rows = []
    paired_rows = []
    for baseline in ["profit_top10", "profit_matched_budget", "eligible_equal_matched_budget", "robust_without_correlation"]:
        base = folds.loc[folds.strategy.eq(baseline)].set_index("cutoff")
        joined = robust.join(base, lsuffix="_robust", rsuffix="_baseline", how="inner")
        comparable = joined.loc[joined.complete_fold_robust & joined.complete_fold_baseline]
        delta = comparable.fold_return_robust - comparable.fold_return_baseline
        dd_delta = comparable.fold_max_drawdown_robust - comparable.fold_max_drawdown_baseline
        for date, q in comparable.iterrows():
            paired_rows.append({"baseline": baseline, "cutoff": date, "robust_return": q.fold_return_robust, "baseline_return": q.fold_return_baseline, "return_difference": q.fold_return_robust-q.fold_return_baseline, "robust_drawdown": q.fold_max_drawdown_robust, "baseline_drawdown": q.fold_max_drawdown_baseline})
        rows.append({"baseline": baseline, "scheduled_folds": len(joined), "jointly_observable_folds": len(comparable), "mean_return_difference": float(delta.mean()) if len(delta) else np.nan, "robust_return_wins": int(delta.gt(0).sum()), "mean_drawdown_difference": float(dd_delta.mean()) if len(dd_delta) else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(paired_rows)


def cutoff_schedule(d, cfg):
    first = d.date.min() + pd.Timedelta(days=cfg["lookback_days"])
    last = d.date.max()
    dates = []
    while first + pd.Timedelta(days=cfg["evaluation_days"]) <= last:
        dates.append(first)
        first += pd.Timedelta(days=cfg["evaluation_days"])
    return dates, first, last


def verify_output_config(cfg):
    if cfg != json.loads(CONFIG_FILE.read_text(encoding="utf-8")):
        raise ValueError('Runtime configuration differs from the source experiment; use output=False or a separately named experiment')


def run_experiment(d=None, cfg=None, output=True, sensitivity=True):
    source_binding = verify_inputs() if d is None else None
    if d is not None and output:
        raise ValueError('In-memory research data cannot overwrite the frozen source experiment')
    cfg = deepcopy(CONFIG if cfg is None else cfg)
    if output:
        verify_output_config(cfg)
    d = load_data() if d is None else d
    cutoffs, partial_cutoff, last = cutoff_schedule(d, cfg)
    results, day_results, individual_results, selection_rows, training_rows = [], [], [], [], []
    for cutoff in cutoffs:
        f, histories = training_features(d, cutoff, cfg)
        training_rows.append(f)
        members, _ = strategy_members(f, histories, cfg)
        for strategy, weights in members.items():
            for user, w in weights.items():
                selection_rows.append({"strategy": strategy, "cutoff": cutoff, "user": user, "weight": w})
            fold, daily, individuals = forward_evaluation(d, cutoff, cutoff + pd.Timedelta(days=cfg["evaluation_days"]), weights, cfg, strategy)
            results.append(fold); day_results.append(daily)
            if len(individuals):
                individual_results.append(individuals)
    folds = pd.DataFrame(results)
    summary = summarize_folds(folds)
    paired_summary, paired_rows = paired_comparisons(folds)
    final_f, final_histories = training_features(d, last, cfg)
    final_members, blocked = strategy_members(final_f, final_histories, cfg)
    final_selected = list(final_members["robust_v1"])
    final_f["status"] = np.select([~final_f.data_eligible, ~final_f.performance_eligible, final_f.user.isin(final_selected)], ["观察：资料条件未满足", "观察：历史表现条件未满足", "历史截止日主候选"], default="候补")
    final_f["pool_reason"] = final_f.user.map(blocked).fillna("")
    final_f["experimental_weight"] = final_f.user.map(final_members["robust_v1"]).fillna(0)
    final_pool = final_f.loc[final_f.user.isin(final_selected)].copy()
    final_pool["selection_order"] = final_pool.user.map({u:i+1 for i,u in enumerate(final_selected)})
    final_pool = final_pool.sort_values("selection_order")
    correlation = pd.DataFrame({u: final_histories[u] for u in final_selected}).corr()
    final_rules = [{"stage": "当前60日窗口出现过记录", "users": len(final_f)}]
    cumulative = final_f.data_eligible.copy()
    final_rules.append({"stage": "60个区间全部通过资料门槛", "users": int(cumulative.sum())})
    stages = ["positive_total", "drawdown_limit", "daily_loss_limit", "profitable_blocks_limit", "nonflat_days_limit", "profit_concentration_limit", "remove_best3_limit"]
    for label in stages:
        cumulative &= final_f["pass_" + label]
        final_rules.append({"stage": label, "users": int(cumulative.sum())})
    final_rules.append({"stage":"排序并施加相关性与名额限制", "users": len(final_pool)})
    sensitivity_rows = []
    variants = {
        "原规则": {}, "回撤上限15%": {"max_drawdown":.15}, "回撤上限25%": {"max_drawdown":.25},
        "最差日上限8%": {"max_single_day_loss":.08}, "最差日上限12%": {"max_single_day_loss":.12},
        "相关性上限0.5": {"maximum_pairwise_positive_correlation":.5}, "相关性上限0.85": {"maximum_pairwise_positive_correlation":.85},
        "净入金比例上限25%": {"max_training_net_flow_ratio":.25}, "只选最多5人": {"maximum_addresses":5},
        "训练45天": {"lookback_days":45}, "训练90天": {"lookback_days":90},
    }
    if sensitivity:
        # The same predeclared calendar cutoffs are used wherever each variant
        # has enough past history; never promote the best variant after testing.
        for label, changes in variants.items():
            local = deepcopy(cfg); local.update(changes)
            local_results, local_base = [], []
            for cutoff in cutoffs:
                if cutoff - pd.Timedelta(days=local["lookback_days"]) < d.date.min():
                    continue
                f, h = training_features(d, cutoff, local)
                members, _ = strategy_members(f, h, local)
                end = cutoff + pd.Timedelta(days=local["evaluation_days"])
                a, _, _ = forward_evaluation(d, cutoff, end, members["robust_v1"], local, "robust_v1")
                b, _, _ = forward_evaluation(d, cutoff, end, members["profit_matched_budget"], local, "profit_matched_budget")
                local_results.append(a); local_base.append(b)
            q = pd.DataFrame(local_results); b = pd.DataFrame(local_base)
            full = q.loc[q.complete_fold]
            joint = q.complete_fold & b.complete_fold
            lf, lh = training_features(d, last, local)
            lm, _ = strategy_members(lf, lh, local)
            sensitivity_rows.append({"variant":label, "scheduled_folds":len(q), "complete_folds":len(full), "jointly_observable_matched_folds":int(joint.sum()), "mean_complete_return":float(full.fold_return.mean()) if len(full) else np.nan, "mean_matched_return_difference":float((q.loc[joint,'fold_return']-b.loc[joint,'fold_return']).mean()) if joint.any() else np.nan, "mean_selected_count":float(q.selected_count.mean()), "final_candidate_count":len(lm['robust_v1']), "overlap_with_primary_final_pool":len(set(lm['robust_v1']) & set(final_selected))})
    # The last six days are intentionally kept separate from the 14-day folds.
    partial_rows = []
    if partial_cutoff < last:
        pf, ph = training_features(d, partial_cutoff, cfg)
        pm, _ = strategy_members(pf, ph, cfg)
        for name, weights in pm.items():
            result, _, _ = forward_evaluation(d, partial_cutoff, last, weights, cfg, name)
            partial_rows.append(result)
    tables = {
        "forward_folds": folds, "strategy_summary": summary,
        "paired_comparisons": paired_summary, "paired_fold_details": paired_rows,
        "forward_daily": pd.concat(day_results, ignore_index=True),
        "forward_address_outcomes": pd.concat(individual_results, ignore_index=True),
        "historical_memberships": pd.DataFrame(selection_rows),
        "historical_training_features": pd.concat(training_rows, ignore_index=True),
        "final_address_assessment": final_f, "final_candidate_pool": final_pool,
        "final_candidate_correlations": correlation.reset_index(names="user"),
        "final_selection_steps": pd.DataFrame(final_rules),
        "parameter_sensitivity": pd.DataFrame(sensitivity_rows),
        "partial_period": pd.DataFrame(partial_rows),
    }
    result = {
        "config": cfg, "source_sha256": source_binding['source_sha256'] if source_binding else None,
        "configuration_sha256": hashlib.sha256(CONFIG_FILE.read_bytes()).hexdigest(),
        "date_range": [str(d.date.min().date()), str(last.date())],
        "evaluation_cutoffs": [str(x.date()) for x in cutoffs],
        "primary_forward_period": [str((cutoffs[0]+pd.Timedelta(days=1)).date()), str((cutoffs[-1]+pd.Timedelta(days=cfg['evaluation_days'])).date())],
        "partial_period": [str((partial_cutoff+pd.Timedelta(days=1)).date()), str(last.date())],
        "final_data_eligible_addresses": int(final_f.data_eligible.sum()),
        "final_performance_eligible_addresses": int(final_f.performance_eligible.sum()),
        "final_candidate_count": len(final_pool), "final_candidate_addresses": final_selected,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Original account-pool sampling may contain survivorship/selection bias.",
            "All proxy returns depend on unconfirmed cashflow and account-scope semantics.",
            "Snapshot timing is asynchronous and not a tradable portfolio valuation at a common instant.",
            "Daily fixed experimental weights imply synthetic rebalancing with no turnover/fees/slippage/funding modelling; this is not copy-trade PnL.",
            "Incomplete forward periods have unknown returns. Complete-fold averages are conditional on observability and cannot establish unconditional full-period performance.",
            "Rules were fixed before this strategy evaluation, but the source had already been structurally audited and past-window feasibility examined; not a truly blind prospective experiment.",
            "Parameter sensitivity variants are exploratory diagnostics. Their best-performing version is not promoted.",
            "Final candidate list is historical as of 2026-01-04, not verified for current live use."
        ],
    }
    if source_binding is not None and verify_inputs() != source_binding:
        raise RuntimeError('Inputs changed during selection; do not write results')
    if output:
        verify_output_config(cfg)
        OUT.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig")
        (OUT / "experiment_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return tables, result


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    verify_inputs()  # Fail before creating a protocol record or writing outputs.
    OUT.mkdir(parents=True, exist_ok=True)
    prereg = {
        "configuration_sha256": hashlib.sha256(CONFIG_FILE.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "protocol_recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "prior_exploration": "Previous source-wide structural audit plus train-only feasibility summaries at 2025-09-08, 2025-10-20, 2025-12-15, 2026-01-04; no forward strategy outcomes computed before this configuration.",
        "sensitivity_variants_predeclared": ["drawdown 15/25%", "worst day 8/12%", "correlation .5/.85", "flow 25%", "max addresses 5", "lookback 45/90 days"],
    }
    protocol_file = OUT / "protocol_record.json"
    if protocol_file.exists():
        previous = json.loads(protocol_file.read_text(encoding="utf-8"))
        if previous["configuration_sha256"] != prereg["configuration_sha256"]:
            raise RuntimeError("Frozen configuration changed; create a separately named experiment")
    else:
        protocol_file.write_text(json.dumps(prereg,ensure_ascii=False,indent=2),encoding="utf-8")
    tables, summary = run_experiment()
    print(tables["strategy_summary"].to_string(index=False))
    print(tables["paired_comparisons"].to_string(index=False))
    print(tables["final_selection_steps"].to_string(index=False))
    print(tables["parameter_sensitivity"].to_string(index=False))
