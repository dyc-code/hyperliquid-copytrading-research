"""Semantic checks and an independent reconstruction from the original CSV."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import numpy as np
import pandas as pd
import selection_experiment as exp

ROOT, OUT = exp.ROOT, exp.OUT


def validate():
    checks = []
    def check(name, passed, detail=""):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        if not passed:
            raise AssertionError(name + ": " + str(detail))

    cfg = exp.CONFIG
    d = exp.load_data()
    protocol = json.loads((OUT / "protocol_record.json").read_text(encoding="utf-8"))
    check("原始数据及冻结参数哈希一致", hashlib.sha256(exp.SOURCE.read_bytes()).hexdigest() == protocol["source_sha256"] and hashlib.sha256(exp.CONFIG_FILE.read_bytes()).hexdigest() == protocol["configuration_sha256"])
    memberships = pd.read_csv(OUT / "historical_memberships.csv", parse_dates=["cutoff"])
    folds = pd.read_csv(OUT / "forward_folds.csv", parse_dates=["cutoff", "test_start", "test_end"])
    daily = pd.read_csv(OUT / "forward_daily.csv", parse_dates=["cutoff", "date"])
    chosen = pd.read_csv(OUT / "final_candidate_pool.csv")

    # Same decisions with all future data removed, for every scheduled cutoff.
    unchanged = True
    for cutoff in folds.cutoff.drop_duplicates():
        f, h = exp.training_features(d.loc[d.date.le(cutoff)], cutoff, cfg)
        members, _ = exp.strategy_members(f, h, cfg)
        for strategy, weights in members.items():
            saved = memberships.loc[memberships.cutoff.eq(cutoff) & memberships.strategy.eq(strategy)]
            unchanged &= set(saved.user) == set(weights)
            unchanged &= all(abs(row.weight - weights[row.user]) < 1e-12 for row in saved.itertuples())
    check("八轮全部名单仅使用过去资料", unchanged)
    weight_sums = memberships.groupby(["cutoff", "strategy"]).weight.sum().unstack()
    check("权重不超预算且对照预算一致", (weight_sums <= 1 + 1e-12).all().all() and np.allclose(weight_sums.robust_v1, weight_sums.profit_matched_budget) and np.allclose(weight_sums.robust_v1, weight_sums.eligible_equal_matched_budget))
    schedule = folds.loc[folds.strategy.eq("robust_v1")].sort_values("cutoff")
    all_dates = [date for q in schedule.itertuples() for date in pd.date_range(q.test_start, q.test_end)]
    check("八轮测试日期不重叠且共112天", len(all_dates) == 112 and len(set(all_dates)) == 112)
    check("缺失日及缺失轮次没有被填零", daily.loc[~daily.complete_day, "portfolio_return_mid"].isna().all() and folds.loc[~folds.complete_fold, "fold_return"].isna().all())

    # Independent raw reconstruction: does not use audit-derived returns or helpers.
    raw = pd.read_csv(exp.SOURCE, parse_dates=["date"])
    raw["timestamp"] = pd.to_datetime(raw.timestamp, format="mixed")
    raw = raw.sort_values(["user", "date"])
    lag = raw.groupby("user")[["equity", "net_deposit", "pnl", "date", "timestamp"]].shift()
    flow = raw.net_deposit - lag.net_deposit
    raw["independent_mid"] = (raw.equity - lag.equity - flow) / (lag.equity + .5 * flow)
    residual = raw.equity - raw.net_deposit - raw.pnl
    prev_residual = lag.equity - lag.net_deposit - lag.pnl
    raw["independent_valid"] = ((raw.date-lag.date).dt.days.eq(1) & lag.equity.gt(0) & residual.abs().le(.01) & prev_residual.abs().le(.01) & np.isfinite(raw.independent_mid) & raw.independent_mid.ge(-1) & (lag.equity+flow).gt(0))
    lookup = raw.set_index(["user", "date"])
    errors = []
    valid_mismatch = 0
    for q in folds.loc[folds.strategy.isin(["robust_v1", "profit_top10"])].itertuples():
        m = memberships.loc[memberships.cutoff.eq(q.cutoff) & memberships.strategy.eq(q.strategy)]
        expected_returns = []
        for date in pd.date_range(q.test_start, q.test_end):
            value, complete = 0., True
            for member in m.itertuples():
                if (member.user, date) not in lookup.index:
                    complete = False
                    continue
                row = lookup.loc[(member.user, date)]
                if not row.independent_valid:
                    complete = False
                else:
                    value += member.weight * row.independent_mid
            saved = daily.loc[daily.cutoff.eq(q.cutoff) & daily.strategy.eq(q.strategy) & daily.date.eq(date)].iloc[0]
            valid_mismatch += bool(saved.complete_day) != complete
            if complete:
                errors.append(abs(value - saved.portfolio_return_mid))
            expected_returns.append(value if complete else np.nan)
        if np.isfinite(expected_returns).all():
            nav = np.r_[1., np.cumprod(1+np.array(expected_returns))]
            errors += [abs((nav[-1]-1)-q.fold_return), abs((1-nav/np.maximum.accumulate(nav)).max()-q.fold_max_drawdown)]
    check("原始CSV独立重算主组和收益榜的每日/整轮收益", valid_mismatch == 0 and max(errors) < 1e-9, {"max_absolute_error": max(errors), "validity_mismatches": valid_mismatch})

    # Independent final training return/drawdown and correlation checks.
    last = raw.date.max()
    series = {}
    final_errors = []
    for q in chosen.itertuples():
        r = raw.loc[raw.user.eq(q.user) & raw.date.between(last-pd.Timedelta(days=59), last), "independent_mid"].to_numpy()
        nav = np.r_[1., np.cumprod(1+r)]
        series[q.user] = r
        final_errors += [abs(nav[-1]-1-q.training_return), abs((1-nav/np.maximum.accumulate(nav)).max()-q.max_drawdown)]
    corr = pd.DataFrame(series).corr().to_numpy()
    check("最终名单收益回撤独立重算且正相关不超过0.7", max(final_errors) < 1e-9 and corr[np.triu_indices(len(chosen),1)].max() <= .7+1e-12, {"max_metric_error": max(final_errors), "max_pairwise_correlation": float(corr[np.triu_indices(len(chosen),1)].max())})

    # Real missing observation: original weight remains unknown, no replacement.
    cutoff = schedule.iloc[0].cutoff
    m = memberships.loc[memberships.cutoff.eq(cutoff) & memberships.strategy.eq("robust_v1")]
    weights = dict(zip(m.user, m.weight))
    member = m.iloc[0].user
    first_day = cutoff + pd.Timedelta(days=1)
    damaged = d.loc[~(d.user.eq(member) & d.date.eq(first_day))]
    fold, days, _ = exp.forward_evaluation(damaged, cutoff, cutoff+pd.Timedelta(days=14), weights, cfg, "check")
    check("人为移除未来一行会保留未知权重并使整轮未知", not fold["complete_fold"] and pd.isna(fold["fold_return"]) and abs(days.iloc[0].unknown_weight-weights[member]) < 1e-12 and pd.isna(days.iloc[0].portfolio_return_mid))

    # Validated extremes must remain; primary and timing scenarios differ.
    synthetic = pd.DataFrame([{"user":"test", "date":first_day, "gap_days":1, "previous_equity":1000., "equity":0., "residual":0., "previous_residual":0., "delta_net_deposit":0., "net_flow_ratio":0., "interval_hours":24., "return_flow_mid":-1., "return_flow_end":-1., "return_flow_start":-1.}])
    fold, _, _ = exp.forward_evaluation(synthetic, cutoff, first_day, {"test":1.}, cfg, "check")
    check("真实100%亏损保留并计算为100%回撤", fold["complete_fold"] and fold["fold_return"] == -1 and fold["fold_max_drawdown"] == 1)
    synthetic.loc[0,["return_flow_mid","return_flow_end","return_flow_start","net_flow_ratio","delta_net_deposit"]] = [-.8,-1.2,-.6,1.,1000.]
    synthetic.loc[0,"equity"] = 800.  # G=800-1000-1000=-1200; midpoint denominator=1500.
    fold, days, _ = exp.forward_evaluation(synthetic, cutoff, first_day, {"test":1.}, cfg, "check")
    check("未来大额流量和大亏不被训练门槛删除，失效时点情景独立标未知", fold["complete_fold"] and abs(fold["fold_return"]+.8)<1e-12 and pd.isna(fold["fold_return_end_flow"]) and days.iloc[0].unknown_weight_end_flow == 1.)

    cases = pd.read_csv(OUT / "forward_address_outcomes.csv")
    worst = cases.loc[cases.strategy.eq("robust_v1") & cases.complete].sort_values("forward_return").iloc[0]
    train = pd.read_csv(OUT / "historical_training_features.csv")
    tr = train.loc[train.user.eq(worst.user) & train.cutoff.eq(worst.cutoff)].iloc[0]
    case = {"user":worst.user, "cutoff":worst.cutoff,"training_return":float(tr.training_return), "training_max_drawdown":float(tr.max_drawdown),"forward_return":float(worst.forward_return),"observable_days":int(worst.observable_days),"cashflow_sensitive_days":int(worst.cashflow_sensitive_days),"interpretation":"账户条件收益代理案例，不能推断交易原因或真实跟单损益。"}
    report = {"passed":all(c["passed"] for c in checks), "checks":checks, "historical_failure_example":case, "method_repair":"时点情景收益低于-100%时仅将该情景标未知；主收益和已冻结选人规则不变。"}
    (OUT/"validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(validate(),ensure_ascii=False,indent=2))
