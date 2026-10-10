# 数据来源与复现边界

## 来源

输入是外部提供的地址级账户日快照 CSV，不是作者从交易所独立抓取的数据。数据提供方未附地址抽样规则、字段字典、货币单位或原始时区说明。当前也没有完整持仓、逐笔成交、盘口或订单状态。原始数据及地址级明细不随公开仓库分发。

原始文件 SHA-256：`cb2a16931470cecaeb16a35af150788e53ea2762963edad98b44902ff8d00705`。该值用于核对版本，不授予对原始数据的访问。

## 输入与运行

若已合法取得相同版本的原始 CSV，将其放在 `data/account_daily.csv`。脚本至少使用 `user`、`date`、`timestamp`、`equity`、`net_deposit`、`pnl` 六列。`user` 是地址标识，`date` 是账户日期，其余金额字段按原文件数值读取。`net_deposit` 暂按累计净入金解释；该解释尚待数据提供方确认。

从仓库根目录，在 Python 3.12 环境中安装 `requirements.txt`，按顺序运行：

```text
python -m pip install -r requirements.txt
python analysis/audit_data.py
python analysis/selection_experiment.py
python analysis/validate_selection.py
```

审计结果写入 `output/audit/`，选人结果写入 `output/selection/`。原始协议记录位于公开的 `results/protocol_record.json`；如需验证本次历史实验的配置与来源绑定，先将它复制到 `output/selection/protocol_record.json`。脚本在数据或配置哈希不符时拒绝将新结果写成原实验。新数据应作为新实验单独保存。

## 可核对与不可核对

无需原始数据即可检查报告与 `results/` 中的聚合数字、八轮覆盖及配对差值，也可阅读完整实验逻辑。仅有公开仓库无法独立复算地址级输入或验证数据提供过程。代码中的地址级中间产物会在持有原始数据并重跑后于本地生成，不属于公开成果。
