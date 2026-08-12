# 名古屋市営地下鉄 RFID/BIBO 運賃回収率モデル

各路線内で完結するODだけを対象に、駅間リンクごとの独立なRFID検出からODをベイズ推定し、推定ODの普通運賃を回収額として複数シード評価します。

## 実行

```bash
python3 -m pip install -r requirements.txt
python3 nagoya_bibo.py
```

軽い動作確認:

```bash
python3 nagoya_bibo.py --seed-count 3 --sample-size 200
```

本評価例:

```bash
python3 nagoya_bibo.py --seed-count 150 --sample-size 3300 --detection-rates 0.65 0.70
```

出力は `nagoya_bibo_outputs/` に作成されます。

- `line_summary.csv`: 路線別総回収率、SD、平均の95%CI
- `trip_length_summary.csv`: 利用区間数別回収率、SD、95%CI、平均人数
- `od_candidates.csv`: OD、営業距離 `d_ij`、リンク数 `m_ij`、真の運賃 `F_ij`、採用OD確率
- `*_recovery_by_trip_length.png`: 路線別グラフ
- `all_lines_overall_recovery.png`: 路線比較
- `run_config.json`: 実行条件

## ODモデル

### gravity（基準方式）

`pi_ij ∝ exp(-gamma * d_ij)`。ハブ補正なし。

### section_fit（PDF断面制約方式）

PDFの色分けされた断面混雑を相対強度に置き換え、重力モデルを事前形状として、経路-リンク接続行列 `A` に対して `A*pi` が断面強度へ近づくよう非負制約付きで推定します。正則化があるため、断面交通量だけではODが一意に定まらない問題を明示的に扱います。

`section_fit_rmse` は、推定断面交通量と入力相対強度の正規化RMSEです。

## 路線構造

- 東山線、鶴舞線、桜通線: 直線
- 名城線: 環状
- 名港線: 金山から分岐
- 乗換を伴う他路線ODは対象外
- 名城・名港線は一つのグラフとして評価

最短営業距離経路を採用します。同距離経路の扱いは実装上の決定順に依存するため、環状線の厳密な利用経路モデルではありません。

## 料金

名古屋市営地下鉄の対キロ区間制を実装しています。

| 営業距離 | 大人普通料金 |
|---|---:|
| 3 kmまで | 210円 |
| 7 kmまで | 240円 |
| 11 kmまで | 270円 |
| 15 kmまで | 310円 |
| 15 km超 | 340円 |

公式ページでは普通券とIC利用で異なる金額体系は示されていないため、本モデルの `F_ij` はこの普通料金を共通に使用します。

## データ根拠（取得日: 2026-08-06）

- 路線・駅順: https://www.kotsu.city.nagoya.jp/rp/subway/
- 普通料金（対キロ区間制）: https://map.kotsu.city.nagoya.jp/rp/subway/trp0000172.htm
- 混雑データ説明・4 PDF: https://www.kotsu.city.nagoya.jp/rp/subway/TRP0004057.htm
- 公式路線図: https://www.kotsu.city.nagoya.jp/rp/Jassets/pdf/subway/rosenzu/subwaymap.pdf

## 重要な仮定・限界

1. **PDFはOD表ではありません。** 公式説明どおり、自動改札通過人員をもとに時間帯別の車内平均混雑を推測した図です。断面交通量からODは一意に復元できません。
2. `section_load_relative.csv` はPDFの色階級を説明用の相対指数へ転記したプロキシで、絶対人数ではありません。再現研究では原PDFを基に研究者が値を再確認・差し替えてください。
3. 駅間距離は固定入力で、各路線の公表総営業キロへ合計が一致するよう丸め補正しています。個々の区間値は運賃検索の公式OD別結果との照合が必要です。
4. 混雑図は方向・時間帯を持ちますが、同梱プロキシは説明用に集約した単一断面です。方向別・時間帯別の厳密推定ではありません。
5. RFID検出は駅間で独立、誤検出生成なし、運賃未確定時は0円回収です。人体遮蔽による相関失敗や追加確認運用は未実装です。
6. `95%CI` はシード平均の信頼区間です。個々の運行で回収率がその範囲に収まるという意味ではありません。

このため本成果物は「PDFから真のODを復元したモデル」ではなく、**距離減衰型の基準ODと、断面混雑を制約に加えた簡易推定ODの感度比較**です。

既定の `--confidence-threshold 0` は常にMAP推定運賃を課す設定です。しきい値を上げると、事後確率が不足したケースを判定不能（0円）として扱います。方向が逆でも営業距離・運賃が同じODがあるため、OD事後確率しきい値をそのまま高くすると回収率を過度に低く見積もる点に注意してください。

## OD正解率・異運賃誤推定率

```bash
python3 run_od_accuracy_validation.py
```

---

## 公式ODデータ（2021年大都市交通センサス）の利用

`census_od` は、e-Statで公開されている「第13回大都市交通センサス
初乗り・最終降車駅間移動人員表」の中京圏シートを利用します。

```bash
python3 nagoya_bibo.py \
  --od-models census_od \
  --detection-rates 0.65 0.70 \
  --seed-count 150 \
  --sample-size 3300
```

OD正解率・異運賃誤推定率も同じODモデルで評価できます。

```bash
python3 run_od_accuracy_validation.py \
  --od-models census_od \
  --detection-rates 0.65 0.70 \
  --seed-count 150 \
  --sample-size 3300
```

既定実行では、次の3方式を比較します。

- `census_od`: 2021年大都市交通センサスの駅間移動人員
- `gravity`: 営業距離に対する距離減衰型OD
- `section_fit`: PDFの断面混雑プロキシに合わせた推定OD

`line_summary.csv`、`trip_length_summary.csv`、`od_candidates.csv` の
`od_model` 列に、どの方式を利用したかが記録されます。

### 公式ExcelからCSVを再生成する

`data/census_od_2021.csv` は同梱済みです。再生成する場合は次を実行します。

```bash
python3 prepare_estat_census_od.py
```

ダウンロード済みExcelを指定する場合：

```bash
python3 prepare_estat_census_od.py --xlsx census13_od.xlsx
```

e-Statで `*` と秘匿されているセルは、同梱CSVでは0人としてOD重みから
除外しています。感度分析では次のように仮置きできます。

```bash
python3 prepare_estat_census_od.py \
  --xlsx census13_od.xlsx \
  --suppressed-value 1.5
```

### 公式ODモデルの重要な限界

この表は初乗り駅と最終降車駅の組を示す駅ペアODです。本スクリプトは、
両端駅が対象路線の駅一覧に含まれるODを抽出します。しかし、利用者が実際に
その路線だけを通ったことや、途中で乗り換えなかったことまでは確認できません。
特に複数路線が接続する駅間ODは、複数路線の候補に入り得ます。このため、CSVの
`route_scope` には `station_pair_filter_not_route_verified` と記録しています。

PDF由来の混雑図は厳密なOD表ではありません。公式ODの代替ではなく、
断面混雑制約を用いた感度分析用の `section_fit` として残しています。

### 出典（取得日: 2026-08-07）

- e-Stat 第13回大都市交通センサス「初乗り・最終降車駅間移動人員表」
  https://www.e-stat.go.jp/stat-search/files?cycle=0&layout=datalist&page=1&tclass1=000001203341&tclass2=000001242436&tclass3=000001242439&tclass4val=0&toukei=00600020&tstat=000001103355
- 使用Excel（statInfId: 000040443413）
  https://www.e-stat.go.jp/stat-search/file-download?fileKind=0&statInfId=000040443413
- 国土交通省 第13回大都市交通センサス調査結果
  https://www.mlit.go.jp/report/press/sogo12_hh_000300.html

次を分離して複数シード集計します。

- `od_accuracy`: 乗車駅・降車駅とも正解
- `same_fare_wrong_od_rate`: ODは不正解だが同じ運賃帯
- `wrong_fare_rate`: 推定運賃が真の運賃と異なる
- `undercharge_rate` / `overcharge_rate`
- `fare_correct_rate`: OD正解＋同一運賃帯誤推定

出力先は `od_accuracy_outputs/` です。路線全体と利用区間別について、シード別CSV、平均・SD・95%CIの要約CSV、および比較グラフを生成します。
