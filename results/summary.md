# SOM posture clustering — results

- Map: 19×21 units, 59 epochs, 16 units without training hits
- Number of clusters: **K = 3**, chosen on validation data by the mean rank of silhouette / Davies–Bouldin / Calinski–Harabasz
- Confidence reference distances (fit data): 95th pct 1.482, 99th pct 1.969

## Data

| set | frames | tracks | recordings |
|---|---|---|---|
| fit | 31121 | 158 | 61 |
| validation | 8677 | 47 | 18 |
| test | 11616 | 76 | 24 |

## Metrics (final weights)

| metric | fit | validation | test |
|---|---|---|---|
| Quantization error | 0.878 | 0.977 | 1.008 |
| Topographic error | 0.107 | 0.126 | 0.142 |
| Silhouette | 0.366 | 0.356 | 0.391 |
| Davies–Bouldin | 1.189 | 1.237 | 1.118 |
| Calinski–Harabasz | 11289.966 | 2841.610 | 5102.586 |
| ARI vs activity | 0.035 | 0.044 | 0.062 |
| NMI vs activity | 0.179 | 0.182 | 0.213 |
| Purity vs activity | 0.239 | 0.256 | 0.340 |
| Mean BMU confidence | 0.417 | 0.355 | 0.333 |
| Mean cluster margin | 0.586 | 0.543 | 0.544 |
| Novelty rate | 0.010 | 0.014 | 0.014 |
| Stable-frame share | 0.994 | 0.989 | 0.999 |
| Mean run length (frames) | 69.467 | 63.801 | 85.412 |
| Cluster switches / s | 0.280 | 0.308 | 0.155 |

## K selection (validation)

| k | silhouette | davies_bouldin | calinski_harabasz | rank_silhouette | rank_davies_bouldin | rank_calinski_harabasz | mean_rank |
|---|---|---|---|---|---|---|---|
| 2 | 0.3539 | 1.0897 | 2804.603 | 2.0 | 1.0 | 2.0 | 1.6667 |
| 3 | 0.3561 | 1.2372 | 2841.6096 | 1.0 | 2.0 | 1.0 | 1.3333 |
| 4 | 0.229 | 1.5098 | 2365.842 | 3.0 | 4.0 | 3.0 | 3.3333 |
| 5 | 0.2047 | 1.5594 | 1948.6301 | 8.0 | 6.0 | 4.0 | 6.0 |
| 6 | 0.2103 | 1.503 | 1839.726 | 5.0 | 3.0 | 5.0 | 4.3333 |
| 7 | 0.2051 | 1.7188 | 1737.7304 | 7.0 | 9.0 | 6.0 | 7.3333 |
| 8 | 0.21 | 1.6377 | 1622.3539 | 6.0 | 7.0 | 8.0 | 7.0 |
| 9 | 0.2227 | 1.5161 | 1664.2392 | 4.0 | 5.0 | 7.0 | 5.3333 |
| 10 | 0.1834 | 1.7177 | 1567.4005 | 9.0 | 8.0 | 9.0 | 8.6667 |
| 11 | 0.1791 | 1.7563 | 1479.1924 | 10.0 | 11.0 | 10.0 | 10.3333 |
| 12 | 0.1755 | 1.7378 | 1384.7006 | 11.0 | 10.0 | 11.0 | 10.6667 |
| 13 | 0.1278 | 1.8921 | 1280.7241 | 18.0 | 19.0 | 12.0 | 16.3333 |
| 14 | 0.1373 | 1.8063 | 1241.6974 | 15.0 | 13.0 | 13.0 | 13.6667 |
| 15 | 0.1276 | 1.8209 | 1195.5769 | 19.0 | 16.0 | 14.0 | 16.3333 |
| 16 | 0.1315 | 1.8337 | 1160.3825 | 17.0 | 17.0 | 15.0 | 16.3333 |
| 17 | 0.1341 | 1.816 | 1126.4325 | 16.0 | 15.0 | 17.0 | 16.0 |
| 18 | 0.1519 | 1.7745 | 1156.0167 | 14.0 | 12.0 | 16.0 | 14.0 |
| 19 | 0.1555 | 1.8155 | 1117.0415 | 12.0 | 14.0 | 18.0 | 14.6667 |
| 20 | 0.1544 | 1.8633 | 1078.8557 | 13.0 | 18.0 | 19.0 | 16.6667 |

## Map-size sensitivity (validation)

| rows | cols | units | val_qe | val_te |
|---|---|---|---|---|
| 10 | 10 | 100 | 1.2141 | 0.109 |
| 13 | 15 | 195 | 1.1023 | 0.104 |
| 19 | 21 | 399 | 0.9772 | 0.1256 |
| 23 | 26 | 598 | 0.9103 | 0.1466 |

Figures are in `figures/`. Activity labels were used only for the external metrics.
