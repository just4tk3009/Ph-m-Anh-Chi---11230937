# Storage-format benchmark (measured)

Environment: `{"run_at": "2026-11-07T01:46:01Z", "python": "3.13.18", "polars": "1.35.2", "platform": "Linux-6.6.159+-x86_64-with-glibc2.39", "cpu_count": 2, "repeats": 5, "filter_year": 2005}`

Median of repeated runs, each in a fresh process. `incr_peak_mb` = peak RSS above the interpreter baseline.

| scale | rows_table | format | query | size_mb | size_vs_csv | read_s_median | speedup_vs_csv | incr_peak_mb_median | result_mb |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1,682,616 | csv | filter_proj | 444.457 | 1.000 | 0.918 | 1.000 | 480.398 | 1.233 |
| 1 | 1,682,616 | delta_zstd | filter_proj | 182.608 | 0.411 | 0.017 | 54.937 | 35.070 | 1.096 |
| 1 | 1,682,616 | parquet_snappy | filter_proj | 216.933 | 0.488 | 0.013 | 70.205 | 25.035 | 1.096 |
| 1 | 1,682,616 | parquet_zstd | filter_proj | 154.102 | 0.347 | 0.027 | 33.590 | 24.664 | 1.096 |
| 1 | 1,682,616 | csv | full_scan | 444.457 | 1.000 | 3.167 | 1.000 | 970.145 | 432.632 |
| 1 | 1,682,616 | delta_zstd | full_scan | 182.608 | 0.411 | 0.851 | 3.723 | 526.918 | 448.650 |
| 1 | 1,682,616 | parquet_snappy | full_scan | 216.933 | 0.488 | 0.662 | 4.783 | 716.172 | 448.764 |
| 1 | 1,682,616 | parquet_zstd | full_scan | 154.102 | 0.347 | 1.058 | 2.993 | 654.656 | 448.764 |
| 1 | 1,682,616 | csv | projection | 444.457 | 1.000 | 0.954 | 1.000 | 502.773 | 44.957 |
| 1 | 1,682,616 | delta_zstd | projection | 182.608 | 0.411 | 0.180 | 5.295 | 81.336 | 44.951 |
| 1 | 1,682,616 | parquet_snappy | projection | 216.933 | 0.488 | 0.065 | 14.749 | 88.094 | 44.968 |
| 1 | 1,682,616 | parquet_zstd | projection | 154.102 | 0.347 | 0.100 | 9.521 | 82.754 | 44.968 |
| 4 | 6,730,464 | csv | filter_proj | 1,783.773 | 1.000 | 3.657 | 1.000 | 1,835.711 | 4.932 |
| 4 | 6,730,464 | delta_zstd | filter_proj | 435.172 | 0.244 | 0.021 | 175.216 | 35.859 | 4.384 |
| 4 | 6,730,464 | parquet_snappy | filter_proj | 868.012 | 0.487 | 0.023 | 160.675 | 33.008 | 4.384 |
| 4 | 6,730,464 | parquet_zstd | filter_proj | 615.149 | 0.345 | 0.054 | 67.793 | 32.215 | 4.384 |
| 4 | 6,730,464 | csv | full_scan | 1,783.773 | 1.000 | 14.889 | 1.000 | 3,846.730 | 1,730.620 |
| 4 | 6,730,464 | delta_zstd | full_scan | 435.172 | 0.244 | 1.979 | 7.522 | 1,907.559 | 1,794.725 |
| 4 | 6,730,464 | parquet_snappy | full_scan | 868.012 | 0.487 | 2.649 | 5.620 | 2,831.051 | 1,795.088 |
| 4 | 6,730,464 | parquet_zstd | full_scan | 615.149 | 0.345 | 4.644 | 3.206 | 2,580.090 | 1,795.088 |
| 4 | 6,730,464 | csv | projection | 1,783.773 | 1.000 | 3.608 | 1.000 | 1,978.680 | 179.844 |
| 4 | 6,730,464 | delta_zstd | projection | 435.172 | 0.244 | 0.209 | 17.229 | 217.785 | 179.814 |
| 4 | 6,730,464 | parquet_snappy | projection | 868.012 | 0.487 | 0.458 | 7.880 | 310.023 | 179.866 |
| 4 | 6,730,464 | parquet_zstd | projection | 615.149 | 0.345 | 0.365 | 9.879 | 286.070 | 179.866 |
