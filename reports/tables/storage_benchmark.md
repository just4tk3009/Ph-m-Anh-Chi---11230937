# Storage-format benchmark (measured)

Environment: `{"run_at": "2026-11-07T12:14:51Z", "python": "3.13.18", "polars": "1.35.2", "platform": "Linux-6.6.159+-x86_64-with-glibc2.39", "cpu_count": 2, "repeats": 5, "filter_year": 2005}`

Median of repeated runs, each in a fresh process. `incr_peak_mb` = peak RSS above the interpreter baseline.

| scale | rows_table | format | query | size_mb | size_vs_csv | read_s_median | speedup_vs_csv | incr_peak_mb_median | result_mb |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1,682,616 | csv | filter_proj | 457.313 | 1.000 | 0.925 | 1.000 | 492.195 | 1.233 |
| 1 | 1,682,616 | delta_zstd | filter_proj | 183.386 | 0.401 | 0.023 | 41.030 | 32.863 | 1.096 |
| 1 | 1,682,616 | parquet_snappy | filter_proj | 217.660 | 0.476 | 0.013 | 72.735 | 24.082 | 1.096 |
| 1 | 1,682,616 | parquet_zstd | filter_proj | 154.531 | 0.338 | 0.016 | 57.471 | 24.055 | 1.096 |
| 1 | 1,682,616 | csv | full_scan | 457.313 | 1.000 | 3.267 | 1.000 | 983.602 | 445.488 |
| 1 | 1,682,616 | delta_zstd | full_scan | 183.386 | 0.401 | 0.876 | 3.728 | 538.609 | 448.468 |
| 1 | 1,682,616 | parquet_snappy | full_scan | 217.660 | 0.476 | 1.013 | 3.224 | 733.086 | 448.583 |
| 1 | 1,682,616 | parquet_zstd | full_scan | 154.531 | 0.338 | 1.355 | 2.411 | 671.852 | 448.583 |
| 1 | 1,682,616 | csv | projection | 457.313 | 1.000 | 0.934 | 1.000 | 514.945 | 44.957 |
| 1 | 1,682,616 | delta_zstd | projection | 183.386 | 0.401 | 0.177 | 5.275 | 79.934 | 44.951 |
| 1 | 1,682,616 | parquet_snappy | projection | 217.660 | 0.476 | 0.065 | 14.431 | 87.617 | 44.968 |
| 1 | 1,682,616 | parquet_zstd | projection | 154.531 | 0.338 | 0.091 | 10.274 | 81.773 | 44.968 |
| 4 | 6,730,464 | csv | filter_proj | 1,835.199 | 1.000 | 3.672 | 1.000 | 1,883.602 | 4.932 |
| 4 | 6,730,464 | delta_zstd | filter_proj | 438.189 | 0.239 | 0.027 | 134.128 | 28.113 | 4.384 |
| 4 | 6,730,464 | parquet_snappy | filter_proj | 870.794 | 0.474 | 0.023 | 161.357 | 32.758 | 4.384 |
| 4 | 6,730,464 | parquet_zstd | filter_proj | 616.876 | 0.336 | 0.030 | 123.887 | 31.113 | 4.384 |
| 4 | 6,730,464 | csv | full_scan | 1,835.199 | 1.000 | 14.711 | 1.000 | 3,902.938 | 1,782.003 |
| 4 | 6,730,464 | delta_zstd | full_scan | 438.189 | 0.239 | 2.865 | 5.134 | 1,949.457 | 1,793.998 |
| 4 | 6,730,464 | parquet_snappy | full_scan | 870.794 | 0.474 | 2.969 | 4.955 | 2,900.004 | 1,794.361 |
| 4 | 6,730,464 | parquet_zstd | full_scan | 616.876 | 0.336 | 4.626 | 3.180 | 2,649.531 | 1,794.361 |
| 4 | 6,730,464 | csv | projection | 1,835.199 | 1.000 | 3.703 | 1.000 | 2,029.426 | 179.841 |
| 4 | 6,730,464 | delta_zstd | projection | 438.189 | 0.239 | 0.222 | 16.686 | 208.441 | 179.814 |
| 4 | 6,730,464 | parquet_snappy | projection | 870.794 | 0.474 | 0.249 | 14.858 | 309.113 | 179.866 |
| 4 | 6,730,464 | parquet_zstd | projection | 616.876 | 0.336 | 0.650 | 5.694 | 284.727 | 179.866 |
