# Storage-format benchmark (measured)

Environment: `{"run_at": "2026-11-06T07:28:02Z", "python": "3.13.18", "polars": "1.35.2", "platform": "Linux-6.6.159+-x86_64-with-glibc2.39", "cpu_count": 2, "repeats": 5, "filter_year": 2013}`

Median of repeated runs, each in a fresh process. `incr_peak_mb` = peak RSS above the interpreter baseline.

| scale | rows_table | format | query | size_mb | size_vs_csv | read_s_median | speedup_vs_csv | incr_peak_mb_median | result_mb |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 693,162 | csv | filter_proj | 175.525 | 1.000 | 0.373 | 1.000 | 199.953 | 0.961 |
| 1 | 693,162 | delta_zstd | filter_proj | 75.320 | 0.429 | 0.015 | 24.412 | 35.328 | 0.854 |
| 1 | 693,162 | parquet_snappy | filter_proj | 85.336 | 0.486 | 0.014 | 26.454 | 22.781 | 0.854 |
| 1 | 693,162 | parquet_zstd | filter_proj | 61.124 | 0.348 | 0.012 | 31.930 | 22.234 | 0.854 |
| 1 | 693,162 | csv | full_scan | 175.525 | 1.000 | 1.217 | 1.000 | 370.141 | 166.052 |
| 1 | 693,162 | delta_zstd | full_scan | 75.320 | 0.429 | 0.359 | 3.392 | 216.477 | 163.408 |
| 1 | 693,162 | parquet_snappy | full_scan | 85.336 | 0.486 | 0.257 | 4.730 | 282.211 | 163.408 |
| 1 | 693,162 | parquet_zstd | full_scan | 61.124 | 0.348 | 0.424 | 2.869 | 258.258 | 163.408 |
| 1 | 693,162 | csv | projection | 175.525 | 1.000 | 0.572 | 1.000 | 207.266 | 18.509 |
| 1 | 693,162 | delta_zstd | projection | 75.320 | 0.429 | 0.053 | 10.766 | 52.855 | 18.509 |
| 1 | 693,162 | parquet_snappy | projection | 85.336 | 0.486 | 0.054 | 10.666 | 43.676 | 18.509 |
| 1 | 693,162 | parquet_zstd | projection | 61.124 | 0.348 | 0.041 | 14.059 | 41.680 | 18.509 |
| 4 | 2,772,648 | csv | filter_proj | 704.537 | 1.000 | 1.444 | 1.000 | 752.203 | 3.844 |
| 4 | 2,772,648 | delta_zstd | filter_proj | 171.194 | 0.243 | 0.020 | 71.723 | 37.836 | 3.417 |
| 4 | 2,772,648 | parquet_snappy | filter_proj | 341.480 | 0.485 | 0.020 | 73.430 | 30.492 | 3.417 |
| 4 | 2,772,648 | parquet_zstd | filter_proj | 244.434 | 0.347 | 0.040 | 36.464 | 29.250 | 3.417 |
| 4 | 2,772,648 | csv | full_scan | 704.537 | 1.000 | 5.037 | 1.000 | 1,431.523 | 664.209 |
| 4 | 2,772,648 | delta_zstd | full_scan | 171.194 | 0.243 | 1.088 | 4.631 | 711.973 | 653.632 |
| 4 | 2,772,648 | parquet_snappy | full_scan | 341.480 | 0.485 | 1.120 | 4.497 | 1,076.621 | 653.632 |
| 4 | 2,772,648 | parquet_zstd | full_scan | 244.434 | 0.347 | 1.676 | 3.006 | 980.832 | 653.632 |
| 4 | 2,772,648 | csv | projection | 704.537 | 1.000 | 1.737 | 1.000 | 792.254 | 74.038 |
| 4 | 2,772,648 | delta_zstd | projection | 171.194 | 0.243 | 0.097 | 17.816 | 109.176 | 74.038 |
| 4 | 2,772,648 | parquet_snappy | projection | 341.480 | 0.485 | 0.103 | 16.847 | 134.539 | 74.038 |
| 4 | 2,772,648 | parquet_zstd | projection | 244.434 | 0.347 | 0.308 | 5.638 | 124.410 | 74.038 |
