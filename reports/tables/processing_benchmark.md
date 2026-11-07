# Processing benchmark: same CAMELS workload, four engines (measured)

Environment: `{"run_at": "2026-11-07T12:47:36Z", "python": "3.13.18", "polars": "1.35.2", "platform": "Linux-6.6.159+-x86_64-with-glibc2.39", "cpu_count": 2, "ram_gb": 12.7, "repeats": 3, "pandas": "2.2.3", "pyspark": "4.0.5"}`

Median wall time (read -> compute -> materialise) over repeated runs in fresh processes. Peak = RSS of Python + child processes (includes the Spark JVM). `count_mismatches` / `max_rel_diff_sums` compare each engine's output checksums with Polars lazy (0 = identical results).

| scale | rows | engine | seconds_median | seconds_min | seconds_max | spark_startup_s | peak_mb_median | incr_peak_mb_median | count_mismatches | max_rel_diff_sums | speedup_vs_pandas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1,682,616 | polars_lazy | 2.688 | 2.661 | 4.020 | 0.000 | 741.398 | 700.145 | 0 | 0.000 | 9.097 |
| 1 | 1,682,616 | polars_eager | 5.382 | 4.818 | 10.257 | 0.000 | 790.469 | 749.195 | 0 | 0.000 | 4.544 |
| 1 | 1,682,616 | pandas | 24.456 | 24.066 | 25.679 | 0.000 | 1,082.141 | 1,040.859 | 0 | 0.000 | 1.000 |
| 1 | 1,682,616 | pyspark | 36.819 | 35.929 | 39.063 | 18.139 | 1,264.785 | 730.582 | 0 | 0.000 | 0.664 |
| 4 | 6,730,464 | polars_lazy | 10.889 | 10.263 | 11.636 | 0.000 | 2,372.645 | 2,331.266 | 0 | 0.000 | 10.025 |
| 4 | 6,730,464 | polars_eager | 11.444 | 10.461 | 12.241 | 0.000 | 2,509.102 | 2,468.105 | 0 | 0.000 | 9.539 |
| 4 | 6,730,464 | pyspark | 78.627 | 77.681 | 79.449 | 17.460 | 1,436.629 | 900.691 | 0 | 0.000 | 1.388 |
| 4 | 6,730,464 | pandas | 109.162 | 108.517 | 114.459 | 0.000 | 3,258.836 | 3,217.508 | 0 | 0.000 | 1.000 |
