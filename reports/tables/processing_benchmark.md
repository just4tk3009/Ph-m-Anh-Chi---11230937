# Processing benchmark: same CAMELS workload, four engines (measured)

Environment: `{"run_at": "2026-11-06T08:18:39Z", "python": "3.13.18", "polars": "1.35.2", "platform": "Linux-6.6.159+-x86_64-with-glibc2.39", "cpu_count": 2, "ram_gb": 12.7, "repeats": 3, "pandas": "2.2.3", "pyspark": "4.0.5"}`

Median wall time (read -> compute -> materialise) over repeated runs in fresh processes. Peak = RSS of Python + child processes (includes the Spark JVM). `count_mismatches` / `max_rel_diff_sums` compare each engine's output checksums with Polars lazy (0 = identical results).

| scale | rows | engine | seconds_median | seconds_min | seconds_max | spark_startup_s | peak_mb_median | incr_peak_mb_median | count_mismatches | max_rel_diff_sums | speedup_vs_pandas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 693,162 | polars_lazy | 1.225 | 1.173 | 1.258 | 0.000 | 351.875 | 310.652 | 0 | 0.000 | 8.840 |
| 1 | 693,162 | polars_eager | 1.295 | 1.213 | 2.043 | 0.000 | 361.578 | 319.973 | 0 | 0.000 | 8.363 |
| 1 | 693,162 | pandas | 10.831 | 10.549 | 12.062 | 0.000 | 523.906 | 482.695 | 0 | 0.000 | 1.000 |
| 1 | 693,162 | pyspark | 30.940 | 26.570 | 32.293 | 19.457 | 914.043 | 352.156 | 0 | 0.000 | 0.350 |
| 4 | 2,772,648 | polars_eager | 5.000 | 4.235 | 5.521 | 0.000 | 1,068.496 | 1,026.965 | 0 | 0.000 | 7.816 |
| 4 | 2,772,648 | polars_lazy | 5.025 | 4.064 | 5.449 | 0.000 | 1,018.527 | 977.160 | 0 | 0.000 | 7.779 |
| 4 | 2,772,648 | pandas | 39.084 | 38.666 | 39.857 | 0.000 | 1,433.102 | 1,391.645 | 0 | 0.000 | 1.000 |
| 4 | 2,772,648 | pyspark | 51.529 | 49.957 | 52.363 | 19.037 | 1,340.312 | 780.680 | 0 | 0.000 | 0.758 |
