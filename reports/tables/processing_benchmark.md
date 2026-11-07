# Processing benchmark: same CAMELS workload, four engines (measured)

Environment: `{"run_at": "2026-11-07T02:18:01Z", "python": "3.13.18", "polars": "1.35.2", "platform": "Linux-6.6.159+-x86_64-with-glibc2.39", "cpu_count": 2, "ram_gb": 12.7, "repeats": 3, "pandas": "2.2.3", "pyspark": "4.0.5"}`

Median wall time (read -> compute -> materialise) over repeated runs in fresh processes. Peak = RSS of Python + child processes (includes the Spark JVM). `count_mismatches` / `max_rel_diff_sums` compare each engine's output checksums with Polars lazy (0 = identical results).

| scale | rows | engine | seconds_median | seconds_min | seconds_max | spark_startup_s | peak_mb_median | incr_peak_mb_median | count_mismatches | max_rel_diff_sums | speedup_vs_pandas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1,682,616 | polars_eager | 3.677 | 2.939 | 4.079 | 0.000 | 778.031 | 736.500 | 0 | 0.000 | 6.133 |
| 1 | 1,682,616 | polars_lazy | 4.065 | 2.818 | 12.698 | 0.000 | 750.559 | 709.215 | 0 | 0.000 | 5.549 |
| 1 | 1,682,616 | pandas | 22.554 | 21.800 | 22.822 | 0.000 | 1,077.637 | 1,036.406 | 0 | 0.000 | 1.000 |
| 1 | 1,682,616 | pyspark | 38.663 | 36.944 | 39.294 | 17.620 | 1,257.461 | 728.410 | 0 | 0.000 | 0.583 |
| 4 | 6,730,464 | polars_lazy | 11.540 | 9.973 | 11.964 | 0.000 | 2,337.344 | 2,295.973 | 0 | 0.000 | 8.417 |
| 4 | 6,730,464 | polars_eager | 11.872 | 10.013 | 12.267 | 0.000 | 2,482.438 | 2,441.125 | 0 | 0.000 | 8.181 |
| 4 | 6,730,464 | pyspark | 79.345 | 78.747 | 81.848 | 18.065 | 1,347.977 | 811.230 | 0 | 0.000 | 1.224 |
| 4 | 6,730,464 | pandas | 97.126 | 96.491 | 109.319 | 0.000 | 3,264.508 | 3,223.238 | 0 | 0.000 | 1.000 |
