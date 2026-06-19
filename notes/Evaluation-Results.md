# Evaluation Results

This file collects the main `SMT-GraphFormer` and benchmark evaluation results. Boarding and alighting are unitless passenger counts, while delay and dwell are measured in seconds.

## Test-Set $R^2$ Summary

The best-performing model for each target is highlighted in bold.

| Model           |  Boarding | Alighting |     Delay |     Dwell |
| --------------- | --------: | --------: | --------: | --------: |
| XGBoost         |     0.487 |     0.449 |     0.115 |     0.311 |
| MLP             |     0.484 |     0.440 |     0.096 |     0.309 |
| ResNet          |     0.486 |     0.439 |     0.046 |     0.301 |
| FT-Transformer  |     0.511 |     0.476 |     0.147 |     0.300 |
| SMT-GraphFormer | **0.550** | **0.715** | **0.186** | **0.346** |

## SMT-GraphFormer

Teacher-forced evaluation across the training, validation, and test splits.

| Target    | Split      |    RMSE |    MAE | $R^2$ |
| --------- | ---------- | ------: | -----: | ----: |
| Boarding  | Training   |   1.901 |  0.887 | 0.651 |
| Boarding  | Validation |   1.978 |  0.940 | 0.554 |
| Boarding  | Test       |   1.861 |  0.892 | 0.550 |
| Alighting | Training   |   1.538 |  0.732 | 0.783 |
| Alighting | Validation |   1.597 |  0.786 | 0.716 |
| Alighting | Test       |   1.505 |  0.743 | 0.715 |
| Delay     | Training   | 121.590 | 66.583 | 0.400 |
| Delay     | Validation | 130.928 | 76.026 | 0.129 |
| Delay     | Test       | 116.498 | 70.958 | 0.186 |
| Dwell     | Training   |  34.213 |  8.914 | 0.355 |
| Dwell     | Validation |  34.430 |  9.236 | 0.337 |
| Dwell     | Test       |  33.739 |  9.109 | 0.346 |

Autoregressive evaluation across the training, validation, and test splits. The teacher-forced and autoregressive results are identical for the delay and dwell targets since they are encoder-side surrogate tasks.

| Target    | Split      |  RMSE |   MAE | $R^2$ |
| --------- | ---------- | ----: | ----: | ----: |
| Boarding  | Training   | 2.082 | 0.945 | 0.581 |
| Boarding  | Validation | 2.063 | 0.979 | 0.515 |
| Boarding  | Test       | 1.932 | 0.926 | 0.515 |
| Alighting | Training   | 2.269 | 0.970 | 0.529 |
| Alighting | Validation | 2.170 | 0.997 | 0.475 |
| Alighting | Test       | 2.044 | 0.941 | 0.474 |

## XGBoost Benchmark

Predictive performance of the XGBoost benchmark model across data splits.

| Target    | Split      |   RMSE |   MAE | $R^2$ |
| --------- | ---------- | -----: | ----: | ----: |
| Boarding  | Training   |  2.081 | 0.927 | 0.581 |
| Boarding  | Validation |  2.180 | 1.048 | 0.459 |
| Boarding  | Test       |  1.987 | 0.954 | 0.487 |
| Alighting | Training   |  2.211 | 0.932 | 0.552 |
| Alighting | Validation |  2.288 | 1.074 | 0.417 |
| Alighting | Test       |  2.092 | 0.972 | 0.449 |
| Delay     | Training   | 128.25 | 66.77 | 0.333 |
| Delay     | Validation | 131.21 | 75.42 | 0.125 |
| Delay     | Test       | 121.44 | 73.57 | 0.115 |
| Dwell     | Training   |  32.70 |  8.40 | 0.411 |
| Dwell     | Validation |  35.37 |  9.36 | 0.300 |
| Dwell     | Test       |  34.62 |  9.12 | 0.311 |

## RTDL Benchmarks

Predictive performance of the MLP and ResNet benchmark models across data splits.

| Target    | Split      | MLP RMSE | MLP MAE | MLP $R^2$ | ResNet RMSE | ResNet MAE | ResNet $R^2$ |
| --------- | ---------- | -------: | ------: | --------: | ----------: | ---------: | -----------: |
| Boarding  | Training   |    2.225 |   0.979 |     0.521 |       2.200 |      0.983 |        0.532 |
| Boarding  | Validation |    2.115 |   1.008 |     0.490 |       2.093 |      1.000 |        0.501 |
| Boarding  | Test       |    1.992 |   0.952 |     0.484 |       1.987 |      0.955 |        0.486 |
| Alighting | Training   |    2.398 |   0.998 |     0.474 |       2.349 |      0.996 |        0.495 |
| Alighting | Validation |    2.214 |   1.024 |     0.454 |       2.192 |      1.014 |        0.464 |
| Alighting | Test       |    2.110 |   0.964 |     0.440 |       2.111 |      0.964 |        0.439 |
| Delay     | Training   |   134.23 |   71.35 |     0.269 |      139.34 |      73.45 |        0.213 |
| Delay     | Validation |   136.70 |   79.14 |     0.050 |      134.62 |      77.43 |        0.079 |
| Delay     | Test       |   122.77 |   74.69 |     0.096 |      126.10 |      76.71 |        0.046 |
| Dwell     | Training   |    35.19 |    8.98 |     0.318 |       35.38 |       9.09 |        0.311 |
| Dwell     | Validation |    35.33 |    9.40 |     0.301 |       35.45 |       9.39 |        0.297 |
| Dwell     | Test       |    34.67 |    9.24 |     0.309 |       34.87 |       9.29 |        0.301 |

Predictive performance of the FT-Transformer benchmark model across data splits.

| Target    | Split      |   RMSE |   MAE | $R^2$ |
| --------- | ---------- | -----: | ----: | ----: |
| Boarding  | Training   |  2.198 | 0.983 | 0.533 |
| Boarding  | Validation |  2.072 | 0.993 | 0.511 |
| Boarding  | Test       |  1.939 | 0.935 | 0.511 |
| Alighting | Training   |  2.359 | 0.992 | 0.491 |
| Alighting | Validation |  2.157 | 1.000 | 0.482 |
| Alighting | Test       |  2.040 | 0.940 | 0.476 |
| Delay     | Training   | 129.15 | 70.72 | 0.324 |
| Delay     | Validation | 128.62 | 74.71 | 0.159 |
| Delay     | Test       | 119.26 | 72.38 | 0.147 |
| Dwell     | Training   |  35.67 |  9.08 | 0.299 |
| Dwell     | Validation |  35.50 |  9.29 | 0.295 |
| Dwell     | Test       |  34.90 |  9.15 | 0.300 |
