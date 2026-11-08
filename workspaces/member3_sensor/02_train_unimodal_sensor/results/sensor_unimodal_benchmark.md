| feature_set | model | n_features | val_acc | val_macro_f1 | test_acc | test_macro_f1 | params | latency_ms |
|---|---|---|---|---|---|---|---|---|
| C_integrated19 | MLP (mean±std, 3 seeds) | 21 | 0.738 ± 0.045 | 0.736 ± 0.042 | 0.713 ± 0.033 | 0.690 ± 0.039 | 6483.0 | 0.1038 |
| C_integrated19 | RandomForest (500 trees) | 21 | 0.607 | 0.591 | 0.655 | 0.658 |  |  |
| A_nondestructive | MLP (mean±std, 3 seeds) | 17 | 0.643 ± 0.051 | 0.637 ± 0.051 | 0.425 ± 0.043 | 0.396 ± 0.063 | 6227.0 | 0.1255 |
| A_nondestructive | RandomForest (500 trees) | 17 | 0.536 | 0.517 | 0.448 | 0.443 |  |  |
| B_with_lab | MLP (mean±std, 3 seeds) | 20 | 0.786 ± 0.029 | 0.787 ± 0.031 | 0.724 ± 0.028 | 0.714 ± 0.026 | 6419.0 | 0.0942 |
| B_with_lab | RandomForest (500 trees) | 20 | 0.607 | 0.591 | 0.655 | 0.658 |  |  |
| T_tabular8 | MLP (mean±std, 3 seeds) | 8 | 0.774 ± 0.017 | 0.776 ± 0.018 | 0.598 ± 0.016 | 0.597 ± 0.020 | 5651.0 | 0.1121 |
| T_tabular8 | RandomForest (500 trees) | 8 | 0.714 | 0.700 | 0.552 | 0.556 |  |  |
| H_thermal_only | MLP (mean±std, 3 seeds) | 4 | 0.476 ± 0.017 | 0.462 ± 0.029 | 0.368 ± 0.043 | 0.368 ± 0.046 | 5395.0 | 0.1455 |
| H_thermal_only | RandomForest (500 trees) | 4 | 0.464 | 0.452 | 0.414 | 0.403 |  |  |
| M_multispectral_only | MLP (mean±std, 3 seeds) | 9 | 0.536 ± 0.000 | 0.530 ± 0.010 | 0.460 ± 0.043 | 0.435 ± 0.041 | 5715.0 | 0.1991 |
| M_multispectral_only | RandomForest (500 trees) | 9 | 0.321 | 0.311 | 0.517 | 0.500 |  |  |