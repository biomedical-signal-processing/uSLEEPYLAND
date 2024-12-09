## Metrics commands

### NSRR DATA
> U-Sleep on abc dataset
```
cm --project_dir ./u-sleep-nsrr-2022 --true ./predictions_all_test_data/test_data/*/*TRUE.npy --pred ./predictions_all_test_data/test_data/*/majority/*PRED.npy --ignore_classes 5
```