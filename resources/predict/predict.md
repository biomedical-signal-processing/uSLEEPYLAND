## Predictions/Evaluation commands

### Evaluate NSRR Data (.h5 + .ids)

- USleep
  - Multi Channel EEG + EOG
    ```
    predict --project_dir "./u-sleep-nsrr-2022" --log_level INFO --seed 123 --folder_regex /[LOCAL_PATH]/data/processed/abc/abc* --eeg_channels F4-M1 C4-M1 --eog_channels E1-M2 E2-M1 --majority --dataset abc --out_dir ./predictions_abc_probs --num_gpus=0 --strip_func strip_to_match --one_shot --no_argmax --save_true --overwrite --log_file ./predictions_abc_probs/prediction_log
    ```
  - Single Channel EEG
    ```
    predict --project_dir "./u-sleep-nsrr-2022_eeg" --log_level INFO --seed 123 --folder_regex /[LOCAL_PATH]/data/processed/abc/abc* --eeg_channels F4-M1 C4-M1 --majority --dataset abc --out_dir ./predictions_abc_probs --num_gpus=0 --strip_func strip_to_match --one_shot --no_argmax --save_true --overwrite --log_file ./predictions_abc_probs/prediction_log
    ```
  - Single Channel EOG
    ```
    predict --project_dir "./u-sleep-nsrr-2022_eog" --log_level INFO --seed 123 --folder_regex /[LOCAL_PATH]/data/processed/abc/abc* --eog_channels E1-M2 E2-M1 --majority --dataset abc --out_dir ./predictions_abc_probs --num_gpus=0 --strip_func strip_to_match --one_shot --no_argmax --save_true --overwrite --log_file ./predictions_abc_probs/prediction_log
    ```

- Deepresnet
  - Multi Channel EEG + EOG
    ```
    predict --project_dir "./deepresnet-nsrr-2022" --log_level INFO --seed 123 --folder_regex /[LOCAL_PATH]/data/processed/abc/abc* --eeg_channels F4-M1 C4-M1 --eog_channels E1-M2 E2-M1 --majority --dataset abc --out_dir ./predictions_abc_probs --num_gpus=0 --strip_func strip_to_match --one_shot --save_true --overwrite --log_file ./predictions_abc_probs/prediction_log 
    ```
  - Single Channel EEG
    ```
    predict --project_dir "./deepresnet-nsrr-2022_eeg" --log_level INFO --seed 123 --folder_regex /[LOCAL_PATH]/data/processed/abc/abc* --eeg_channels F4-M1 C4-M1 --majority --dataset abc --out_dir ./predictions_abc_probs --num_gpus=0 --strip_func strip_to_match --one_shot --no_argmax --save_true --overwrite --log_file ./predictions_abc_probs/prediction_log
    ```
  - Single Channel EOG
    ```
    predict --project_dir "./deepresnet-nsrr-2022_eog" --log_level INFO --seed 123 --folder_regex /[LOCAL_PATH]/data/processed/abc/abc* --eog_channels E1-M2 E2-M1 --majority --dataset abc --out_dir ./predictions_abc_probs --num_gpus=0 --strip_func strip_to_match --one_shot --no_argmax --save_true --overwrite --log_file ./predictions_abc_probs/prediction_log
    ```

- Sleeptransformer
  - Multi Channel EEG + EOG
    ```
    predict --project_dir "./sleeptransformer-nsrr-2022" --log_level INFO --seed 123 --folder_regex /[LOCAL_PATH]/data/processed/abc/abc* --eeg_channels F4-M1 C4-M1 --eog_channels E1-M2 E2-M1 --majority --dataset abc --out_dir ./predictions_abc_probs --num_gpus=0 --strip_func strip_to_match --one_shot --no_argmax --save_true --overwrite --log_file ./predictions_abc_probs/prediction_log
    ```
  - Single Channel EEG
    ```
    predict --project_dir "./sleeptransformer-2022_eeg" --log_level INFO --seed 123 --folder_regex /[LOCAL_PATH]/data/processed/abc/abc* --eeg_channels F4-M1 C4-M1 --majority --dataset abc --out_dir ./predictions_abc_probs --num_gpus=0 --strip_func strip_to_match --one_shot --no_argmax --save_true --overwrite --log_file ./predictions_abc_probs/prediction_log
    ```
  - Single Channel EOG
    ```
    predict --project_dir "./sleeptransformer-nsrr-2022_eog" --log_level INFO --seed 123 --folder_regex /[LOCAL_PATH]/data/processed/abc/abc* --eog_channels E1-M2 E2-M1 --majority --dataset abc --out_dir ./predictions_abc_probs --num_gpus=0 --strip_func strip_to_match --one_shot --no_argmax --save_true --overwrite --log_file ./predictions_abc_probs/prediction_log
    ```

- YASA
  - Multi Channel EEG + EOG
    ```
    predict --project_dir "./yasa" --log_level INFO --seed 123 --folder_regex /[LOCAL_PATH]/data/processed/abc/abc* --eeg_channels F4-M1 C4-M1 --eog_channels E1-M2 E2-M1 --majority --dataset abc --out_dir ./predictions_abc_eeg_eog_probs --num_gpus=0 --strip_func strip_to_match --one_shot --no_argmax --save_true --overwrite --log_file ./predictions_abc_eeg_eog_probs/prediction_log --model_external yasa
    ```

[//]: # (  - Single Channel EEG)

[//]: # (    ```)

[//]: # (    predict --project_dir "./yasa" --log_level INFO --seed 123 --folder_regex /[LOCAL_PATH]/data/processed/abc/abc* --eeg_channels F4-M1 C4-M1 --majority --dataset abc --out_dir ./predictions_abc_eeg_probs --num_gpus=0 --strip_func strip_to_match --one_shot --no_argmax --save_true --overwrite --log_file ./predictions_abc_eeg_probs/prediction_log --model_external yasa)

[//]: # (    ```)

### Predict on myEDF Data (.edf)

- USleep
  - Multi Channel EEG + EOG
    ```
    predict_one --project_dir ./u-sleep-nsrr-2022 --log_level INFO --seed 123 -f /[LOCAL_PATH]/sleepyland/notebook/src/lunapi-notebooks/tutorial/edfs/*1.edf --out_dir ./predictions_myEDF_probs --logging_out_path ./predictions_myEDF_probs/prediction_log --num_gpus 0 --auto_channel_grouping EEG EOG MASTOID --auto_reference_types EEG EOG --majority --strip_func trim_psg_trailing --overwrite --no_argmax
    ```
  - Single Channel EEG
    ```
    predict_one --project_dir ./u-sleep-nsrr-2022_eeg --log_level INFO --seed 123 -f /[LOCAL_PATH]/sleepyland/notebook/src/lunapi-notebooks/tutorial/edfs/*1.edf --out_dir ./predictions_myEDF_probs --logging_out_path ./predictions_myEDF_probs/prediction_log --num_gpus 0 --auto_channel_grouping EEG MASTOID --auto_reference_types EEG --majority --strip_func trim_psg_trailing --overwrite --no_argmax
    ```
  - Single Channel EOG
    ```
    predict_one --project_dir ./u-sleep-nsrr-2022_eog --log_level INFO --seed 123 -f /[LOCAL_PATH]/sleepyland/notebook/src/lunapi-notebooks/tutorial/edfs/*1.edf --out_dir ./predictions_myEDF_probs --logging_out_path ./predictions_myEDF_probs/prediction_log --num_gpus 0 --auto_channel_grouping EOG MASTOID --auto_reference_types EOG --majority --strip_func trim_psg_trailing --overwrite --no_argmax
    ```

- Deepresnet
  - Multi Channel EEG + EOG
    ```
    predict_one --project_dir ./deepresnet-nsrr-2022 --log_level INFO --seed 123 -f /[LOCAL_PATH]/sleepyland/notebook/src/lunapi-notebooks/tutorial/edfs/*1.edf --out_dir ./predictions_myEDF_probs --logging_out_path ./predictions_myEDF_probs/prediction_log --num_gpus 0 --auto_channel_grouping EEG EOG MASTOID --auto_reference_types EEG EOG --majority --strip_func trim_psg_trailing --overwrite --no_argmax
    ```
  - Single Channel EEG
    ```
    predict_one --project_dir ./deepresnet-nsrr-2022_eeg --log_level INFO --seed 123 -f /[LOCAL_PATH]/sleepyland/notebook/src/lunapi-notebooks/tutorial/edfs/*1.edf --out_dir ./predictions_myEDF_probs --logging_out_path ./predictions_myEDF_probs/prediction_log --num_gpus 0 --auto_channel_grouping EEG MASTOID --auto_reference_types EEG --majority --strip_func trim_psg_trailing --overwrite --no_argmax
    ```
  - Single Channel EOG
    ```
    predict_one --project_dir ./deepresnet-nsrr-2022_eog --log_level INFO --seed 123 -f /[LOCAL_PATH]/sleepyland/notebook/src/lunapi-notebooks/tutorial/edfs/*1.edf --out_dir ./predictions_myEDF_probs --logging_out_path ./predictions_myEDF_probs/prediction_log --num_gpus 0 --auto_channel_grouping EOG MASTOID --auto_reference_types EOG --majority --strip_func trim_psg_trailing --overwrite --no_argmax
    ```

- Sleeptransformer
  - Multi Channel EEG + EOG
    ```
    predict_one --project_dir ./sleeptransformer-nsrr-2022 --log_level INFO --seed 123 -f /[LOCAL_PATH]/sleepyland/notebook/src/lunapi-notebooks/tutorial/edfs/*1.edf --out_dir ./predictions_myEDF_probs --logging_out_path ./predictions_myEDF_probs/prediction_log --num_gpus 0 --auto_channel_grouping EEG EOG MASTOID --auto_reference_types EEG EOG --majority --strip_func trim_psg_trailing --overwrite --no_argmax
    ```
  - Single Channel EEG
    ```
    predict_one --project_dir ./sleeptransformer-nsrr-2022_eeg --log_level INFO --seed 123 -f /[LOCAL_PATH]/sleepyland/notebook/src/lunapi-notebooks/tutorial/edfs/*1.edf --out_dir ./predictions_myEDF_probs --logging_out_path ./predictions_myEDF_probs/prediction_log --num_gpus 0 --auto_channel_grouping EEG MASTOID --auto_reference_types EEG --majority --strip_func trim_psg_trailing --overwrite --no_argmax
    ```
  - Single Channel EOG
    ```
    predict_one --project_dir ./sleeptransformer-nsrr-2022_eog --log_level INFO --seed 123 -f /[LOCAL_PATH]/sleepyland/notebook/src/lunapi-notebooks/tutorial/edfs/*1.edf --out_dir ./predictions_myEDF_probs --logging_out_path ./predictions_myEDF_probs/prediction_log --num_gpus 0 --auto_channel_grouping EOG MASTOID --auto_reference_types EOG --majority --strip_func trim_psg_trailing --overwrite --no_argmax
    ```

- YASA
  - Multi Channel EEG + EOG
    ```
    predict_one --project_dir ./yasa --log_level INFO --seed 123 -f /[LOCAL_PATH]/sleepyland/notebook/src/lunapi-notebooks/tutorial/edfs/*1.edf --out_dir ./predictions_myEDF_eeg_eog_probs --logging_out_path ./predictions_myEDF_eeg_eog_probs/prediction_log --num_gpus 0 --auto_channel_grouping EEG EOG MASTOID --auto_reference_types EEG EOG --majority --strip_func trim_psg_trailing --overwrite --model_external yasa --no_argmax
    ```
  - Single Channel EEG
    ```
    predict_one --project_dir ./yasa --log_level INFO --seed 123 -f /[LOCAL_PATH]/sleepyland/notebook/src/lunapi-notebooks/tutorial/edfs/*1.edf --out_dir ./predictions_myEDF_eeg_probs --logging_out_path ./predictions_myEDF_eeg_probs/prediction_log --num_gpus 0 --auto_channel_grouping EEG MASTOID --auto_reference_types EEG --majority --strip_func trim_psg_trailing --overwrite --model_external yasa --no_argmax
    ```