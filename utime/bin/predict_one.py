import logging
import os
import numpy as np
import mne
import math
from pprint import pformat
from collections import namedtuple
from argparse import ArgumentParser, Namespace
from utime import Defaults
from utime.utils.system import find_and_set_gpus
from utime.bin.evaluate import get_and_load_one_shot_model, get_and_load_model
from utime.utils.nn_utils import softmax
from psg_utils.dataset.sleep_study import SleepStudy
from psg_utils.hypnogram.utils import dense_to_sparse
from psg_utils.io.header import extract_header
from psg_utils.io.channels import infer_channel_types, VALID_CHANNEL_TYPES, is_eeg_central
from psg_utils.io.channels import auto_infer_referencing as infer_channel_refs
from psg_utils.io.channels.utils import get_channel_group_combinations
from psg_utils.preprocessing.scaling import batch_scaling
from psg_utils.preprocessing.spectrogram import compute_spectrogram
from utime.utils.scriptutils import add_logging_file_handler

logger = logging.getLogger(__name__)


def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description='Predict using a U-Time-Sleepyland model.')
    parser.add_argument("-f", type=str, required=True,
                        help='Path to file to predict on.')
    parser.add_argument("--out_dir", type=str, default="predictions",
                        help="Output folder to store results")
    parser.add_argument("--header_file_name", type=str, default=None,
                        help='Optional header file name. Header must be in the same folder of the input file, see -f.')
    parser.add_argument("--logging_out_path", type=str, default=None,
                        help='Optional path to store prediction log. If not set, <out_folder>/<file_name>.log is used.')
    parser.add_argument("--channels", nargs='+', type=str, default=None,
                        help="A list of channels to use for prediction. "
                             "To predict on multiple channel groups, pass a string where "
                             "each channel in each channel group is separated by '++' and different groups are "
                             "separated by space or '&&'. E.g. to predict on {EEG1, EOG1} and {EEG2, EOG2}, pass "
                             "'EEG1++EOG1' 'EEG2++EOG2'. Each group will be used for prediction once, and the final "
                             "results will be a majority vote across all. "
                             "You may also specify a list of individual channels and use the --auto_channel_grouping to"
                             " predict on all channel group combinations possible by channel types. "
                             "You may optionally also specify channel types using general channel declarations "
                             "['EEG', 'EOG', 'EMG'] which will be considered when using the --auto_channel_grouping "
                             "flag. Use '<channel_name>==<channel_type>', e.g. 'C3-A2==EEG' 'EOGl==EOG'.")
    parser.add_argument("--auto_channel_grouping", nargs="+", type=str, default=None,
                        help="Attempt to automatically group all channels specified with --channels into channel "
                             "groups by types. Pass a string of format '<type_1> <type_2>' (optional && separaters) "
                             "using the general channel types declarations ['EEG', 'EOG', 'EMG']. "
                             "E.g. to predict on all available channel groups with 1 EEG and 1 EOG channel "
                             "(in that order), pass '--auto_channel_grouping=EEG EOG' and all channels to consider "
                             "with the --channels argument. Channel types may be passed with --channels (see above), "
                             "otherwise, channel types are automatically inferred from the channel names. "
                             "Note that not all models are designed to work with all types, e.g. U-Sleep V1.0 "
                             "does not need EMG inputs and should not be passed.")
    parser.add_argument("--auto_reference_types", nargs='+', type=str, default=None,
                        help="Attempt to automatically reference channels to MASTOID typed channels. Pass channel "
                             "types in ['EEG', 'EOG'] for which this feature should be active. E.g., with "
                             "--channels C3 C4 A1 A2 passed and --auto_reference_types EEG set, the referenced "
                             "channels C3-A2 and C4-A1 will be used instead.")
    parser.add_argument("--majority", action="store_true",
                        help="Output a majority vote across channel groups in addition "
                             "to the individual channels.")
    parser.add_argument("--strip_func", type=str, default='trim_psg_trailing',
                        help="Strip function to use, default = 'trim_psg_trailing'.")
    parser.add_argument("--model", type=str, default=None,
                        help="Specify a model by string identifier of format <model_name>:<model_version> "
                             "available in the U-Sleep package. OBS: The U-Sleep package must be installed or an "
                             "error is raised. Cannot specify both --model and --project_dir")
    parser.add_argument("--data_per_prediction", type=int, default=None,
                        help='Number of samples that should make up each sleep'
                             ' stage scoring. Defaults to sample_rate*30, '
                             'giving 1 segmentation per 30 seconds of signal. '
                             'Set this to 1 to score every data point in the '
                             'signal.')
    parser.add_argument("--one_shot", action="store_true", default=False,
                        help="Segment each SleepStudy in one forward-pass "
                             "instead of using (GPU memory-efficient) sliding "
                             "window predictions.")
    parser.add_argument("--is_logits", action="store_true", default=False,
                        help='Model outputs are logits and should be softmaxed.')
    parser.add_argument("--num_gpus", type=int, default=0,
                        help="Number of GPUs to use for this job")
    parser.add_argument("--force_gpus", type=str, default="")
    parser.add_argument("--no_argmax", action="store_true",
                        help="Do not argmax prediction volume prior to save.")
    parser.add_argument("--weights_file_name", type=str, required=False,
                        help="Specify the exact name of the weights file "
                             "(located in <project_dir>/model/) to use.")
    parser.add_argument("--overwrite", action='store_true',
                        help='Overwrite existing output files and log files.')
    parser.add_argument("--model_external", type=str, default=None,
                        help="")
    parser.add_argument("--hdeeg", action='store_true',
                        help='Prediction on hdeeg data.')
    return parser


def get_processed_args(args):
    """
    Validate and prepare args.
    Returns a new set of args with potential modifications.

    Returns:
         Path to a validated project directory as per --project_dir.
    """
    modified_args = {}
    for key, value in vars(args).items():
        if isinstance(value, list):
            # Allow list-like arguments to be passed either space-separated as normally,
            # or using '&&' delimiters. This is useful e.g. when using Docker.
            split_list = []
            for item in value:
                split_list.extend(map(lambda s: s.strip(), item.split("&&")))
            value = split_list
        modified_args[key] = value
    args = Namespace(**modified_args)
    assert args.num_gpus >= 0, "--num_gpus must be positive or 0."

    manual_grouping_is_used = args.channels and any('++' in c for c in args.channels)
    if args.model:
        logger.info(f"Using the --model flag. "
                    f"Models (if any) at project directory path {Defaults.PROJECT_DIRECTORY} (if set) "
                    f"will not be considered.")
        try:
            import usleep
        except ImportError as e:
            raise RuntimeError("Cannot use the --model flag when the U-Sleep package is "
                               "not installed.") from e
        model_name, model_version = args.model.split(":")
        project_dir = usleep.get_model_path(model_name, model_version)
        if not manual_grouping_is_used and not args.auto_channel_grouping:
            channel_types = usleep.get_model_description(model_name, model_version)['channel_types']
            args.auto_channel_grouping = channel_types
    else:
        if args.channels is None and args.auto_channel_grouping is None:
            raise RuntimeError("Must specify the --channels or --auto_channel_grouping flag arguments "
                               "when not using the --model flag.")
        project_dir = os.path.abspath(Defaults.PROJECT_DIRECTORY)

    if manual_grouping_is_used and args.auto_channel_grouping:
        # If manually specifying groups, don't allow auto channel grouping
        args.auto_channel_grouping = None
        raise RuntimeWarning("Note that the --auto_channel_grouping flag has no effect when manually specifying "
                             "prediction groups in the --channels flag and should thus not be set.")

    # Check project folder is valid
    from utime.utils.scriptutils.scriptutils import assert_project_folder
    if not args.model_external:
        assert_project_folder(project_dir, evaluation=True)
    args.project_dir = project_dir

    # Set absolute input file path
    args.f = os.path.abspath(args.f)

    # Check header exists if specified
    if args.header_file_name and not os.path.exists(os.path.join(os.path.split(args.f)[0], args.header_file_name)):
        raise ValueError(f"Could not find header file with name {args.header_file_path} in the "
                         f"folder where input file {args.f} is stored.")

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Set logging out path
    default_log_file_path = os.path.splitext(args.out_dir)[0] + ".log"
    if args.logging_out_path is None:
        args.logging_out_path = default_log_file_path
    elif os.path.isdir(args.logging_out_path):
        args.logging_out_path = os.path.join(args.logging_out_path, os.path.split(default_log_file_path)[-1])

    return args

def save_file(path, arr, argmax):
    path = os.path.abspath(path)
    dir_ = os.path.split(path)[0]
    os.makedirs(dir_, exist_ok=True)
    if argmax:
        arr = arr.argmax(-1)
    logger.info(f"Saving array of shape {arr.shape} to {path}")
    np.save(path, arr)


def get_updated_majority_voted(majority_voted, pred):
    if majority_voted is None:
        majority_voted = pred.copy()
    else:
        majority_voted += pred
    return majority_voted


def get_save_path(out_dir, file_name, sub_folder_name=None):
    # Get paths
    if sub_folder_name is not None:
        out_dir_pred = os.path.join(out_dir, sub_folder_name)
    else:
        out_dir_pred = out_dir
    out_path = os.path.join(out_dir_pred, file_name)
    return out_path


def run_pred_on(study, channel_group, model, model_external, args, hparams):
    pred = None
    if not model_external:
        # sliding window prediction
        if not args.one_shot:
            psg = study.get_all_periods()
            psg_subset = psg[..., tuple(channel_group.channel_indices)]
            entire_psg = psg_subset.copy()

            psg_shape = entire_psg.shape
            total_epochs = psg_shape[0]

            batch_shape = hparams.get("build").get("batch_shape")
            num_epochs_per_sample = batch_shape[1]

            total_samples = math.ceil(total_epochs / num_epochs_per_sample)
            last_sample_epochs = total_epochs % num_epochs_per_sample

            psg_subset = np.zeros((total_samples, num_epochs_per_sample, *psg_shape[1:]), dtype=psg_subset.dtype)
            for i in range(total_samples):
                start_ind = i * num_epochs_per_sample
                end_ind = start_ind + num_epochs_per_sample
                # in the last sample, which can be shorter that the rest, pick the last epochs
                if i == total_samples - 1 and last_sample_epochs != 0:
                    psg_subset[i] = entire_psg[-num_epochs_per_sample:, ...]
                else:
                    psg_subset[i] = entire_psg[start_ind:end_ind, ...]
        else: # one-shot prediction
            psg = np.expand_dims(study.get_all_periods(), 0)  # adds batch dimension
            psg_subset = psg[..., tuple(channel_group.channel_indices)]

        logger.info(f"Spectrogram {hparams.get('fit').get('get_spectrogram', False)}")
        logger.info(f"Scaling {hparams.get('scaler', None)}")
        logger.info(f"Extracted PSG shape: {psg_subset.shape}")
        # for models requiring spectrograms compute them and possibly scale them
        if hparams["fit"].get("get_spectrogram"):
            psg_subset = compute_spectrogram(psg_subset)
            if bool(hparams.get("scaler", None)):
                batch_scaling(psg_subset, hparams.get("scaler"))

        logger.info(f"\n--- Channel names: {channel_group.channel_names}\n"
                    f"--- Channel inds: {channel_group.channel_indices}\n"
                    f"--- Processed PSG shape: {psg_subset.shape}")

        pred = model.predict_on_batch(psg_subset)

        if args.is_logits:
            logger.info("Applying softmax to predictions of shape: {}".format(pred.shape))
            pred = softmax(pred, axis=-1)

        if not args.one_shot:
            # reshape to (total_epochs, n_classes)
            pred = pred.reshape(-1, pred.shape[-1])
            # must throw away the last predictions if the last sample is shorter than the rest
            if last_sample_epochs != 0:
                num_invalid_predictions = num_epochs_per_sample - last_sample_epochs
                logger.info(f"Cutting off {num_invalid_predictions} duplicate predictions")
                pred = np.concatenate([pred[:-num_epochs_per_sample], pred[-last_sample_epochs:]])
    else:
        psg = study.psg[:, tuple(channel_group.channel_indices)]
        if model_external == 'yasa':
            import yasa
            from psg_utils.io.channels import infer_channel_types
            # Infer channel types from selected channels
            channel_types = infer_channel_types(channel_group.channel_names)

            # Create a raw object to give in input to yasa classifier
            info_ = mne.create_info(ch_names=channel_group.channel_names,
                                   ch_types=[ch.lower() for ch in channel_types],
                                   sfreq=study.sample_rate)
            raw_ = mne.io.RawArray(psg.T, info_)

            # Identify EEG and EOG channels based on inferred types
            eeg_name = next((name for name, ch_type in zip(channel_group.channel_names, channel_types) if ch_type.lower() == 'eeg'),
                            None)
            eog_name = next((name for name, ch_type in zip(channel_group.channel_names, channel_types) if ch_type.lower() == 'eog'),
                            None)
            emg_name = next((name for name, ch_type in zip(channel_group.channel_names, channel_types) if ch_type.lower() == 'emg'),
                            None)

            # Predict on yasa
            sls_yasa = yasa.SleepStaging(raw_,
                                         eeg_name=eeg_name,
                                         eog_name=eog_name,
                                         emg_name=emg_name)
            df_pred = sls_yasa.predict_proba()
            pred_list = [df_pred[['W', 'N1', 'N2', 'N3', 'R']].values.astype(np.float32)]

            # Get prediction
            pred = pred_list[0].copy()

        if model_external == 'pops':
            import lunapi as lp
            proj = lp.proj()

            sig = psg
            ch_eeg = channel_group.channel_names[0]
            # d = pd.read_table('d.txt', header=None)

            p = proj.empty_inst(study.identifier,
                                int(sig.shape[0]/study.sample_rate), 1)

            p.insert_signal(ch_eeg, sig, study.sample_rate)

            p.eval(f'RUN-POPS sig={ch_eeg} {Defaults.PROJECT_DIRECTORY}+"-model"')

            df_pred = p.table('RUN_POPS', 'E')

            pred_list = [df_pred[['PP_W', 'PP_N1', 'PP_N2', 'PP_N3', 'PP_R']].values.astype(np.float32)]

            # Get prediction
            pred = pred_list[0].copy()

    if callable(getattr(pred, "numpy", None)):
        pred = pred.numpy()
    pred = pred.reshape(-1, pred.shape[-1])
    return pred


def predict_study(study, model, channel_groups, model_external, args, hparams):
    identifier, _ = os.path.splitext(os.path.split(study.psg_file_path)[-1])
    majority_voted = None
    path_mj = get_save_path(args.out_dir, identifier + "_PRED.npy", "majority")

    k, k_ext_model = 0, 0

    for k, channel_group in enumerate(channel_groups):

        # Infer channel types from selected channels
        channel_names = list(channel_group.channel_names)
        channel_types = infer_channel_types(channel_names)

        # Skip prediction if predict with YASA and EEG is present but not from central region
        if model_external == "yasa" and "EEG" in channel_types and not is_eeg_central(channel_names[0]):
            logger.info(f"Skipping (channels={channel_group}) - YASA run on central derivations.")
            k_ext_model += 1
            continue

        # Skip prediction if predict with POPS, EEG not unique, and EEG is present but not from central region
        # Check if there's exactly one EEG channel and no other types
        only_one_eeg = channel_types.count('EEG') == 1 and all(ch_type == 'EEG' for ch_type in channel_types)
        if model_external == "pops" and only_one_eeg and not is_eeg_central(channel_names[0]):
            logger.info(f"Skipping (channels={channel_group}) - POPS run on EEG only and central derivations.")
            k_ext_model += 1
            continue

        # Get prediction out path
        path_pred = get_save_path(args.out_dir, identifier + "_PRED.npy", '+'.join(channel_names))

        # If not --overwrite set, and path exists, we skip it here
        if os.path.exists(path_pred) and not args.overwrite:
            logger.info(f"Skipping (channels={channel_group.channel_names}) - already exists and --overwrite not set.")
            # Load and increment the majority_voted array before continue
            majority_voted = get_updated_majority_voted(majority_voted, np.load(path_pred))
            continue

        # Get the prediction and true values
        pred = run_pred_on(
            study=study,
            channel_group=channel_group,
            model=model,
            model_external=model_external,
            args=args,
            hparams=hparams
        )

        # Sum the predictions into the majority_voted array
        majority_voted = get_updated_majority_voted(majority_voted, pred)

        # Save prediction
        save_file(path_pred, arr=pred, argmax=not args.no_argmax)

    if args.majority:
        if not os.path.exists(path_mj) or args.overwrite:
            nch = (k - k_ext_model) + 1
            save_file(path_mj, arr=majority_voted/nch, argmax=not args.no_argmax)
        else:
            logger.info("Skipping (channels=MAJORITY) - already exists and --overwrite not set.")


def split_channel_types(channels):
    """
    TODO

    Args:
        channels: list of channel names

    Returns:
        stripped channels
        channel_types
    """
    stripped, types = [], []
    for channel in channels:
        type_ = None
        if "==" in channel:
            channel, type_ = channel.split("==")
            type_ = type_.strip().upper()
            if type_ not in VALID_CHANNEL_TYPES:
                raise ValueError(f"Invalid channel type '{type_}' specified for channel '{channel}'. "
                                 f"Valid are: {VALID_CHANNEL_TYPES}")
        types.append(type_)
        stripped.append(channel)
    return stripped, types


def unpack_channel_groups(channels):
    """
    TODO
    """
    channels_to_load, channel_groups = [], []
    grouped = map(lambda chan: "++" in chan, channels)
    if all(grouped):
        for channel in channels:
            group = channel.split("++")
            channels_to_load.extend(group)
            channel_groups.append(group)
        # Remove duplicated while preserving order
        from collections import OrderedDict
        channels_to_load = list(OrderedDict.fromkeys(channels_to_load))
    elif not any(grouped):
        channels_to_load = channels
        channel_groups = [channels]
    else:
        raise ValueError("Must specify either a list of channels "
                         "or a list of channel groups, got a mix: {}".format(channels))

    return channels_to_load, channel_groups


def strip_and_infer_channel_types(channels_to_load, channel_groups):
    """
    TODO

    Args:
        channels_to_load:
        channel_groups:

    Returns:

    """
    # Infer and strip potential channel types
    channels_to_load, channel_types = split_channel_types(channels_to_load)
    channel_groups = [split_channel_types(group)[0] for group in channel_groups]

    # Infer channel types, may not be specified by user
    # If user did not specify all channels, use inferred for those missing
    inferred_channel_types = infer_channel_types(channels_to_load)
    for i, (inferred, passed) in enumerate(zip(inferred_channel_types, channel_types)):
        if passed is None:
            channel_types[i] = inferred

    return channels_to_load, channel_groups, channel_types


def get_channel_groups(channels, channel_types, channel_group_spec):
    def upper_stripped(s):
        return s.strip().upper()
    channel_types = list(map(upper_stripped, channel_types))
    channel_group_spec = list(map(upper_stripped, channel_group_spec))
    if "MASTOID" in channel_group_spec:
        channel_group_spec.remove("MASTOID")
    if any([c not in channel_group_spec for c in channel_types]) or \
            any([c not in channel_types for c in channel_group_spec]):
        raise ValueError(f"Cannot get channel groups for spec {channel_group_spec} with channels "
                         f"{channels} and types {channel_types}: One or more types are not in the requested "
                         f"channel group spec or vice versa.")
    channels_by_group = [[] for _ in range(len(channel_group_spec))]
    for channel, type_ in zip(channels, channel_types):
        inds = np.where(np.asarray(channel_group_spec) == type_)[0]
        for ind in inds:
            channels_by_group[ind].append(channel)
    # Return all combinations except unordered duplicates ([[EEG 1, EEG 2], [EEG 2, EEG 1]] -> [[EEG 1, EEG 2]])
    combinations = get_channel_group_combinations(*channels_by_group, remove_unordered_duplicates=True)
    if len(combinations) == 0:
        raise NotImplementedError("Unexpected empty channel_groups list found. "
                                  "Please raise an issue on GitHub if you encounter this error.")
    return combinations


def get_load_and_group_channels(auto_channel_grouping,
                                auto_reference_types,
                                channels=None,
                                channels_in_file=None):
    """
    TODO

    Args:
        channels: list
        auto_channel_grouping: list
        auto_reference_types: list
        channels_in_file: list
        inferred_types: list

    Returns:

    """
    if channels is None and channels_in_file is None:
        raise ValueError("Must specify either 'channels' or 'channels_in_file' (recieved None for both)")
    logger.info(f"Processing input channels: {channels or 'AUTOMATIC (not set)'}")
    channels_to_load, channel_groups = unpack_channel_groups(channels or channels_in_file)
    channels_to_load, channel_groups, channel_types = strip_and_infer_channel_types(channels_to_load,
                                                                                    channel_groups)
    if not channels:
        # Keep only channels as per auto_channel_grouping
        channels_to_load, channel_types = zip(*[(chan, type_) for chan, type_ in
                                                zip(channels_to_load, channel_types) if type_ in auto_channel_grouping])
        channel_groups = [channels_to_load]
        logger.warning(f"--channels flag not set! Considering only the automatically inferred channels of "
                       f"types {auto_channel_grouping}: {channels_to_load} (types: {channel_types}) from the "
                       f"original full set of channels in the file: {channels_in_file}")

    logger.info(f"\nFound:\n"
                f"-- Load channels: {channels_to_load}\n"
                f"-- Groups: {channel_groups}\n"
                f"-- Types: {channel_types}")
    if isinstance(auto_reference_types, list):
        assert len(channel_groups) == 1, "Cannot use channel groups with --auto_reference_types."
        channels_to_load, channel_types = infer_channel_refs(channel_names=channels_to_load,
                                                             channel_types=channel_types,
                                                             types=auto_reference_types,
                                                             on_already_ref="warn")
        channel_groups = [channels_to_load]
        logger.warning(f"OBS: Auto referencing returned channels: {channels_to_load}")
    if isinstance(auto_channel_grouping, list):
        channel_groups = get_channel_groups(channels_to_load, channel_types, auto_channel_grouping)
        logger.warning(f"OBS: Auto channel grouping returned groups: {channel_groups} (required groups: {auto_channel_grouping})")

    # Add channel inds to groups
    channel_set = namedtuple("ChannelSet", ["channel_names", "channel_indices"])
    for i, group in enumerate(channel_groups):
        channel_groups[i] = channel_set(
            channel_names=group,
            channel_indices=[channels_to_load.index(channel) for channel in group]
        )

    return channels_to_load, channel_groups


def get_sleep_study(psg_path,
                    channels,
                    header_file_name=None,
                    auto_channel_grouping=False,
                    auto_reference_types=False,
                    **hparams):
    """
    Loads a specified sleep study object with no labels
    Sets scaler and quality control function

    Returns:
        A loaded SleepStudy object
    """
    # if hparams.get('batch_wise_scaling'):
    #     raise NotImplementedError("Batch-wise scaling is currently not "
    #                               "supported. Use ut predict/evaluate instead")
    # used_hparams = {key: hparams.get(key) for key in ("period_length", "time_unit", "strip_func",
    #                                                   "set_sample_rate", "scaler", "quality_control_func")}
    used_hparams = {key: hparams.get(key) for key in ("strip_func", "set_sample_rate",
                                                      "scaler", "quality_control_func")}
    period_length = (hparams.get("train_data") or
                     hparams.get("test_data") or
                     hparams).get('period_length', None)
    time_unit = (hparams.get("train_data") or
                 hparams.get("test_data") or
                 hparams).get('time_unit', "SECOND")
    logger.info(f"Evaluating using parameters:\n{pformat(used_hparams)}")
    dir_, regex = os.path.split(os.path.abspath(psg_path))
    study = SleepStudy(subject_dir=dir_,
                       psg_regex=regex,
                       header_regex=header_file_name,
                       no_hypnogram=True,
                       period_length=period_length,
                       time_unit=time_unit)

    file_header = extract_header(study.psg_file_path, study.header_file_path)

    channels_to_load, channel_groups = get_load_and_group_channels(auto_channel_grouping,
                                                                   auto_reference_types,
                                                                   channels=channels,
                                                                   channels_in_file=file_header['channel_names'])

    logger.info(f"\nLoading channels: {channels_to_load}\n"
                f"Channel groups: {channel_groups}")
    study.set_strip_func(**used_hparams['strip_func'])
    study.select_channels = channels_to_load
    study.sample_rate = used_hparams['set_sample_rate']
    study.scaler = used_hparams['scaler']
    study.set_quality_control_func(**used_hparams['quality_control_func'])
    study.load()
    logger.info(f"\nStudy loaded with shape: {study.get_psg_shape()}\n"
                f"Channels: {study.select_channels} (org names: {study.select_channels.original_names})")
    return study, channel_groups


def replace_hdeeg_names(input_list):
    # Count elements that start with 'E' followed by a number
    e_count = sum(1 for item in input_list if item.startswith("E") and item[1:].isdigit())

    # If there are more than 64 such elements, replace 'E' with 'EEG' in relevant items
    if e_count > 64:
        input_list = ["EEG" + item[1:] if item.startswith("E") and item[1:].isdigit() else item for item in input_list]

    return input_list


def run(args):
    """
    Run the script according to args - Please refer to the argparser.
    """
    args = get_processed_args(args)
    logger.info(f"Args dump: {vars(args)}")

    # Get a logger
    log_dir, log_file_name = os.path.split(os.path.abspath(args.logging_out_path))
    add_logging_file_handler(log_file_name, args.overwrite, log_dir=log_dir, mode="w")

    # Get hyperparameters and init all described datasets
    from utime.hyperparameters import YAMLHParams
    hparams = YAMLHParams(Defaults.get_hparams_path(args.project_dir), no_version_control=True)
    datasets = hparams.get('datasets')
    if datasets:
        path = list(datasets.values())[0]
        if not os.path.isabs(path):
            path = os.path.join(Defaults.get_hparams_dir(args.project_dir), path)
        hparams.update(
            YAMLHParams(path, no_version_control=True)
        )

    # Get the sleep study
    logger.info("Loading and pre-processing PSG file...")
    hparams['channels'] = args.channels
    hparams['strip_func']['strip_func'] = args.strip_func
    study, channel_groups = get_sleep_study(psg_path=args.f,
                                            header_file_name=args.header_file_name,
                                            auto_channel_grouping=args.auto_channel_grouping,
                                            auto_reference_types=args.auto_reference_types,
                                            **hparams)

    # Set GPU and get model
    find_and_set_gpus(args.num_gpus, args.force_gpus)
    hparams["build"]["data_per_prediction"] = args.data_per_prediction
    logger.info(f"Predicting with {args.data_per_prediction} data per prediction")

    model, model_func = None, None

    if not args.model_external:
        if args.one_shot:
            model = get_and_load_one_shot_model(
                n_periods=study.n_periods,
                project_dir=args.project_dir,
                hparams=hparams,
                weights_file_name=args.weights_file_name
            )
        else:
            model = get_and_load_model(
                project_dir=args.project_dir,
                hparams=hparams,
                weights_file_name=args.weights_file_name
            )

    if args.one_shot:
        logger.info("Predicting with one-shot model.")
    else:
        logger.info("Predicting with sliding window model.")

    predict_study(study=study,
                  model=model,
                  channel_groups=channel_groups,
                  model_external=args.model_external,
                  args=args,
                  hparams=hparams)

    # logger.info(f"Predicted shape: {pred.shape}")

    # pred_period_length_sec = Defaults.PERIOD_LENGTH_SEC
    # save_prediction(pred=pred,
    #                 out_path=args.o,
    #                 period_length_sec=pred_period_length_sec,
    #                 no_argmax=args.no_argmax)


def entry_func(args=None):
    # Parse command line arguments
    parser = get_argparser()
    args = parser.parse_args(args)
    run(args)


if __name__ == "__main__":
    entry_func()
