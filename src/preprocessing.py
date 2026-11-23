import numpy as np
import mne
from pathlib import Path
from mne.datasets import eegbci

SUBJECTS = range(1, 110)
RUNS = list(range(3, 15))

CLASS_NAMES = [
    "Rest",
    "Left Fist",
    "Right Fist",
    "Both Fists",
    "Both Feet",
]

CLASS_LABELS = {
    "Rest": 0,
    "Left Fist": 1,
    "Right Fist": 2,
    "Both Fists": 3,
    "Both Feet": 4,
}

LEFT_RIGHT_RUNS = {3, 4, 7, 8, 11, 12}
BOTH_RUNS = {5, 6, 9, 10, 13, 14}

LOW_FREQ = 8
HIGH_FREQ = 30
EPOCH_DURATION = 2.0

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


def get_event_labels(run):
    """Map EEG event markers to movement classes."""
    if run in LEFT_RIGHT_RUNS:
        return {
            "T0": CLASS_LABELS["Rest"],
            "T1": CLASS_LABELS["Left Fist"],
            "T2": CLASS_LABELS["Right Fist"],
        }

    if run in BOTH_RUNS:
        return {
            "T0": CLASS_LABELS["Rest"],
            "T1": CLASS_LABELS["Both Fists"],
            "T2": CLASS_LABELS["Both Feet"],
        }

    raise ValueError(f"Unsupported run: {run}")


def preprocess_run(subject, run):
    """Load, filter, and extract EEG trials from one recording."""
    print(f"Processing subject {subject}, run {run}")

    files = eegbci.load_data(subject, [run])
    raw = mne.io.read_raw_edf(
        files[0], preload=True, verbose=False
    )

    eegbci.standardize(raw)

    montage = mne.channels.make_standard_montage("colin27_1005")
    raw.set_montage(montage, on_missing="warn")

    raw.filter(
        l_freq=LOW_FREQ,
        h_freq=HIGH_FREQ,
        verbose=False,
    )

    events, event_id = mne.events_from_annotations(
        raw, verbose=False
    )

    label_mapping = get_event_labels(run)

    selected_events = {
        name: event_id[name]
        for name in label_mapping
        if name in event_id
    }

    if len(selected_events) != len(label_mapping):
        raise ValueError(
            f"Missing event annotations for subject {subject}, run {run}"
        )

    epochs = mne.Epochs(
        raw,
        events,
        event_id=selected_events,
        tmin=0,
        tmax=EPOCH_DURATION - 1 / raw.info["sfreq"],
        baseline=None,
        preload=True,
        verbose=False,
    )

    X = epochs.get_data()

    event_to_class = {
        event_id[name]: label_mapping[name]
        for name in selected_events
    }

    y = np.array(
        [event_to_class[event] for event in epochs.events[:, 2]],
        dtype=np.int64,
    )

    return X, y


def preprocess_dataset(subjects=SUBJECTS, runs=RUNS):
    """Process and save each participant separately."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    completed = 0
    failed = []

    for subject in subjects:
        output_path = OUTPUT_DIR / f"subject_{subject:03d}.npz"

        if output_path.exists():
            print(f"Subject {subject} already processed, skipping.")
            completed += 1
            continue

        subject_X = []
        subject_y = []
        failed_runs = []

        for run in runs:
            try:
                X, y = preprocess_run(subject, run)

                if len(y) == 0:
                    failed_runs.append(run)
                    print(f"No trials for subject {subject}, run {run}")
                    continue

                subject_X.append(X)
                subject_y.append(y)

                print(f"  Extracted {len(y)} trials")

            except (OSError, ValueError, RuntimeError) as error:
                failed_runs.append(run)
                print(f"Error in subject {subject}, run {run}: {error}")

        if failed_runs:
            failed.append((subject, failed_runs))
            print(
                f"Subject {subject} incomplete. "
                f"Failed runs: {failed_runs}"
            )
            continue

        if not subject_X:
            failed.append((subject, list(runs)))
            continue

        X = np.concatenate(subject_X, axis=0)
        y = np.concatenate(subject_y, axis=0)
        groups = np.full(len(y), subject, dtype=np.int64)

        np.savez_compressed(
            output_path,
            X=X,
            y=y,
            groups=groups,
            class_names=np.array(CLASS_NAMES),
            sfreq=np.array(160),
        )

        completed += 1
        print(f"Saved subject {subject}: {len(y)} trials\n")

    print("\nPreprocessing complete!")
    print("Participants saved:", completed)
    print("Participants with failed runs:", len(failed))

    for subject, failed_runs in failed:
        print(f"Subject {subject}: failed runs {failed_runs}")


def load_processed_dataset(subjects=SUBJECTS):
    """Load saved participant files into arrays for model training."""
    all_X = []
    all_y = []
    all_groups = []

    for subject in subjects:
        path = OUTPUT_DIR / f"subject_{subject:03d}.npz"

        if not path.exists():
            continue

        with np.load(path) as data:
            all_X.append(data["X"])
            all_y.append(data["y"])
            all_groups.append(data["groups"])

    if not all_X:
        raise FileNotFoundError("No processed EEG files found.")

    return (
        np.concatenate(all_X, axis=0),
        np.concatenate(all_y, axis=0),
        np.concatenate(all_groups, axis=0),
    )


if __name__ == "__main__":
    print("Starting EEG preprocessing...\n")
    preprocess_dataset()
