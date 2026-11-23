import numpy as np
from scipy.signal import welch
from mne.decoding import CSP
from preprocessing import load_processed_dataset

FREQUENCY_BANDS = {
    "mu": (8, 13),
    "beta": (13, 30),
}


def extract_bandpower(X, sfreq=160):
    """Extract log mu and beta bandpower from EEG trials."""
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 3:
        raise ValueError("X must have shape (trials, channels, samples)")

    if sfreq <= 0:
        raise ValueError("sfreq must be positive")

    frequencies, psd = welch(
        X, fs=sfreq, nperseg=min(256, X.shape[-1]), axis=-1
    )

    features = []

    for low_freq, high_freq in FREQUENCY_BANDS.values():
        mask = (frequencies >= low_freq) & (frequencies <= high_freq)

        if not np.any(mask):
            raise ValueError("Not enough frequency resolution for the selected band")

        band_power = np.trapezoid(
            psd[..., mask], x=frequencies[mask], axis=-1
        )

        log_power = np.log10(np.maximum(band_power, 1e-20))
        features.append(log_power)

    return np.concatenate(features, axis=1)


def fit_csp(X_train, y_train, n_components=4):
    """Fit CSP on training data only to avoid data leakage."""
    csp = CSP(
        n_components=n_components,
        reg=None,
        log=True,
        norm_trace=False,
    )

    csp.fit(X_train, y_train)
    return csp


def transform_csp(csp, X):
    """Apply fitted CSP to EEG trials."""
    return csp.transform(X)


def combine_features(bandpower_features, csp_features):
    """Combine bandpower and CSP features."""
    bandpower_features = np.asarray(bandpower_features)
    csp_features = np.asarray(csp_features)

    if bandpower_features.shape[0] != csp_features.shape[0]:
        raise ValueError("Feature arrays must contain the same number of trials")

    return np.concatenate([bandpower_features, csp_features], axis=1)


if __name__ == "__main__":
    X, y, groups = load_processed_dataset(subjects=[1, 2, 3])

    print("Original EEG shape:", X.shape)

    bandpower = extract_bandpower(X)
    print("Bandpower feature shape:", bandpower.shape)

    # Fit CSP on training subjects only
    train_mask = groups != 3

    csp = fit_csp(X[train_mask], y[train_mask])
    csp_features = transform_csp(csp, X[train_mask])

    print("CSP feature shape:", csp_features.shape)

    combined = combine_features(
        bandpower[train_mask],
        csp_features,
    )

    print("Combined feature shape:", combined.shape)
    print("Feature extraction test complete!")
