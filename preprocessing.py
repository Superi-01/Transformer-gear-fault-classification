import os
import pandas as pd
import numpy as np

def load_mcc5thu_data(
    data_dir,
    state_list,
    window_size=2000,
    num_samples=1000,
    rpm_range=(1500, 2500),
    time_range=(0, 10),
    signal_col=['gearbox_vibration_x', 'gearbox_vibration_y', 'gearbox_vibration_z'],
    fault_level='H',
    mode_name='speed',
    torque_choice='20Nm',
    speed_choice='3000rpm'
):
    X_all, y_all, d_all = [], [], []

    for i, fault_name in enumerate(state_list):
        # Construct file name based on fault type
        if fault_name == 'health' or fault_name == 'miss_teeth':
            file_name = f'{fault_name}_{mode_name}_circulation_{torque_choice}-{speed_choice}.csv'        
        else:
            file_name = f'{fault_name}_{fault_level}_{mode_name}_circulation_{torque_choice}-{speed_choice}.csv'

        file_path = os.path.join(data_dir, file_name)

        print(f"[{i}] Loading: {file_path}")
        try:
            X, freq_label = load_mcc5thu_data_sliding_window(
                file_path=file_path,
                window_size=window_size,
                num_samples=num_samples,
                rpm_range=rpm_range,
                time_range=time_range,
                signal_col=signal_col
            )
        except Exception as e:
            print(f"❌ Failed to load {fault_name}: {e}")
            continue

        # Classification label (0, 1, 2, ...)
        y_label = np.full(len(X), i)

        # Append to the main lists
        X_all.append(X)
        y_all.append(y_label)
        d_all.append(freq_label)

    # Concatenate all data
    X_data = np.concatenate(X_all, axis=0)
    Y_data = np.concatenate(y_all, axis=0)
    D_data = np.concatenate(d_all, axis=0)

    return X_data, Y_data, D_data


def load_mcc5thu_data_sliding_window(
    file_path,
    window_size=2000,
    num_samples=1000,
    rpm_range=(1500, 2500),
    time_range=(0, 10),
    signal_col=['gearbox_vibration_x', 'gearbox_vibration_y', 'gearbox_vibration_z']
):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Estimate RPM using the 'speed' column
    speed_signal = df['speed'].copy()
    speed_signal[speed_signal <= 2] = 0
    speed_signal[speed_signal > 2] = 1
    rising_edges_index = np.where(np.diff(speed_signal) == 1)[0]

    # Calculate the time vector (60 seconds duration)
    time = np.linspace(0, 60, len(speed_signal))
    rising_time_point = time[rising_edges_index]
    period = np.diff(rising_time_point)
    frequency = 1 / period  # Frequency in Hz
    speed_rpm = frequency * 60
    mean_time_point = pd.Series(rising_time_point).rolling(window=2).mean().dropna().to_numpy()

    # Filter based on time and RPM range conditions
    valid_mask = (
        (mean_time_point >= time_range[0]) & 
        (mean_time_point <= time_range[1]) & 
        (speed_rpm >= rpm_range[0]) & 
        (speed_rpm <= rpm_range[1])
    )
    valid_times = mean_time_point[valid_mask]
    valid_freqs = frequency[valid_mask]  # Frequency in Hz

    # Extract the signal and calculate the sampling rate (60 seconds duration)
    signal = df[signal_col].values
    sampling_rate = len(signal) / 60  # Samples per second
    sample_indices = (valid_times * sampling_rate).astype(int)
    
    # Uniformly sample the specified number of samples
    selected_indices = np.linspace(0, len(sample_indices) - 1, num_samples, dtype=int)
    sampled_indices = sample_indices[selected_indices]
    sampled_times = valid_times[selected_indices]
    sampled_freqs = valid_freqs[selected_indices]

    def get_windows_and_labels(start_indices, times, freqs):
        X, D = [], []
        for idx in start_indices:
            s = idx
            e = s + window_size
            if e <= len(signal):
                window = signal[s:e].T
                start_time = s / sampling_rate
                end_time = e / sampling_rate

                # Calculate the average frequency in the window
                mask = (times >= start_time) & (times <= end_time)
                avg_freq = np.mean(freqs[mask]) if np.any(mask) else 0.0

                X.append(window)
                D.append(avg_freq)
                
        return np.array(X), np.array(D)

    # Generate windowed samples and labels
    X_total, d_label = get_windows_and_labels(sampled_indices, sampled_times, sampled_freqs)

    return X_total, d_label