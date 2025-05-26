#%% Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('default')
%matplotlib tk

#%% Define Functions
def segment_trials(data: pd.DataFrame) -> list:
    """
    Segment the data into trials based on stimulus onset.
    """
    # Identify indices for stimulus onset transitions
    stimulus_on_idx = np.where(np.diff(data['stimulus_on'].to_numpy(), prepend=0) > 0)[0]
    trials = [
        data.iloc[stimulus_on_idx[i] : (stimulus_on_idx[i+1] if i < len(stimulus_on_idx)-1 else None)]
        for i in range(len(stimulus_on_idx))
    ]
    print(f'Number of trials: {len(trials)}')
    return trials

def group_trials_by_orientation(trials: list) -> dict:
    """
    Group trials by orientation and stack related arrays.
    """
    grouped = {}
    for trial in trials:
        orientation = trial['orientation'].iloc[0]
        if orientation not in grouped:
            grouped[orientation] = []
        entry = {
            'time': trial['time (s)'].to_numpy(),
            'stimulus_status': trial['stimulus_on'].to_numpy(),
            'neuron_responses': trial.drop(columns=['time (s)', 'stimulus_on', 'orientation']).to_numpy()
        }
        grouped[orientation].append(entry)
    
    print(f'Number of orientations: {len(grouped)}')
    for orient, entries in grouped.items():
        times = np.stack([entry['time'] for entry in entries], axis=0)
        statuses = np.stack([entry['stimulus_status'] for entry in entries], axis=0)
        responses = np.stack([entry['neuron_responses'] for entry in entries], axis=0)
        grouped[orient] = {'time': times, 'stimulus_status': statuses, 'neuron_responses': responses}
    return grouped

def compute_average_response(grouped: dict) -> tuple:
    """
    Compute average neuron responses for each orientation.
    Returns: orientation_vec, mean_responses_arr, weighted_mean_vec,
             preferred_orientation, and orientation_selectivity.
    """
    orientation_vec = []
    mean_responses_arr = []
    for orient, group in grouped.items():
        orientation_vec.append(np.exp(1j * np.deg2rad(float(orient))))
        responses_arr = group['neuron_responses']
        responses_on_mean = np.nanmean(responses_arr[:, group['stimulus_status'][0] == 1, :], axis=1)
        responses_off_mean = np.nanmean(responses_arr[:, group['stimulus_status'][0] == 0, :], axis=1)
        responses_amplitude_mean = np.nanmean(responses_on_mean - responses_off_mean, axis=0)
        mean_responses_arr.append(responses_amplitude_mean)

    orientation_vec = np.array(orientation_vec)
    mean_responses_arr = np.array(mean_responses_arr)
    weighted_mean_vec = orientation_vec[None, :] @ (mean_responses_arr - np.nanmin(mean_responses_arr, axis=0, keepdims=True))
    preferred_orientation = np.rad2deg(np.angle(weighted_mean_vec))
    orientation_selectivity = np.abs(weighted_mean_vec) / np.nansum(mean_responses_arr, axis=0)
    return orientation_vec, mean_responses_arr, weighted_mean_vec, preferred_orientation, orientation_selectivity

def plot_heatmaps(grouped: dict) -> None:
    """
    Plot neuron responses as heat maps for each orientation.
    """
    orientations = sorted(grouped.keys())
    n_orient = len(orientations)
    fig = plt.figure(1)
    fig.clf()
    axes = fig.subplots(1, n_orient) if n_orient > 1 else [fig.subplots()]

    for ax, orient in zip(axes, orientations):
        d = grouped[orient]
        responses = d['neuron_responses']
        n_trials, n_time_steps, n_neurons = responses.shape
        
        heatmap_rows = []
        for neuron in range(n_neurons):
            neuron_block = []
            for trial in range(n_trials):
                neuron_block.append(responses[trial, :, neuron])
                if trial == n_trials - 1:
                    neuron_block.append(np.full((n_time_steps,), np.nan))
            heatmap_rows.append(np.vstack(neuron_block))
            heatmap_rows.append(np.full((1, n_time_steps), np.nan))
        heatmap = np.vstack(heatmap_rows[:-1])  # Remove extra nan row

        time_array = d['time'][0]
        time_array -= time_array[0]  # Normalize time to start at 0
        ax.imshow(heatmap, extent=[time_array[0], time_array[-1], 0, heatmap.shape[0]],
                  aspect='auto', interpolation='none')
        
        trial_status = d['stimulus_status'][0]
        offset_idx = np.where(trial_status == 0)[0]
        if offset_idx.size:
            ax.axvline(x=time_array[offset_idx[0]], color='red', linestyle='--')
        
        if ax == axes[0]:
            offset = 0
            y_ticks, y_labels = [], []
            for neuron in range(n_neurons):
                block_len = n_trials + 2 if neuron != n_neurons - 1 else n_trials + 1
                tick = offset + block_len / 2 - 0.5
                y_ticks.append(tick)
                y_labels.append(f"Neuron {n_neurons - neuron}")
                offset += n_trials + 2
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels)
            ax.set_title(f'Orientation: {orient}')
        else:
            ax.set_yticks([])
            ax.set_title(str(orient))
        ax.set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.show()

#%% Main Execution
if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv('visual_neuroscience_dummy_data_4min.csv')
    
    # Process data into trials and group by orientation
    trials = segment_trials(data)
    grouped = group_trials_by_orientation(trials)
    
    # Compute average responses (results available if needed)
    compute_average_response(grouped)
    
    # Plot neuron heatmaps
    plot_heatmaps(grouped)