#%% Define Functions
import pandas as pd
data = pd.read_csv('visual_neuroscience_dummy_data_4min.csv')


# A simple way to get trials from up there
import numpy as np
RAD2ANGLE = np.pi / 180
stimulus_on_idx = np.where(np.diff(data['stimulus_on'].to_numpy(), prepend=0) > 0)[0]
trials = [
    data.iloc[stimulus_on_idx[i] : (stimulus_on_idx[i+1] if i < len(stimulus_on_idx)-1 else None)]
    for i in range(len(stimulus_on_idx))
]


# Get them grouped!
def groupping(trials):
    """
    Group trials by orientation and stack related arrays.
    """
    grouped = {}
    for temp in trials:
        orientation = temp['orientation'].iloc[0]
        if orientation not in grouped:
            grouped[orientation] = []
        
        appendix_col = ['time (s)', 'stimulus_on', 'orientation']
        
        entry = {
            'time(s)': temp[appendix_col[0]].to_numpy(),
            'stimulus_status': temp[appendix_col[1]].to_numpy(),
            'neuron_responses': temp.drop(columns=['time (s)', 'stimulus_on', 'orientation']).to_numpy()
        }
        grouped[orientation].append(entry)
    
    return grouped
grouped = groupping(trials)

# Concate arrays
for k, v in grouped.items():
    times = np.stack([entry['time(s)'] for entry in v], axis=0)
    statuses = np.stack([entry['stimulus_status'] for entry in v], axis=0)
    responses = np.stack([entry['neuron_responses'] for entry in v], axis=0)
    grouped[k] = {'time(s)': times, 'stimulus_status': statuses, 'neuron_responses': responses}

O_vec = []
mean_responses_arr = []
for k, v in grouped.items():
    O_vec.append(np.exp(1j * np.deg2rad(float(k))))
    data = v['neuron_responses']
    on_avg = np.nanmean(data[:, v['stimulus_status'][0] == 1, :], axis=1)
    off_mean = np.nanmean(data[:, v['stimulus_status'][0] == 0, :], axis=1)
    amp = np.nanmean(on_avg - off_mean, axis=0)
    mean_responses_arr.append(amp)

weightedVec = np.array(O_vec)[None, :] @ (np.array(mean_responses_arr) - np.min(np.array(mean_responses_arr), axis=0, keepdims=True))
oFunc = lambda v: np.angle(v)/RAD2ANGLE
pref_Ori = oFunc(weightedVec)
orientation_selectivity = np.abs(weightedVec) / np.sum(np.array(mean_responses_arr), axis=0)

orientations = list(grouped.keys())

# Now plot now
ON_TIME = 2
plot_offset = 0

import matplotlib.pyplot as plt
%matplotlib tk

fig = plt.figure(1)
plt.style.use('default')
fig.clf()

axes = fig.subplots(1, len(orientations)) if len(orientations) > 1 else [fig.subplots()]
for ax, k in zip(axes, orientations):
    d = grouped[k]
    responses = d['neuron_responses']
    n_trials, n_time_steps, n_neurons = responses.shape
    
    heatmap_rows = []
    for neuron in range(responses.shape[2]):
        neuron_block = []
        for temp in range(n_trials):
            neuron_block.append(responses[temp, :, neuron])
            if temp == n_trials - 1:
                neuron_block.append(np.full((n_time_steps,), np.nan))
        heatmap_rows.append(np.vstack(neuron_block))
        heatmap_rows.append(np.full((1, responses.shape[1]), np.nan))
    heatmap = np.vstack(heatmap_rows[:-1])  # Remove extra nan row

    time_array = d['time(s)'][0] - d['time(s)'][0][0]
    ax.imshow(heatmap, extent=[time_array[0], time_array[-1], 0, heatmap.shape[0]],
                aspect='auto', interpolation='none')
    
    trial_status = d['stimulus_status'][0]
    offset_idx = np.where(trial_status == 0)[0]
    if offset_idx.size:
        ax.axvline(x=ON_TIME, color='red', linestyle='--')
    
    if ax == axes[0]:
        y_ticks, y_labels = [], []
        for neuron in range(n_neurons):
            block_len = n_trials + 2 if neuron != n_neurons - 1 else n_trials + 1
            tick = plot_offset + block_len / 2 - 0.5
            y_ticks.append(tick)
            y_labels.append(f"Neuron {n_neurons - neuron}")
            plot_offset += n_trials + 2
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_title(f'Orientation: {k}')
    else:
        ax.set_yticks([])
        ax.set_title(str(k))
    ax.set_xlabel('Time (s)')

plt.tight_layout()
plt.show()
# %%
