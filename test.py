import numpy as np

# Assuming `foot_indices` and `durations` are lists or arrays with appropriate dimensions.
foot_indices = [np.random.rand(1) for _ in range(4)]  # Example input
durations = np.random.rand(1)  # Example input

# Concatenate foot_indices and apply remainder operation
print(foot_indices)
self_foot_indices = np.remainder(np.column_stack([foot_indices[i] for i in range(4)]), 1.0)
print(self_foot_indices)
print(np.remainder(self_foot_indices, 1.0))

# Iterate over each set of indices
for idxs in foot_indices:
    stance_idxs = np.remainder(idxs, 1) < durations
    swing_idxs = np.remainder(idxs, 1) > durations

    idxs[stance_idxs] = np.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
    idxs[swing_idxs] = 0.5 + (np.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
        0.5 / (1 - durations[swing_idxs]))

