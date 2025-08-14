import numpy as np
import pandas as pd
from d3rlpy.datasets import MDPDataset
from d3rlpy.algos import BC, BCConfig, DiscreteBC, DiscreteBCConfig

def pwm_to_index(pwm):
    return int(pwm + 255)

def index_to_pwm(index):
    return int(index - 255)

def build_mdp_dataset_from_df(log_df, terminal_col='Terminal', done_str='done'):
    """
    Builds a d3rlpy MDPDataset from a pandas DataFrame of logged data using
    vectorized operations to create consistent data arrays.

    Args:
        log_df (pd.DataFrame): The input DataFrame containing logged observations,
                               actions, and terminal state information.
        terminal_col (str, optional): The name of the column indicating a terminal state.
                                      Defaults to 'Terminal'.
        done_str (str, optional): The string value that signifies a terminal state.
                                  Defaults to 'done'.

    Returns:
        d3rlpy.MDPDataset: A dataset object ready for offline reinforcement learning.
    """
    # Extract numpy arrays for observations and actions
    # Observations are the angle (theta) and angular velocity (theta_dot)
    observations = log_df[['theta', 'theta_dot']].values.astype(np.float32)
    # Actions are the PWM signal
    log_df['PWM_index'] = log_df['PWM'].apply(pwm_to_index)
    actions = log_df['PWM_index'].values.astype(np.int64).reshape(-1, 1)
    # Use actions in MDPDataset
    
    # Reward design: penalize angle and action, and a large negative reward on terminal steps
    k_angle, k_action = 1.0, 0.01
    rewards = -1*(k_angle * np.abs(log_df['theta'].values))
    
    # Identify terminal steps for episode segmentation
    terminals = (log_df[terminal_col] == done_str).values

    # Override the reward for terminal steps with a very large penalty
    rewards[terminals] = -1000.0
    
    # The `d3rlpy` library expects a `timeouts` array. Since the log only has
    # terminal states and no explicit timeouts, we can set this to all False.
    timeouts = np.zeros_like(terminals, dtype=bool)

    # Create and return the MDPDataset with the flat data arrays.
    # The `next_observations` array and keyword argument have been removed
    # to be compatible with your d3rlpy version.
    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts
    )
    return dataset

# --- Main Script Execution ---
# Load the data from the CSV file
log_df = pd.read_csv("CleanedLog.csv")
log_df.columns = ['theta', 'theta_dot', 'PWM', 'Terminal']

# Build the MDP dataset from the DataFrame
dataset = build_mdp_dataset_from_df(log_df)
config = DiscreteBCConfig(batch_size=256, gamma=1, learning_rate=1e-4)

# Initialize the Behavior Cloning (BC) agent with a learning rate
bc = DiscreteBC(config, device='cpu', enable_ddp=False)

# Fit the model to the dataset
# This trains the agent to mimic the actions in the log data
bc.fit(dataset, n_steps=10000)

# Save the trained policy model to a file
bc.save_model("bc_policy.pt")
