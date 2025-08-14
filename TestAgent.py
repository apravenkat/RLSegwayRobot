from d3rlpy.algos import DiscreteBC, DiscreteBCConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

log_df = pd.read_csv("CleanedLogTest.csv")
log_df.columns = ['theta', 'theta_dot', 'PWM', 'Terminal']

config = DiscreteBCConfig(batch_size=1024, gamma=0.99, learning_rate=1e-4)

discrete_BC = DiscreteBC(config=config, device='cpu', enable_ddp=False)
observations = log_df[['theta', 'theta_dot']].values.astype(np.float32)
discrete_BC.create_impl(observation_shape=(2,), action_size=256)
discrete_BC.load_model("bc_policy.pt")
                  

predicted_actions = discrete_BC.predict(observations)
actual_actions = log_df[['PWM']].values.astype(np.float32)
actual_actions = actual_actions.flatten()
print(actual_actions)
x = np.arange(len(actual_actions))
differences = predicted_actions - actual_actions
plt.plot(x, differences)
plt.show()