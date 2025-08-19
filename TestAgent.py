from d3rlpy.algos import DiscreteBC, DiscreteBCConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from d3rlpy.models.encoders import VectorEncoderFactory
def pwm_to_index(pwm):
    return int(pwm + 255)

def index_to_pwm(index):
    return int(index - 255)

log_df = pd.read_csv("CleanedLogTest.csv")
log_df.columns = ['theta', 'theta_dot', 'PWM', 'Terminal']

config = DiscreteBCConfig(encoder_factory=VectorEncoderFactory(
        hidden_units=[512, 512], activation="relu"), batch_size=1, gamma=1, learning_rate=1e-4)

discrete_BC = DiscreteBC(config=config, device='cpu', enable_ddp=False)
observations = log_df[['theta', 'theta_dot']].values.astype(np.float32)
discrete_BC.create_impl(observation_shape=(2,), action_size=511)
discrete_BC.load_model("bc_policy.pt")
                  

predicted_actions = discrete_BC.predict(observations)
predicted_pwm = np.array([index_to_pwm(idx) for idx in predicted_actions])
actual_actions = log_df[['PWM']].values.astype(np.float32)
actual_actions = actual_actions.flatten()
x = np.arange(len(actual_actions))
plt.plot(x, actual_actions, label='Actual Actions', alpha=0.7)
plt.plot(x, predicted_pwm, label='Predicted Actions', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('PWM Value')
plt.title('Predicted vs Actual Actions')
plt.legend()
plt.show()