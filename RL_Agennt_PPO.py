from d3rlpy.algos import DiscreteBC, DiscreteBCConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from d3rlpy.models.encoders import VectorEncoderFactory
import serial
import time
import csv
from d3rlpy.algos import PPO, PPOConfig
from d3rlpy.datasets import Episode

ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
time.sleep(1)  # Let Arduino reset

def get_states():
    if ser.in_waiting:
        line = ser.readline().decode('utf-8').strip()
        try:
            theta, theta_dot = line.split(',')
            angle = float(theta)
            angular_velocity = float(theta_dot)
            print(f"Angle: {angle}, Angular Velocity: {angular_velocity}") 
            return angle, angular_velocity
        except ValueError:
            print(f"Invalid data format: {line}")
            pass
    return None, None

def pwm_to_index(pwm):
    return int(pwm + 255)

def index_to_pwm(index):
    return int(index - 255)

def get_actuation(angle, angular_velocity, discrete_BC):
    observation = np.array([[angle, angular_velocity]], dtype=np.float32)
    predicted_action = discrete_BC.predict(observation)
    pwm = index_to_pwm(predicted_action[0])
    return pwm

config = DiscreteBCConfig(encoder_factory=VectorEncoderFactory(
        hidden_units=[512, 512], activation="relu"), batch_size=1, gamma=1, learning_rate=1e-4)

episode = Episode()
ppo_config = PPOConfig(
    actor_encoder_factory=VectorEncoderFactory(hidden_units=[512, 512], activation="relu"),
    critic_encoder_factory=VectorEncoderFactory(hidden_units=[512, 512], activation="relu"),
    learning_rate=3e-4,
    gamma=0.99,
    rollout_length=2048,   # how many steps before each update
    batch_size=64,
    n_epochs=10,           # PPO epochs per rollout
)
ppo = PPO(config=ppo_config, device='cpu', enable_ddp=False)
ppo.create_impl(observation_shape=(2,), action_size=511)
discrete_BC = DiscreteBC(config=config, device='cpu', enable_ddp=False)
discrete_BC.create_impl(observation_shape=(2,), action_size=511)
discrete_BC.load_model("bc_policy.pt")
ppo.impl.policy.q_func.encoder.load_state_dict(
    discrete_BC.impl.policy.q_func.encoder.state_dict()
)
try:
    while True:
        angle, angular_velocity = get_states()
        if angle is not None and angular_velocity is not None:
            pwm = get_actuation(angle, angular_velocity, discrete_BC)
            ser.write(f"{pwm}\n".encode('utf-8'))
except KeyboardInterrupt:
    print("KeyboardInterrupt detected. Sending 0 PWM to stop the motors.")
    ser.write(f"0\n".encode('utf-8'))
    ser.close()
