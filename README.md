# Warm-Starting RL Agents with a PD Controller on a Self-Balancing Robot  

## Motivation  
Reinforcement learning (RL) in the real world is hard. Action spaces are huge, and exploration in simulation or with real hardware is often too expensive and time-consuming. For many tasks, a simple control method can provide a good starting point instead of learning everything from scratch.  

This project explores using a **PD controller to warm-start an RL agent** for a self-balancing robot, and then improving performance through **online learning**. I’m still experimenting and learning—so expect mistakes along the way 🙂.  

## Hardware  
- Robot: [ELEGOO Tumbller Self-Balancing Car](https://www.amazon.com/ELEGOO-Tumbller-Self-Balancing-Engineering-Compatible/dp/B07QWJH77V)  
- Microcontroller: Arduino (provided with kit)  

The base controller code comes from the [ELEGOO Tumbller Tutorial Repository](https://github.com/elegooofficial/ELEGOO-TumbllerV1.1-Self-Balancing-Car-Tutorial).  

## Repository Structure  

### Arduino  
- **`BalancedCar.ino`**  
  Flash this onto the Arduino to run the PD controller.  
  - Set `MODE` in `Balanced.h` to:
    - `0` → PD controller + data logging  
    - `1` → RL agent actuation  



### Python  
- **`LogData.py`**  
  Uses Serial communication to log:  
  - Robot angle  
  - Angular velocity  
  - Motor PWM actuation  
  Logged data is stored in a CSV file while the robot balances using the PD controller.  

- **`DataProcessing.py`**  
  Processes a saved log file and calculates rewards for each state.  

- **`TrainAgent.py`**  
  Trains a discrete Behavior Cloning (BC) RL agent using the logged data.  
  > ⚡ Tip: Use a GPU for faster training.  

- **`TestAgent.py`**  
  Tests the trained RL agent in software to ensure actuations are reasonable before deployment to the Arduino.  

- **`RL_Agent_Arduino.py`**  
  Runs the trained RL agent and sends actuation commands to the Arduino over Serial.  

## Workflow  
1. Run **PD controller** (`BalancedCar.ino`) to balance the robot.  
2. Use **`LogData.py`** to record state-action data.  
3. Process logs with **`DataProcessing.py`** to assign rewards.  
4. Train the RL agent using **`TrainAgent.py`**.  
5. Validate with **`TestAgent.py`**.  
6. Deploy the RL agent with **`RL_Agent_Arduino.py`**.  

## Dependencies & Installation

### Create a conda environment
```bash
# Create new conda environment
conda create -n segway-rl python=3.10 -y

# Activate environment
conda activate segway-rlconda install numpy pandas matplotlib -y
conda install -c conda-forge pyserial -y
pip install d3rlpy torch

## Disclaimer  
This is an experimental project for learning and exploration. Expect bugs, inefficiencies, and plenty of iteration 🚀.  

