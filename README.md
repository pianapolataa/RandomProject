# Reinforcement Learning Car Simulator

This project is an interactive web-based simulation for a reinforcement learning (RL)-trained car agent to follow a specified path. 

---

## Features

- **Path Simulation**  
  Visualize the car's path-following behavior using a pre-trained SAC model on a sine wave path, a straight path, and an alternating curve consisting of a combination of sine and straight patterns

- **Parameter Tuning**
  Interactively adjust physical environment parameters such as friction and steering sensitivity and visualize their impact on the performance of the pre-trained model.
  
- **Retrain Agent**  
  After modifying simulation parameters, you can retrain the model with a single click. The interface remains responsive during training.

- **Manual Control Mode**  
  Toggle into manual mode to drive the car using `W`, `A`, `S`, `D` keys. Real-time feedback and final reward score are displayed, allowing for comparison with the model's performance.

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```
---

### 2. Run the Server

```bash
python app.py
```

Then open your browser and go to:

```
http://localhost:5000
```

This will launch the interactive simulation interface.

---
## Dependencies

This project uses:

- **Flask** – for the backend server  
- **gymnasium** – custom OpenAI Gym-compatible environment  
- **numpy**, **matplotlib** – math and plotting utilities  
- **stable-baselines3[sac]** – reinforcement learning algorithm  
- *(optional)* **torch** and **tensorboard** – for model training and debugging  

---

## UI Controls

- **Parameter Sliders**:
  - Friction
  - Gas Sensitivity
  - Brake Sensitivity
  - Steer Sensitivity

- **Path Type Selector**:
  - Choose between sine, straight, or alternating paths

- **Buttons**:
  - `Simulate` – Run an automated simulation
  - `Manual Mode` – Switch to manual driving
  - `Reset` – Clear current simulation and restart
  - `Reset to Defaults` – Restore default parameters
  - `Retrain Agent` – Retrain with new parameters

- **Theme Toggle**:
  - Switch between light and dark mode

---
 
## Keyboard Controls (Manual Mode)

Use the following keys while in **Manual Mode**:

- **W** – Accelerate  
- **S** – Brake  
- **A** – Steer Left  
- **D** – Steer Right  

You’ll receive a reward score when the episode ends.
