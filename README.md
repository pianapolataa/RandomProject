# Reinforcement Learning Car Simulator

This project is a web-based simulation and control platform for a reinforcement learning (RL)-trained car agent to follow a specified path. Users can:

- Run path-following simulations with different path types
- Adjust physical/environmental parameters in real time
- Retrain the agent with new configurations
- Manually drive the car and view reward feedback

The backend is powered by Stable-Baselines3 (SAC), and the frontend is a lightweight HTML/CSS/JS interface using Plotly for visualization.

---

## Features

### Core Functionality

- **Path Simulation**  
  Visualize the car's path-following behavior using a pre-trained SAC model on:
  - A sine wave path
  - A straight path
  - An alternating curve consisting of a combination of sine and straight patterns

- **Manual Control Mode**  
  Toggle into manual mode to drive the car using `W`, `A`, `S`, `D` keys. Real-time feedback and final reward score are displayed.

- **Retrain Agent**  
  After modifying simulation parameters, you can retrain the model with a single click. The interface remains responsive during training.

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

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
