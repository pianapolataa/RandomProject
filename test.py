import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from path_env import CarPathEnv
import numpy as np
import time

# Load environment and model
env = CarPathEnv(random_start=False)  # ← always starts at (0, 0)
model = SAC.load("car_path_sac_model_BEST.zip")
obs, _ = env.reset()

# Extract the path to plot
path_x = [p[0] for p in env.path]
path_y = [p[1] for p in env.path]

# Setup the plot
plt.ion()
fig, ax = plt.subplots()
ax.plot(path_x, path_y, 'k--', label='Target Path')
car_point, = ax.plot([], [], 'ro', label='Car')
ax.set_xlim(min(path_x) - 5, max(path_x) + 5)
ax.set_ylim(min(path_y) - 10, max(path_y) + 10)
ax.set_title("Car Following Sine Path")
ax.legend()

# Simulation loop
for _ in range(300):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)

    # Update car position
    car_x, car_y = obs[0], obs[1]
    car_point.set_data([car_x], [car_y])
    plt.pause(0.01)
    print(f"Action: {action}, Position: ({car_x:.2f}, {car_y:.2f}), Speed: {obs[3]:.2f}")

    if terminated or truncated:
        break

plt.ioff()
plt.show()

