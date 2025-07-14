from flask import Flask, request, jsonify, send_from_directory
from stable_baselines3 import SAC
from path_env import CarPathEnv, generate_sine_path
import numpy as np
import os

app = Flask(__name__, static_folder="static")

# Preload the model
model = SAC.load("car_path_sac_model.zip")

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/simulate", methods=["POST"])
def simulate():
    data = request.json

    # === Extract parameters from request ===
    friction = data.get("friction", 0.7)
    path_type = data.get("path_type", "sine")
    gas_sensitivity = data.get("gas_sensitivity", 0.9)
    brake_sensitivity = data.get("brake_sensitivity", 0.4)
    steer_sensitivity = data.get("steer_sensitivity", 0.1)

    # === Generate path ===
    if path_type == "sine":
        path = generate_sine_path()
    else:
        path = [(x, 0) for x in np.linspace(0, 50, 100)]

    # === Create environment with all parameters ===
    env = CarPathEnv(
        path=path,
        friction=friction,
        gas_sensitivity=gas_sensitivity,
        brake_sensitivity=brake_sensitivity,
        steer_sensitivity=steer_sensitivity,
        random_start=False  # ← always starts at (0, 0)
    )
    obs, _ = env.reset()

    car_positions = []
    for _ in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        car_positions.append([obs[0], obs[1]])
        if terminated or truncated:
            break

    # === Convert to JSON-safe types ===
    car_serializable = [[float(x), float(y)] for x, y in car_positions]
    path_serializable = [[float(x), float(y)] for x, y in path]

    return jsonify({
        "car": car_serializable,
        "path": path_serializable
    })

if __name__ == "__main__":
    app.run(debug=True)
