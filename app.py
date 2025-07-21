from flask import Flask, request, jsonify, send_from_directory
from stable_baselines3 import SAC
from path_env import CarPathEnv, generate_sine_path, generate_straight_path, generate_alternating_path
import numpy as np
import os

app = Flask(__name__, static_folder="static")

# Preload the model
model = SAC.load("car_path_sac_model_BEST.zip")
@app.route("/get_path", methods=["POST"])
def get_path():
    data = request.json
    path_type = data.get("path_type", "sine")

    if path_type == "sine":
        path = generate_sine_path()
        num_waypoints = 5
    elif path_type == "alternating":
        path = generate_alternating_path()
        num_waypoints = 10
    else:
        path = [(x, 0) for x in np.linspace(0, 50, 100)]
        num_waypoints = 5

    indices = np.linspace(0, len(path) - 1, num_waypoints, dtype=int)
    waypoints = [path[i] for i in indices]

    path_serializable = [[float(x), float(y)] for x, y in path]
    waypoints_serializable = [[float(x), float(y)] for x, y in waypoints]

    return jsonify({
        "path": path_serializable,
        "waypoints": waypoints_serializable
    })

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/simulate", methods=["POST"])
def simulate():
    data = request.json

    friction = data.get("friction", 0.7)
    path_type = data.get("path_type", "sine")
    gas_sensitivity = data.get("gas_sensitivity", 0.9)
    brake_sensitivity = data.get("brake_sensitivity", 0.4)
    steer_sensitivity = data.get("steer_sensitivity", 0.1)

    if path_type == "sine":
        path = generate_sine_path()
        num_waypoints = 5
    elif path_type == "alternating":
        path = generate_alternating_path()
        num_waypoints = 10
    else:
        path = [(x, 0) for x in np.linspace(0, 50, 100)]
        num_waypoints = 5

    indices = np.linspace(0, len(path) - 1, num_waypoints, dtype=int)
    waypoints = [path[i] for i in indices]

    env = CarPathEnv(
        path=path,
        friction=friction,
        gas_sensitivity=gas_sensitivity,
        brake_sensitivity=brake_sensitivity,
        steer_sensitivity=steer_sensitivity,
        random_start=False
    )
    obs, _ = env.reset()

    steps = 220
    if path_type == "alternating":
        steps = 1000
    elif path_type == "straight":
        steps = 150

    car_positions = []
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        car_positions.append([obs[0], obs[1]])
        if terminated or truncated:
            break

    return jsonify({
        "car": [[float(x), float(y)] for x, y in car_positions],
        "path": [[float(x), float(y)] for x, y in path],
        "waypoints": [[float(x), float(y)] for x, y in waypoints]
    })


if __name__ == "__main__":
    app.run(debug=True)
