from flask import Flask, request, jsonify, send_from_directory
from stable_baselines3 import SAC
from path_env import CarPathEnv, generate_sine_path, generate_straight_path, generate_alternating_path
import numpy as np
import os

manual_env = None  # Global environment instance for manual control
manual_episode_return = 0.0  # Track cumulative reward for manual mode

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
    total_penalty = 0.0
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_penalty += reward  # Accumulate rewards (penalty)
        car_positions.append([obs[0], obs[1]])
        if terminated or truncated:
            break

    return jsonify({
        "car": [[float(x), float(y)] for x, y in car_positions],
        "path": [[float(x), float(y)] for x, y in path],
        "waypoints": [[float(x), float(y)] for x, y in waypoints],
        "penalty": float(total_penalty)
    })


@app.route("/manual_start", methods=["POST"])
def manual_start():
    global manual_env, manual_episode_return
    data = request.json
    friction = data.get("friction", 0.7)
    path_type = data.get("path_type", "sine")
    gas_sensitivity = data.get("gas_sensitivity", 0.9)
    brake_sensitivity = data.get("brake_sensitivity", 0.4)
    steer_sensitivity = data.get("steer_sensitivity", 0.1)

    if path_type == "sine":
        path = generate_sine_path()
    elif path_type == "alternating":
        path = generate_alternating_path()
    else:
        path = [(x, 0) for x in np.linspace(0, 50, 100)]

    manual_env = CarPathEnv(
        path=path,
        friction=friction,
        gas_sensitivity=gas_sensitivity,
        brake_sensitivity=brake_sensitivity,
        steer_sensitivity=steer_sensitivity,
        random_start=False
    )
    obs, _ = manual_env.reset()
    manual_episode_return = 0.0  # Reset cumulative reward

    return jsonify({
        "observation": obs.tolist(),
        "path": [[float(x), float(y)] for x, y in path]
    })


@app.route("/manual_step", methods=["POST"])
def manual_step():
    global manual_env, manual_episode_return
    if manual_env is None:
        return jsonify({"error": "Manual environment not initialized. Call /manual_start first."}), 400

    data = request.json
    gas = float(data.get("gas", 0.0))
    brake = float(data.get("brake", 0.0))
    steer = float(data.get("steer", 0.0))

    gas = np.clip(gas, 0.0, 1.0)
    brake = np.clip(brake, 0.0, 1.0)
    steer = np.clip(steer, -1.0, 1.0)

    obs, reward, terminated, truncated, info = manual_env.step([gas, brake, steer])

    manual_episode_return += reward
    done = terminated or truncated or (obs[0] > 50)

    response = {
        "observation": obs.tolist(),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "done": bool(done),
        "cumulative_reward": float(manual_episode_return),
        "penalty": float(manual_episode_return)
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
