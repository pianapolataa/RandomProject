from flask import Flask, request, jsonify, send_from_directory
from stable_baselines3 import SAC
from path_env import CarPathEnv, generate_sine_path, generate_straight_path, generate_alternating_path
import numpy as np
import os
from threading import Lock

app = Flask(__name__, static_folder="static")

# Global environment instance for manual control
manual_env = None
manual_episode_return = 0.0
max_x = 50.0

# Model management
model_lock = Lock()
model_path = "car_path_sac_model_BEST.zip"
temp_model_path = "model_temp.zip"
current_model = SAC.load(model_path)

# Attempt to load temp model if available
if os.path.exists(temp_model_path):
    try:
        current_model = SAC.load(temp_model_path)
        print("Loaded model_temp.zip instead of default.")
    except Exception as e:
        print(f"Failed to load model_temp.zip: {e}")


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
    global current_model

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
        with model_lock:
            action, _ = current_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_penalty += reward
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
    global manual_env, manual_episode_return, max_x
    data = request.json
    friction = data.get("friction", 0.7)
    path_type = data.get("path_type", "sine")
    gas_sensitivity = data.get("gas_sensitivity", 0.9)
    brake_sensitivity = data.get("brake_sensitivity", 0.4)
    steer_sensitivity = data.get("steer_sensitivity", 0.1)

    if path_type == "sine":
        path = generate_sine_path()
        max_x = 50
    elif path_type == "alternating":
        path = generate_alternating_path()
        max_x = 200
    else:
        path = [(x, 0) for x in np.linspace(0, 50, 100)]
        max_x = 50

    manual_env = CarPathEnv(
        path=path,
        friction=friction,
        gas_sensitivity=gas_sensitivity,
        brake_sensitivity=brake_sensitivity,
        steer_sensitivity=steer_sensitivity,
        random_start=False
    )
    obs, _ = manual_env.reset()
    manual_episode_return = 0.0

    return jsonify({
        "observation": obs.tolist(),
        "path": [[float(x), float(y)] for x, y in path]
    })


@app.route("/manual_step", methods=["POST"])
def manual_step():
    global manual_env, manual_episode_return, max_x
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
    done = terminated or truncated or obs[0] > max_x

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

@app.route("/load_model", methods=["POST"])
def load_model():
    global current_model
    data = request.json
    model_type = data.get("model_type")

    if model_type == "pretrained":
        model_path = "car_path_sac_model_BEST.zip"
    elif model_type == "new":
        model_path = "model_temp.zip"
    else:
        return jsonify({"error": "Invalid model type"}), 400

    if not os.path.exists(model_path):
        return jsonify({"error": f"Model file {model_path} not found."}), 404

    try:
        with model_lock:
            current_model = SAC.load(model_path)
        return jsonify({"message": f"{model_type.capitalize()} model loaded successfully."})
    except Exception as e:
        return jsonify({"error": f"Failed to load model: {str(e)}"}), 500


@app.route("/retrain_model", methods=["POST"])
def retrain_model():
    global current_model
    data = request.json
    friction = data.get("friction", 0.7)
    path_type = data.get("path_type", "sine")
    gas_sensitivity = data.get("gas_sensitivity", 0.9)
    brake_sensitivity = data.get("brake_sensitivity", 0.4)
    steer_sensitivity = data.get("steer_sensitivity", 0.1)
    env2 = CarPathEnv(path=path_type, friction=friction, gas_sensitivity=gas_sensitivity, brake_sensitivity=brake_sensitivity, steer_sensitivity=
        steer_sensitivity, random_start=True)

    model = SAC("MlpPolicy", env2, learning_rate=1e-3, verbose=1, tensorboard_log="./car_rl_logs/")
    model.learn(total_timesteps=100000)

    model.save("model_temp")
    if not os.path.exists(temp_model_path):
        return jsonify({"error": f"{temp_model_path} not found"}), 400

    try:
        with model_lock:
            current_model = SAC.load(temp_model_path)
        return jsonify({"message": "New model loaded successfully."})
    except Exception as e:
        return jsonify({"error": f"Failed to load model: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
