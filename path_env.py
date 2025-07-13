import numpy as np
import gymnasium as gym
from gymnasium import spaces

import numpy as np

def generate_sine_path(x_range=(0, 50), step=0.5, amplitude=10, frequency=0.2):
    x_vals = np.arange(x_range[0], x_range[1], step)
    path = [(x, amplitude * np.sin(frequency * x)) for x in x_vals]
    return path

class CarPathEnv(gym.Env):
    def __init__(self, path=None, friction=0.7, gas_sensitivity=0.9, brake_sensitivity=0.4, steer_sensitivity=0.1, random_start=True):
        super().__init__()
        self.friction = friction
        self.gas_sensitivity = gas_sensitivity
        self.brake_sensitivity = brake_sensitivity
        self.steer_sensitivity = steer_sensitivity
        self.random_start = random_start

        # ==== Path ====
        self.path = path or generate_sine_path()

        # ==== Action Space ====
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # ==== Observation Space ====
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.random_start:
            radius = 20
            angle = np.random.uniform(0, 2 * np.pi)
            self.x = radius * np.cos(angle)
            self.y = radius * np.sin(angle)
            self.heading = np.random.uniform(-np.pi, np.pi)
        else:
            self.x = 0.0
            self.y = 0.0
            self.heading = 0.0

        self.speed = 0.0
        self.step_count = 0

        self._update_dist_and_angle()
        return self._get_obs(), {}


    def step(self, action):
        gas, brake, steer = action
        steer = np.clip(steer, -1.0, 1.0)

        # Update heading, speed, position
        self.heading += steer * self.steer_sensitivity
        self.speed += gas * self.gas_sensitivity
        self.speed -= brake * self.brake_sensitivity
        self.speed *= (1 - self.friction)
        self.speed = max(self.speed, 0)
        # print(self.speed)

        self.x += self.speed * np.cos(self.heading)
        self.y += self.speed * np.sin(self.heading)

        self.step_count += 1
        self._update_dist_and_angle()

        obs = self._get_obs()
        forward_velocity = self.speed * np.cos(self.angle_to_path)
        reward = (
            10 * forward_velocity                    # reward moving faster
            - 0.1 * (self.dist_to_path ** 2)                # penalty for distance squared
            - 0.01 * (self.angle_to_path ** 2)         # penalty for heading angle difference
            - 0.001 * abs(steer)                       # penalty for steering effort (smoothness)
        )
        reward += 0.1

        terminated = bool(self.dist_to_path > 20.0)  # e.g. crashed/far away
        truncated = bool(self.step_count >= 300)      # episode length limit

        info = {}

        return obs, reward, terminated, truncated, info


    def _update_dist_and_angle(self):
        car_pos = np.array([self.x, self.y])
        min_dist = float('inf')
        closest_point = None
        closest_segment_dir = None

        # Go through all segments: (p0, p1)
        for i in range(len(self.path) - 1):
            p0 = np.array(self.path[i])
            p1 = np.array(self.path[i + 1])
            segment = p1 - p0
            length = np.linalg.norm(segment)

            if length == 0:
                continue  # skip zero-length segments

            # Vector from p0 to car
            to_car = car_pos - p0

            # Project car onto the segment
            t = np.clip(np.dot(to_car, segment) / length**2, 0.0, 1.0)
            proj = p0 + t * segment

            dist = np.linalg.norm(car_pos - proj)

            if dist < min_dist:
                min_dist = dist
                closest_point = proj
                closest_segment_dir = segment / length  # normalized direction

            # Store results
            self.dist_to_path = min_dist

            # Compute angle to path: difference between heading and path direction
            path_angle = np.arctan2(closest_segment_dir[1], closest_segment_dir[0])
            angle_diff = path_angle - self.heading
            self.angle_to_path = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))  # wrap to [-π, π]


    def _get_obs(self):
        return np.array([
            self.x,
            self.y,
            self.heading,
            self.speed,
            self.dist_to_path,
            self.angle_to_path
        ], dtype=np.float32)

    def render(self, mode='human'):
        print(f"Pos=({self.x:.2f},{self.y:.2f}) Speed={self.speed:.2f} Heading={self.heading:.2f} Dist={self.dist_to_path:.2f}")

if __name__ == "__main__":
    env = CarPathEnv()
    obs = env.reset()
    print("Initial state:", obs)

    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        print("Step:", obs, "Reward:", reward)
