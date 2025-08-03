from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from path_env import CarPathEnv

env = CarPathEnv(path='sine')
check_env(env) 

model = SAC("MlpPolicy", env, learning_rate=1e-3, verbose=1, tensorboard_log="./car_rl_logs/")
model.learn(total_timesteps=100000)

model.save("car_path_sac_model_BEST")
