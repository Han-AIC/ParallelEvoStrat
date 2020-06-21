import gym
import pybullet_envs
env = gym.make('MinitaurBulletEnv-v0')
current_state = env.reset()
state_shape = env.observation_space.shape
action_shape = env.action_space
print(action_shape.shape )
print(action_shape.sample())
print(state_shape)

env.step([ 0.7914073,   0.051298,    0.3879828,   0.11389165, -0.3248997,  -0.17246456,
  0.6853794,  -0.5336161 ])
# from gym import envs
# print(envs.registry.all())
