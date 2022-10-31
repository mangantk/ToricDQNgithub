import gym
from gym import spaces
import numpy as np
from Toric_Model import ToricModel




class ToricEnv(gym.Env):
    def __init__(self, dim, p, position_lattice, action_lattice):
        super(ToricEnv, self).__init__()
        self.dim = dim
        self.p = p
        self.position_lattice =  position_lattice
        self.action_lattice = action_lattice
        self.Toric = ToricModel(self.dim, self.p, self.position_lattice, self.action_lattice)
        #self.action_space = spaces.Tuple((spaces.Discrete(4), spaces.Box(-1, self.Toric.dim-1, shape=(2,),  dtype=np.int32)))
        self.action_space = spaces.Discrete(4)
        self.star_lattice = self.Toric.star_lattice
        self.observation_space = spaces.Box(self.star_lattice, np.ones((self.Toric.dim,self.Toric.dim)), dtype=np.int32)
        
    def step(self, action):
        observation = self.Toric.p_random_errors()
        #observation = self.Toric.n_random_errors(5)
        #num_errors1 = np.sum(np.sum(observation))
        observation = self.Toric.step(action)
        #num_errors2 = np.sum(np.sum(observation))
        # if num_errors2 < num_errors1:
        #     reward = -1
        #     reward = -abs(num_errors2 - num_errors1).astype(np.float32)
        #     print(reward)
        #     print(reward.dtype)
        # else:
        #     reward = -10
        #     reward = -(num_errors2 + num_errors1).astype(np.float32)
        #     print(reward)
        #     print(reward.dtype)
        if self.Toric.done(observation):
            done = True
            reward = 0
        else:
            done = False
            reward = -1
        #is_ground = {self.Toric.is_ground()}
        is_ground = {}
        return observation, reward, done, is_ground
    
    def reset(self):
        #del self.Toric
        self.Toric = ToricModel(self.dim, self.p, self.position_lattice, self.action_lattice)
        observation = np.array(self.Toric.p_random_errors())
        return observation
    
    def render(self):
        pass
    
    def close(self):
        pass
        