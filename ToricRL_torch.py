import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from ToricEnv import ToricEnv
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

dim = 5
p = .1
BATCH_SIZE = 32
learning_rate = .001
TARGET_UPDATE = 100

center = math.floor((dim*dim)/2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def action_lattice(dim):
    action_lattice = np.zeros((4, dim, dim))
    f=0
    for i in range(action_lattice.shape[0]):
        for j in range(action_lattice.shape[1]):
            for k in range(action_lattice.shape[2]):
                action_lattice[i,j,k] = f
                f+=1
    return action_lattice

def position_lattice(dim):
    position_lattice = np.zeros((4, dim, dim), dtype=object)
    for j in range(position_lattice.shape[1]):
        for k in range(position_lattice.shape[2]):
            up= (j-1)
            down= j
            left= (k-1)
            right= k
            position_lattice[0][j][k]= np.array([0,up,k])
            position_lattice[1][j][k]= np.array([0,down,k])
            position_lattice[2][j][k]= np.array([1,j,left])
            position_lattice[3][j][k]= np.array([1,j,right])
    return position_lattice

position_lattice = position_lattice(dim)

action_lattice = action_lattice(dim)

act = np.array((1,1), dtype=object)

env = ToricEnv(dim, p, position_lattice, action_lattice)


def roll_to_center(state):
    center = math.floor((state.shape[0]**2) / 2 ) % state.shape[0]
    states = []
    for vertex in np.transpose(np.where(state==1)):
        temp = state
        if vertex[0] < center and vertex[1] < center:
            temp = np.roll(state, center - vertex[0], axis = 0)
            temp = np.roll(temp, center - vertex[1], axis = 1)
        if vertex[0] < center and vertex[1] > center:
            temp = np.roll(state, center - vertex[0], axis = 0)
            temp = np.roll(temp, center - vertex[1], axis = 1)
        if vertex[0] > center and vertex[1] < center:
            temp = np.roll(state, center - vertex[0], axis = 0)
            temp = np.roll(temp, center - vertex[1], axis = 1)
        if vertex[0] > center and vertex[1] > center:
            temp = np.roll(state, center - vertex[0], axis = 0)
            temp = np.roll(temp, center - vertex[1], axis = 1)
        if vertex[0] < center and vertex[1] == center:
            temp = np.roll(state, center - vertex[0], axis = 0)
        if vertex[0] == center and vertex[1] < center:
            temp = np.roll(state, center -  vertex[1], axis = 1)
        if vertex[0] > center and vertex[1] == center:
            temp = np.roll(state, center - vertex[0], axis = 0)
        if vertex[0] == center and vertex[1] > center:
            temp = np.roll(state, center - vertex[1], axis = 1)
        if vertex[0] == center and vertex[1] == center:
            temp =  state
        states.append(temp)
    return states


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    
    def __init__(self, num_memories):
        self.memory = deque([], maxlen=num_memories)
        
    def save_memory(self, *args):
        self.memory.append(Transition(*args))
        
    def sample_memory(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)    
        


class ToricCodeNet(nn.Module):
    
    def __init__(self, env):
        super(ToricCodeNet, self).__init__()
        self.input_dim = int(np.prod(env.observation_space.shape))
        self.action_dim = 4
        self.net = self.generate_net()
        self.epsilon = 1
        self.gamma = .95
        self.epsilon_decay = .995
        self.epsilon_min = .01
        self.memory = ReplayMemory(1000)
        
        
    def generate_net(self):
        #Want conv 2 stride -> 4 FC layers
        #return nn.Sequential(nn.Linear(self.input_dim, 128), nn.ReLU(), nn.Linear(128, self.action_dim))
        return nn.Sequential(nn.Conv2d(self.input_dim, 512, 3, 2), 
                             nn.ReLU(), 
                             nn.Linear(256, 128), 
                             nn.ReLU(), 
                             nn.Linear(128, 64), 
                             nn.ReLU(), 
                             nn.Linear(64, 32),  
                             nn.ReLU(), 
                             nn.Linear(32, self.action_dim) )

    def forward(self, x):
        return self.net(x)

        
    def remember(self, state, action, next_state, reward, done):
       self.memory.save_memory(state, action, next_state, reward, done) 

    def act(self, state, step):
        
        eps_threshold = self.epsilon_min + (self.epsilon - self.epsilon_min) * \
            math.exp(-1. * step / self.epsilon_decay)
        vertices = np.transpose(np.where(state==1))
        if random.random() <= eps_threshold:
            r = random.randint(0, len(vertices))
            vertex = vertices[r]
            return random.randrange(0, self.action_dim), vertex
        else:
            states = roll_to_center(state)
            q_action = np.array([])
            for i in range(len(states)):
                state_tensor = torch.as_tensor(states[i], dtype=torch.int32)
                q_values = self(state_tensor.unsqueeze(0))
                max_q_index = torch.argmax(q_values, dim=1)[0]
                q_action.append(max_q_index.detach().item())
            return max(q_action), vertices[q_action.index(max(q_action))]
    
    
    def replay(self, BATCH_SIZE, net, target_net, optimizer):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        state_action_values = net(state_batch).gather(1, action_batch)
    
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
    
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
        # Optimize the model
        
        optimizer.zero_grad()
        loss.backward()
        for param in net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        

    def load(self, name):
        pass

    def save(self, name):
        pass

episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    #     display.clear_output(wait=True)
    #     display.display(plt.gcf())


def main():
    num_episodes = 100
    step = 0
    net = ToricCodeNet(env)
    target_net = ToricCodeNet(env)
    target_net.load_state_dict(net.state_dict())
    target_net.eval()
    
    memory = net.memory
    
    optimizer= torch.optim.Adam(net.parameters(), lr = learning_rate) 
    state = env.reset()
    done = False
    for ep in range(num_episodes):
        state = env.reset()
        for t in count():
            if not done:
                for vertex in np.transpose(np.where(state==1)):
                    action = net.act(state, step)
                    act[0] = action
                    act[1] = np.array(vertex)
                    next_state, reward, done, _ = env.step(act)
                    reward = torch.tensor([reward], device=device)
                step += 1
            else:
                next_state = None
            
            memory.push(state, action, next_state, reward)
            
            state =  next_state
            net.replay(BATCH_SIZE, net, target_net, optimizer)
            
            if done:
                step = 0
                episode_durations.append(t + 1)
                plot_durations()
                break
            
            if ep % TARGET_UPDATE == 0:
                target_net.load_state_dict(net.state_dict())
                
    print('Done training')
    plt.show()
if __name__ == '__main__':
	main()





