# Loading required libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu") # uncomment if you want to use CPU
print('Device used for training: ' + "GPU" if torch.cuda.is_available() else "CPU")

import gym
from gym_minigrid.envs import EmptyEnv
import vizdoomgym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from PIL import Image
from skimage.transform import resize
from arguments import get_args

# Importing local packages
from utils import *
from replay_memory import *
from networks import *

#### Initializing required variables

args = get_args()
# Environment params
IMG_HEIGHT = 80
IMG_WIDTH = 80
ENV_NAME = args.env
env = gym.make(ENV_NAME)

# Memory buffer params
MEMORY_SIZE = 5000
PRIORITY_MEMORY_SIZE = 1000

# Action selection params
steps_done = 0
BATCH_SIZE = 4000
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 2000
TARGET_UPDATE = 20
EPS = 1

# Creating all the networks
tnet = thetaNet().to(device)
tnet2 = theta2Net().to(device)
anet = alphaNet().to(device)
wnet = wNet().to(device)
anet_target = alphaNet().to(device)
anet_target.load_state_dict(anet.state_dict())
anet_target.eval() # CHECK: what this actually does

# Creating the buffer objects (normal and reward scenarios)
memory = ReplayMemory(MEMORY_SIZE)
memory_win = ReplayMemory(PRIORITY_MEMORY_SIZE)

# Defining loss functions and setting up variables to track loss
loss = nn.MSELoss()
loss_a = nn.MSELoss()
loss_b = nn.MSELoss()
L_r_vec = []
L_m_vec = []
L_a_vec = []

# Optimization settings
tw_params = list(tnet.parameters()) + list(tnet2.parameters()) + list(wnet.parameters())
optimizer_tw = optim.Adam(tw_params, lr=50e-5)
optimizer_a = optim.Adam(anet.parameters(), lr=25e-5)

#### Defining required functions 

def select_action(phi, w, greedy=False):
    """Function to select action
    
    Inputs:
        phi -> abstracted states batch
                (tensor of size b x 512)
        w -> weights of the reward network
                (tensor of size 512: CHECK)
        greedy -> returns greedy action if true
                    (can be used during testing phase)
                    returns eps-greedy if False
    """
    
    # Calculating greedy action
    with torch.no_grad():
        greedy_action = anet(phi).matmul(w).max(1)[1]
    
    if(greedy):
        return greedy_action
    
    global steps_done
    global EPS
    
    # recalculating epsilon value
    sample = random.random()
    EPS = EPS_END + (EPS_START - EPS_END) *         math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    # doing epsilon greedy
    if sample > EPS:
        with torch.no_grad():
            aout = anet(phi)
            return aout.matmul(w).max(1)[1]
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model():
    """Function that samples from buffer, estimates loss and backprops
    to update network parameters.
    
    """
    
    # When memory is not ready, do nothing
    if (len(memory) < BATCH_SIZE) or (not memory_win.is_ready()):
        return
    
    # Training reward and reconstruction branches
    if(np.random.rand()>0.8): # winning samples 20% times this runs
        transitions, bs = memory_win.sample(BATCH_SIZE)
        
    else: # intermediate samples
        transitions, bs = memory.sample(BATCH_SIZE)
        
    # Putting data is neat format
    batch = Transition(*zip(*transitions))
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    nstate_batch = torch.cat(batch.next_state)
    
    # Optimizing the reward and reconstruction branches
    L_r = F.smooth_l1_loss(reward_batch, wnet(tnet(nstate_batch)).squeeze(1))
    L_a = loss_a(state_batch, tnet2(tnet(state_batch))) + loss_b(nstate_batch, tnet2(tnet(nstate_batch)))
    L_r_vec.append(L_r.item())
    L_a_vec.append(L_a.item())
    L_ra = L_a + L_r
    optimizer_tw.zero_grad()
    L_ra.backward()
    optimizer_tw.step()
    
    
    # Sampling for buffer: for training SR branch
    transitions, bs = memory.sample(32)
    batch = Transition(*zip(*transitions))
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    nstate_batch = torch.cat(batch.next_state)
    done_batch = torch.cat(batch.done)
    
    
    # Create a non-final state mask
    non_final_mask = torch.tensor(tuple(map(lambda s: s==0,
                                          batch.done)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for en, s in enumerate(batch.next_state)
                                                if batch.done[en]==0])
    # Finding max actions for each batch
    action_max = anet(tnet(non_final_next_states)).matmul(wnet.head.weight.data.view(-1,1)).max(1)[1]
    # initialize them to values we need for terminal states
    next_state_ests = tnet(nstate_batch) 
    # replace the values of non-terminal states based on update equation
    next_state_ests[non_final_mask] = anet_target(tnet(non_final_next_states))[torch.arange(0, non_final_mask.sum()),action_max.squeeze(),:]
    
    # Optimizing the SR branch
    U_observed = anet(tnet(state_batch))[torch.arange(0, bs),action_batch.squeeze(),:]
    U_estimated = tnet(state_batch) + GAMMA * next_state_ests
    L_m = loss(U_observed, U_estimated)
    L_m_vec.append(L_m.item())
    optimizer_a.zero_grad()
    L_m.backward()
    optimizer_a.step()

#### Training starts here
# (line number below are the same as line numbers in the pseudo code)
print('Training is starting...')

# Initializations: Line 1
num_episodes = 300 # CHANGE
n_actions = 3
R_eps = []
ed = []; eps_vec = [];
actions = []
eval_r_mean = []; eval_r_std = []

# Setting seeds
torch.manual_seed(0); np.random.seed(0)
env.seed(0)
torch.backends.cudnn.deterministic = True

global i_episode
for i_episode in tqdm(range(num_episodes)): # Line 2
    R = 0
    # Trick from paper: dividing batch size by 2
    if(BATCH_SIZE>2):
        BATCH_SIZE = BATCH_SIZE // 2
    
    # Initialize the environment and state: Line 3
    env.reset()
    state = get_screen(env, h=IMG_HEIGHT, w=IMG_WIDTH, device=device)
    
    for t in count(): # Line 4
        
        # Find abstracted states: Line 5
        phi = tnet(state)
        
        # Select an action: Line 6
        action = select_action(phi, wnet.head.weight.data.view(-1,1))
        actions.append(action.item())
        
        # Perform an action: Line 7
        _, reward, done, _ = env.step(action.item())
        done = torch.tensor([done], device=device)
        if(reward > 0):
            reward = 1
        R = R + reward
        reward = torch.tensor([reward], device=device).float()
        next_state = get_screen(env, h=IMG_HEIGHT, w=IMG_WIDTH, device=device)
        
        # Store the transition in memory: Line 8
        if(reward<=0):
            memory.push(state, action, next_state, reward, done)
        else:
            memory_win.push(state, action, next_state, reward, done)
            memory.push(state, action, next_state, reward, done)            
            
        # Move to the next state
        state = next_state

        # Lines 9 - 11
        optimize_model() # TODO
        
        # Additional tracking
        if done:
            ed.append(t+1)
            R_eps.append(R)
            eps_vec.append(EPS)
            break
    
    # Updating target network  
    if i_episode % TARGET_UPDATE == 0:
        anet_target.load_state_dict(anet.state_dict())
        
    # End training if you're almost exploiting, don't waste iterations
    if EPS < 1.05*EPS_END:
        break

# Visaulizing/Evaluating the trained model
def visualize_results():
    plt.figure(figsize=(18,9))
    plt.subplot(3,3,1); plt.plot(ed); plt.title('ep_length');
    plt.subplot(3,3,2); plt.plot(R_eps); plt.title('R'); 
    plt.subplot(3,3,3); plt.plot(eps_vec); plt.title('eps values'); 
    plt.subplot(3,3,4); plt.plot(L_r_vec[:]); plt.title('reward loss'); 
    plt.subplot(3,3,5); plt.plot(L_m_vec[:]); plt.title('SR loss'); 
    plt.subplot(3,3,6); plt.plot(L_a_vec[:]); plt.title('reconstruction loss'); 
    plt.subplot(3,3,7); plt.plot(np.cumsum(ed)); plt.title('cumm episode steps')
    plt.show()

visualize_results()

a, b = evaluate(10)
print('Completely solved (out of 10): ', a)