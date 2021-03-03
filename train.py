import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from PIL import Image
from torch.random import initial_seed

from state import bitsToStrings, stringsToBits, step
from neural_net import DQN
from replay_buffer import ReplayMemory, Transition

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import random
from itertools import count

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
GAMMA = 0.999
EPSILON = 0.5
TARGET_UPDATE = 10

height = 6
width = 5
n_actions = 4
map =   [[["no"], ["no"], ["no"], ["no"], ["no"]],
            [["ft"], ["no"], ["wt"], ["no"], ["no"]],
            [["bt"], ["is"], ["yt"], ["no"], ["no"]],
            [["no"], ["bo"], ["ro"], ["ro"], ["no"]],
            [["no"], ["is"], ["wo"], ["no"], ["no"]],
            [["no"], ["no"], ["no"], ["no"], ["fo"]]]

target_net = DQN(height, width, n_actions).to(device)
target_net.eval()

optimizer = optim.RMSprop(target_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0

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

    plt.show()

def select_action(state): #Select random a_t with probability epsilon, else a_t*
    global steps_done
    sample = random.random()
    steps_done += 1
    if sample > EPSILON:
        Q = target_net(state)
        return Q.max(1)[1]
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = target_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = nn.MSELoss(expected_state_action_values.unsqueeze(1), state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in target_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 50
for i_episode in range(num_episodes):
    print('episode', i_episode)
    # Initialize the environment and state
    state = stringsToBits(map)
    for t in count():
        # Select and perform an action
        action = select_action(torch.unsqueeze(state, 0))
        next_state, reward, done = step(state, action.item())
        reward = torch.tensor([reward], device=device)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            #plot_durations()
            break

print('Complete')
plt.show()