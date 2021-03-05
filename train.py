import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.random import initial_seed
import math

from state import stringsToBits, step
from neural_net import DQN
from replay_buffer import ReplayMemory, Transition

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
from itertools import count

from parameters import rewards, learning_param, env

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(env.height, env.width, env.n_actions).to(device)
target_net = DQN(env.height, env.width, env.n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

episode_durations = []

steps_done = 0

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
    eps = learning_param.EPS_END + (learning_param.EPS_START - learning_param.EPS_END) * \
        math.exp(-1. * steps_done / learning_param.EPS_DECAY)
    steps_done += 1
    if sample > eps:
        Q = policy_net(state)
        action = torch.unsqueeze(Q.max(1)[1], 0)
        print("Select action ", action.item())
        return action #index of action with best reward for each row
    else:
        action = torch.tensor([[random.randrange(env.n_actions)]], device=device, dtype=torch.long)
        print("Select random action ", action.item())
        return action

def optimize_model():
    if len(memory) < learning_param.BATCH_SIZE:
        return
    transitions = memory.sample(learning_param.BATCH_SIZE) #sample a batch of transitions
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Concatenate the batch elements
    state_batch = torch.cat([s.unsqueeze(0) for s in batch.state])
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    #predicted value for state and chosen action
    predicted_values = torch.cat([policy_net(s.unsqueeze(0)) for s in state_batch]) 
    state_action_values = torch.tensor([ predicted_values[i][action_batch[i]] for i in range(predicted_values.shape[0]) ]) 

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    #if the state was final, V(s_{t+1}) is set to zero
    next_state_values = torch.cat([target_net(s.unsqueeze(0)).max(1).values for s in batch.next_state])
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * learning_param.GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values.unsqueeze(1), expected_state_action_values.unsqueeze(1))
    print("                                  Loss : ", loss.item())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.data.clamp_(-1, 1)
    optimizer.step()

for i_episode in range(learning_param.num_episodes):
    print('Episode', i_episode, '=========================================================================================================')
    # Initialize the env and state
    state = stringsToBits(env.grille)
    n_step = 0
    for t in count():
        # Select and perform an action 
        action = select_action(torch.unsqueeze(state, 0))
        next_state, reward, done = step(state, action.item())
        reward = torch.tensor([reward], device=device)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state
        n_step+=1

        print(('Episode [{}/{}] - Etape {}').format(i_episode, learning_param.num_episodes, n_step))

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            #plot_durations()
            break
        
    # Update the target network, copying all weights and biases in DQN
    if i_episode % learning_param.TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')

#Save the model after train
torch.save(target_net.state_dict(), 'model.pth')
print("Saved model to disk")

plt.show()