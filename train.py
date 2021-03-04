import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.random import initial_seed

from state import stringsToBits, step
from neural_net import DQN
from replay_buffer import ReplayMemory, Transition

import torch
import torch.nn as nn
import torch.optim as optim

import random
from itertools import count

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
GAMMA = 0.999
EPSILON = 0.5
TARGET_UPDATE = 10

num_episodes = 200

height = 6
width = 5
n_actions = 4
grille =   [[["no"], ["no"], ["no"], ["no"], ["no"]],
            [["ft"], ["no"], ["wt"], ["no"], ["no"]],
            [["bt"], ["is"], ["yt"], ["no"], ["no"]],
            [["no"], ["bo"], ["ro"], ["ro"], ["no"]],
            [["no"], ["is"], ["wo"], ["no"], ["no"]],
            [["no"], ["no"], ["no"], ["no"], ["fo"]]]
#numéros des actions
#       0
#      3  1
#       2

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
        return torch.unsqueeze(Q.max(1)[1], 0) #index of action with best reward for each row
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE) #sample a batch of transitions
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Concatenate the batch elements
    state_batch = torch.cat([s.unsqueeze(0) for s in batch.state])
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    #predicted value for state and chosen action
    state_action_values = torch.cat([target_net(state_batch)[action_batch[i].item()]
                                        for i in range(state_batch.shape[0]) ]) 

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    #if the state was final, V(s_{t+1}) is set to zero
    next_state_values = torch.cat([target_net(s.unsqueeze(0)).max(1).values if s is not None
                                        else torch.tensor([0])
                                            for s in batch.next_state])
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = nn.MSELoss()
    output = loss(expected_state_action_values.unsqueeze(1), state_action_values)
    print("Optimize the model - Loss : ", output.item())

    # Optimize the model
    optimizer.zero_grad()
    output.backward()
    for param in target_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

for i_episode in range(num_episodes):
    print('episode', i_episode)
    # Initialize the environment and state
    state = stringsToBits(grille)
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

#Save the model after train
torch.save(target_net.state_dict(), 'model.pth')
print("Saved model to disk")

plt.show()