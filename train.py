import matplotlib.pyplot as plt
import math

from state import stringsToBits, step, isWinStringState, isDeathStringState, bitsToStrings, best_possible_action
from neural_net import DQN
from replay_buffer import ReplayMemory, Transition
from parameters import rewards, learning_param, env

import torch
import torch.optim as optim
import torch.nn.functional as F

import random
from itertools import count

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(env.width, env.height, env.n_actions).to(device)
target_net = DQN(env.width, env.height, env.n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=learning_param.learning_rate)
memory = ReplayMemory(10000)

episode_durations = []
print_freq = 1
steps_done = 0
n_step = 0


# Select random a_t with probability epsilon, else a_t*
def select_action(state):
    global steps_done
    global n_step
    sample = random.random()
    eps = learning_param.EPS_END + (learning_param.EPS_START - learning_param.EPS_END)* math.exp(-steps_done / learning_param.EPS_DECAY)
    steps_done += 1
    if sample > eps:
        Q = policy_net(state)
        action = best_possible_action(state.squeeze(0), Q)
        if n_step%print_freq==0:
            print("Select action ", action)
        return torch.tensor([[action]], device=device, dtype=torch.long) 

    else:
        Q = torch.rand(1, 4)
        #a random possible action
        action = best_possible_action(state.squeeze(0), Q) 
        if n_step%print_freq==0:
            print("Select random action ", action)
        return torch.tensor([[action]], device=device, dtype=torch.long)



def optimize_model():
    global n_step
    if len(memory) < learning_param.BATCH_SIZE:
        return
    # sample a batch of transitions
    transitions = memory.sample(learning_param.BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Concatenate the batch elements
    state_batch = torch.cat([s.unsqueeze(0) for s in batch.state])
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    print(state_batch.shape)

    # predicted value for state and chosen action
    predicted_values = torch.cat([policy_net(s.unsqueeze(0)) for s in state_batch])
    state_action_values = torch.tensor([predicted_values[i][action_batch[i]]
                                        for i in range(predicted_values.shape[0])])

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # if the state was final, V(s_{t+1}) is set to zero
    next_state_values = torch.cat([target_net(s.unsqueeze(0)).max(1).values for s in batch.next_state])
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * learning_param.GAMMA) + reward_batch
    
    #print("expected :", next_state_values[0].item(), "new :", expected_state_action_values[0].item(), "différence :", next_state_values[0].item()-expected_state_action_values[0].item())


    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values.unsqueeze(1), expected_state_action_values.unsqueeze(1))
    if n_step % print_freq == 0:
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
    #print(state[7])
    n_step = 0
    for t in count():
        # Select and perform an action
        action = select_action(torch.unsqueeze(state, 0))
        next_state, reward, done = step(state, action.item())
        reward = torch.tensor([reward], device=device)
        #print(next_state[7])

        if n_step > learning_param.MAX_ITERATIONS:
            reward = rewards.death
            reward = torch.tensor([reward], device=device)
            done = True

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state
        n_step += 1

        if n_step % print_freq == 0:
            print(('Episode [{}/{}] - Etape {}').format(i_episode, learning_param.num_episodes, n_step))

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        if done:
            if (isWinStringState(bitsToStrings(next_state))):
                print("WIN !")
            if (isDeathStringState(bitsToStrings(next_state))):
                print("DEATH !")
            episode_durations.append(t + 1)
            break

    # Update the target network, copying all weights and biases in DQN
    if i_episode % learning_param.TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')

# Save the model after train
torch.save(target_net.state_dict(), 'model.pth')
print("Saved model to disk")
