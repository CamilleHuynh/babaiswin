# python -m pip install numpy
import torch
from parameters import rewards
from torch.autograd.grad_mode import F


# Fonctions pouvant être utilisées pour l'apprentissage : step

strings = ["bt", "ft", "wt", "yt", "is", "ro", "wo", "bo", "fo"]


def step(state, action):
    babaiswin, babaisyou, flagiswin, flagisyou = getRules(state)
    upstate = torch.rot90(state, action, [1, 2])
    stepUp(upstate, babaisyou, flagisyou)
    newState = torch.rot90(upstate, -action, [1, 2])
    #reward to be calculated according new state
    babaiswin,babaisyou,flagiswin,flagisyou = getRules(newState)
    
    if isWinState(newState, babaiswin, babaisyou, flagiswin, flagisyou):
        reward = rewards.win
        isFinal = True
    elif isDeathState(newState, babaisyou, flagisyou):
        reward = rewards.death
        isFinal = True
    elif babaiswin or flagiswin:
        reward = rewards.canWin
        isFinal = False
    else:
        reward = rewards.default
        isFinal = False
    return newState, reward, isFinal


# This function is not meant to be used for the learning
# it is only here for tests and to run the game
def stepbis(state, action):
    newState, reward, terminal = step(stringsToBits(state), action)
    state[:] = bitsToStrings(newState)[:]
    return state, reward, terminal

def best_possible_action(state, Q):
    """For state and a tensor of expected q-values, return the best possible action"""
    x = Q[0].detach().numpy().argsort()
    i=len(x)-1
    action = x[i]
    next_state, _, _ = step(state, action)
    while torch.equal(state, next_state):
        i-=1
        action = x[i]
        if i==0: return action
        next_state, _, _ = step(state,action)
    return action


def stringsToBits(state):
    nrow, ncol = len(state), len(state[0])
    newState = torch.zeros((9, nrow, ncol))
    for i in range(nrow):
        for j in range(ncol):
            for k in range(len(strings)):
                if strings[k] in state[i][j]:
                    newState[k][i][j] = 1
    return newState


def bitsToStrings(state):
    _, nrow, ncol = state.shape
    newState = []
    for i in range(nrow):
        newState.append([])
        for j in range(ncol):
            newState[i].append([])
            for k in range(len(strings)):
                if state[k][i][j]:
                    newState[i][j].append(strings[k])
    return newState


def getRules(state):
    babaisyou = False
    babaiswin = False
    flagisyou = False
    flagiswin = False
    _, nrow, ncol = state.shape
    for i in range(1, nrow-1):
        for j in range(1, ncol-1):
            if state[4][i][j]:  # state[4] is the "if" matrix
                if not babaiswin:
                    if ((state[0][i-1][j] and state[2][i+1][j])
                        or (state[0][i][j-1] and state[2][i][j+1])):
                        babaiswin = True
                if not babaisyou:
                    if ((state[0][i-1][j] and state[3][i+1][j])
                        or (state[0][i][j-1] and state[3][i][j+1])):
                        babaisyou = True
                if not flagiswin:
                    if ((state[1][i-1][j] and state[2][i+1][j])
                        or (state[1][i][j-1] and state[2][i][j+1])):
                        flagiswin = True
                if not flagisyou:
                    if ((state[1][i-1][j] and state[3][i+1][j])
                        or (state[1][i][j-1] and state[3][i][j+1])):
                        flagisyou = True
    for i in range(1, nrow-1):
        for j in [0, ncol-1]:
            if state[4][i][j]:  # state[4] is the "if" matrix
                if not babaiswin:
                    if (state[0][i-1][j] and state[2][i+1][j]):
                        babaiswin = True
                if not babaisyou:
                    if (state[0][i-1][j] and state[3][i+1][j]):
                        babaisyou = True
                if not flagiswin:
                    if (state[1][i-1][j] and state[2][i+1][j]):
                        flagiswin = True
                if not flagisyou:
                    if (state[1][i-1][j] and state[3][i+1][j]):
                        flagisyou = True
    for i in [0, ncol-1]:
        for j in range(1, nrow-1):
            if state[4][i][j]:  # state[4] is the "if" matrix
                if not babaiswin:
                    if (state[0][i][j-1] and state[2][i][j+1]):
                        babaiswin = True
                if not babaisyou:
                    if (state[0][i][j-1] and state[3][i][j+1]):
                        babaisyou = True
                if not flagiswin:
                    if (state[1][i][j-1] and state[2][i][j+1]):
                        flagiswin = True
                if not flagisyou:
                    if (state[1][i][j-1] and state[3][i][j+1]):
                        flagisyou = True
    return babaiswin, babaisyou, flagiswin, flagisyou


def printRulesFromString(state):
    bitstate = stringsToBits(state)
    babaiswin, babaisyou, flagiswin, flagisyou = getRules(bitstate)
    print("Rules :")
    if (babaiswin):
        print("Baba is Win")
    if (babaisyou):
        print("Baba is You")
    if (flagiswin):
        print("Flag is Win")
    if (flagisyou):
        print("Flag is You")


def isStop(state, i, j):
    return state[6][i][j]


def isPush(state, i, j):
    for k in range(6):
        if state[k][i][j]:
            return True
    return False


def isYou(state, i, j, babaisyou, flagisyou):
    return (babaisyou and state[7][i][j]) or (flagisyou and state[8][i][j])


def isWin(state, i, j, babaiswin, flagiswin):
    return (babaiswin and state[7][i][j]) or (flagiswin and state[8][i][j])


def stepUp(state, babaisyou, flagisyou):
    _, nrow, ncol = state.shape
    for col in range(ncol):
        for row in range(1, nrow):
            if isYou(state, row, col, babaisyou, flagisyou):
                emptyFound = False
                dist = 1
                while True:
                    if row-dist < 0 or isStop(state, row-dist, col):
                        break
                    elif isPush(state, row-dist, col):
                        dist += 1
                        continue
                    else:
                        emptyFound = True
                        break
                if emptyFound:
                    for i in range(1, dist):
                        j = dist-i
                        for k in range(6):
                            if state[k][row-j][col]:
                                state[k][row-j][col] = 0
                                state[k][row-j-1][col] = 1
                    if babaisyou and state[7][row][col]:
                        state[7][row][col] = 0
                        state[7][row-1][col] = 1
                    if flagisyou and state[8][row][col]:
                        state[8][row][col] = 0
                        state[8][row-1][col] = 1


def isWinState(state, babaiswin, babaisyou, flagiswin, flagisyou):
    if not((flagisyou or babaisyou) and (flagiswin or babaiswin)):
        return False
    _, nrow, ncol = state.shape
    for i in range(nrow):
        for j in range(ncol):
            if (isYou(state, i, j, babaisyou, flagisyou)
                and isWin(state, i, j, babaiswin, flagiswin)):
                return True
    return False


def isDeathState(state, babaisyou, flagisyou):
    if not(babaisyou or flagisyou):
        return True
    _, nrow, ncol = state.shape
    for i in range(nrow):
        for j in range(ncol):
            if isYou(state, i, j, babaisyou, flagisyou):
                return False
    return True


def isWinStringState(state):
    bitstate = stringsToBits(state)
    babaiswin, babaisyou, flagiswin, flagisyou = getRules(bitstate)
    return isWinState(bitstate, babaiswin, babaisyou, flagiswin, flagisyou)


def isDeathStringState(state):
    bitstate = stringsToBits(state)
    _, babaisyou, _, flagisyou = getRules(bitstate)
    return isDeathState(bitstate, babaisyou, flagisyou)
