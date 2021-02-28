#python -m pip install numpy
import numpy as np
import stateHandler as sh

strings = ["bt","ft","wt","yt","is","ro","wo","bo","fo"]

def step(state,action):
    newState,reward,terminal = sh.step(bitsToStrings(state),action)
    return stringsToBits(newState),reward,terminal

def stepbis(state,action):
    newState,reward,terminal = step(stringsToBits(state),action)
    return bitsToStrings(newState),reward,terminal

def stringsToBits(state):
    nrow,ncol = len(state),len(state[0])
    newState = np.zeros((nrow,ncol,9))
    for i in range(nrow):
        for j in range(ncol):
            for k in range(len(strings)):
                if strings[k] in state[i][j]:
                    newState[i][j][k] = 1
    return newState

def bitsToStrings(state):
    nrow,ncol,_ = state.shape
    newState = []
    for i in range(nrow):
        newState.append([])
        for j in range(ncol):
            newState[i].append([])
            for k in range(len(strings)):
                if state[i][j][k]:
                    newState[i][j].append(strings[k])
    return newState
                