#python -m pip install numpy
import numpy as np
import stateHandler as sh

strings = ["bt","ft","wt","yt","is","ro","wo","bo","fo"]

def step(state,action):
    newState,reward,terminal = sh.step(bitsToStrings(state),action)
    return stringsToBits(newState),reward,terminal

def stringsToBits(state):
    nrow,ncol = len(state),len(state[0])
    newState = np.full((9,ncol,nrow),0)
    for i in range(nrow):
        for j in range(ncol):
            for k in range(len(strings)):
                if strings[k] in state[i][j]:
                    newState[k][j][i] = 1
    return newState

def bitsToStrings(state):
    _,ncol,nrow = state.shape
    newState = []
    for i in range(nrow):
        newState.append([])
        for j in range(ncol):
            newState[i].append([])
            for k in range(len(strings)):
                if state[k][j][i]:
                    newState[i][j].append(strings[k])
                else:
                    newState[i][j].append("no")
    return newState
                