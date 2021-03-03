#python -m pip install numpy
import numpy as np
import stateHandler as sh
import torch

strings = ["bt","ft","wt","yt","is","ro","wo","bo","fo"]

def step(state,action):
    newState,reward,terminal = sh.step(bitsToStrings(state),action)
    return stringsToBits(newState),reward,terminal

#This function is not meant to be used for the learning, it is only here for tests and to run the game
def stepbis(state,action):
    newState,reward,terminal = step(stringsToBits(state),action)
    calcul=bitsToStrings(newState)
    for i in range(len(state)):
        state[i]=calcul[i]
    return state,reward,terminal

def stringsToBits(state):
    nrow,ncol = len(state),len(state[0])
    newState = torch.zeros((9,ncol,nrow))
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
                