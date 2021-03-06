# python -m pip install -U pygame --user
# To check it works :
# python -m pygame.examples.aliens
# To execute
# python babaiswin.py


import pygame as pg
from spritesheet import Spritesheet
from copy import deepcopy
from state import stepbis, printRulesFromString, isWinStringState, stringsToBits, best_possible_action
from neural_net import DQN
import torch
from parameters import env


# Initialize pygame
# Solve play sounds latency
pg.mixer.pre_init(44100, -16, 2, 1024)
pg.init()

# objects : no = vide; wo = wall; bo = baba; fo = flag;
# text    : bt = baba; ft = flag; is = is; wt = win; yt = you
# Ce qui est est affiché à l'écran est
# la matrice grille

# load the model
model = DQN(env.width, env.height, env.n_actions)
model.load_state_dict(torch.load('model.pth'))
model.eval()

state = env.grille

states = [deepcopy(state)]

nrow = len(state)
ncol = len(state[0])

# Create the window
screen = pg.display.set_mode((ncol*90, nrow*90))
pg.display.set_caption('')

# Import images

images = Spritesheet("spritesheet.png")

test = pg.transform.scale(images.get_sprite(0*24, 57*24, 24, 24), (90, 90))
baba = pg.transform.scale(images.get_sprite(1*24, 0*24, 24, 24), (90, 90))
baba_text = pg.transform.scale(images.get_sprite(6*24, 27*24, 24, 24), (90, 90))
flag = pg.transform.scale(images.get_sprite(6*24, 21*24, 24, 24), (90, 90))
flag_text = pg.transform.scale(images.get_sprite(1*24, 30*24, 24, 24), (90, 90))
is_text = pg.transform.scale(images.get_sprite(18*24, 30*24, 24, 24), (90, 90))
you_text = pg.transform.scale(images.get_sprite(20*24, 42*24, 24, 24), (90, 90))
win_text = pg.transform.scale(images.get_sprite(17*24, 42*24, 24, 24), (90, 90))
wall = pg.transform.scale(images.get_sprite(0*24, 57*24, 24, 24), (90, 90))
rock = pg.transform.scale(images.get_sprite(15*24, 21*24, 24, 24), (90, 90))


def init():
    running = True
    quitting = False
    while running:
        screen.fill(env.background_color)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
                quitting = True
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    printRulesFromString(state)
                elif event.key == pg.K_p:
                    init_state = deepcopy(state)
                    Q = model(stringsToBits(init_state).unsqueeze(0))
                    action = best_possible_action(stringsToBits(init_state), Q)
                    stepbis(state,action)
                    states.append(deepcopy(state))
                elif event.key == pg.K_r:
                    if(len(states) > 1):
                        del states[-1]
                        state[::] = deepcopy(states[-1])
        drawState(state)
        if isWinStringState(state):
            print("Win !")
            running = False
            stateWin = [[["yt"], ["wt"]]]
            screen.fill(env.background_color)
            drawState(stateWin)
        pg.display.update()
    while not quitting:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                quitting = True
            elif event.type == pg.KEYDOWN:
                quitting = True
    pg.quit()


def drawState(state):
    # Draws each element
    nrow, ncol = len(state), len(state[0])
    for row in range(nrow):
        for col in range(ncol):
            pos = (col * 90, row * 90)
            if "wo" in state[row][col]:
                screen.blit(wall, pos)
            elif "bo" in state[row][col]:
                screen.blit(baba, pos)
            elif "fo" in state[row][col]:
                screen.blit(flag, pos)
            elif "bt" in state[row][col]:
                screen.blit(baba_text, pos)
            elif "ft" in state[row][col]:
                screen.blit(flag_text, pos)
            elif "yt" in state[row][col]:
                screen.blit(you_text, pos)
            elif "wt" in state[row][col]:
                screen.blit(win_text, pos)
            elif "is" in state[row][col]:
                screen.blit(is_text, pos)
            elif "ro" in state[row][col]:
                screen.blit(rock, pos)


init()
