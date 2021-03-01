# python -m pip install -U pygame --user
# To check it works :
# python -m pygame.examples.aliens
# To execute
# python babaiswin.py


import numpy as np
import pygame as pg
from stateHandler import step, printRules, simplify, isWinState
from spritesheet import Spritesheet
from copy import deepcopy



# Initialize pygame
# Solve play sounds latency
pg.mixer.pre_init(44100, -16, 2, 1024)
pg.init()

background_color = (225, 225, 225)

#objects : no = vide; wo = wall; bo = baba; fo = flag;
#text    : bt = baba; ft = flag; is = is; wt = win; yt = you
# Ce qui est est affiché à l'écran est 
# la matrice map

def reset():
    #Si changement, modifier aussi liste de états dans stateHandler (getStateList())
    map =   [[["no"], ["no"], ["no"], ["no"], ["no"]],
             [["ft"], ["no"], ["wt"], ["no"], ["no"]],
             [["bt"], ["is"], ["yt"], ["no"], ["no"]],
             [["no"], ["bo"], ["ro"], ["ro"], ["no"]],
             [["no"], ["is"], ["wo"], ["no"], ["no"]],
             [["no"], ["no"], ["no"], ["no"], ["fo"]]]
    
    return map

map = reset()

state =[]
for i in range(len(map[0])):
    state.append([])
    for j in range(len(map)):
        state[i].append(map[j][i])

simplify(state)
states = [deepcopy(state)]

nrow = len(state)
ncol = len(state[0])

# Create the window
screen = pg.display.set_mode((nrow*90, ncol*90))
pg.display.set_caption('')

# Import images

images = Spritesheet("spritesheet.png")

test = pg.transform.scale(images.get_sprite(0*24,57*24,24,24),(90,90))
baba = pg.transform.scale(images.get_sprite(1*24,0*24,24,24),(90,90))
baba_text = pg.transform.scale(images.get_sprite(6*24,27*24,24,24),(90,90))
flag = pg.transform.scale(images.get_sprite(6*24,21*24,24,24),(90,90))
flag_text = pg.transform.scale(images.get_sprite(1*24,30*24,24,24),(90,90))
is_text = pg.transform.scale(images.get_sprite(18*24,30*24,24,24),(90,90))
you_text = pg.transform.scale(images.get_sprite(20*24,42*24,24,24),(90,90))
win_text = pg.transform.scale(images.get_sprite(17*24,42*24,24,24),(90,90))
wall = pg.transform.scale(images.get_sprite(0*24,57*24,24,24),(90,90))
keke = pg.transform.scale(images.get_sprite(2*24,3*24,24,24),(90,90))
keke_text = pg.transform.scale(images.get_sprite(20*24,30*24,24,24),(90,90))
rock = pg.transform.scale(images.get_sprite(15*24,21*24,24,24),(90,90))





def init():
    running = True
    quitting = False
    while running:
        screen.fill(background_color)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False      
                quitting = True
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    printRules(state)
                elif event.key == pg.K_UP:
                    step(state,0)
                    states.append(deepcopy(state))
                elif event.key == pg.K_RIGHT:
                    step(state,1)
                    states.append(deepcopy(state))
                elif event.key == pg.K_DOWN:
                    step(state,2)
                    states.append(deepcopy(state))
                elif event.key == pg.K_LEFT:
                    step(state,3)
                    states.append(deepcopy(state))
                elif event.key == pg.K_r:
                    if(len(states)>1):
                        del states[-1]
                        state[::] = deepcopy(states[-1])
        drawState(state)
        if isWinState(state):
            print("Win !")
            running = False
            stateWin=[[["yt"]],[["wt"]]]
            screen.fill(background_color)
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
    nrow,ncol = len(state),len(state[0])
    for row in range(nrow):
        for col in range(ncol):
            pos = (row * 90, col * 90)
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
            elif "ko" in state[row][col]:
                screen.blit(keke, pos)
            elif "kt" in state[row][col]:
                screen.blit(keke_text, pos)
            elif "ro" in state[row][col]:
                screen.blit(rock, pos)


init()
