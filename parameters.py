# game parameters
class env:
    background_color = (225, 225, 225)
    height = 6
    width = 5
    n_actions = 4
    grille = [[["no"], ["no"], ["no"], ["no"], ["no"]],
              [["ft"], ["no"], ["wt"], ["no"], ["no"]],
              [["bt"], ["is"], ["yt"], ["no"], ["no"]],
              [["bo"], ["no"], ["ro"], ["ro"], ["no"]],
              [["no"], ["is"], ["wo"], ["no"], ["no"]],
              [["no"], ["no"], ["no"], ["no"], ["fo"]]]

    # numéros des actions
    #       0
    #      3  1
    #       2


# learning parameters
class learning_param:
    BATCH_SIZE = 50
    GAMMA = 0.9
    EPS_END = 0.2
    EPS_START = 0.9
    EPS_DECAY = 50
    TARGET_UPDATE = 10
    MAX_ITERATIONS = 100
    num_episodes = 50
    learning_rate = 0.1


class rewards:
    win = 100
    death = -100
    #something is win
    canWin = 10
    default = -0.01

