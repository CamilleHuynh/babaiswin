#game parameters
class env:
    background_color = (225, 225, 225)
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


#learning parameters
class learning_param:
    BATCH_SIZE = 128
    GAMMA = 0.9
    EPSILON = 0.2
    TARGET_UPDATE = 10
    num_episodes = 10

class rewards:
    win = 10
    death = -10
    unnecessary = -2
    default = -1