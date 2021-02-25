lValues = ["bt","ft","kt"]
rValues = ["wt","yt"]
stopList = ["wo"]
pushList = ["ro","kt","bt","ft","is","wt","yt"]

def updateRules(state,rule_list,you_list,win_list):
    nrow,ncol = len(state),len(state[0])
    rule_list.clear()
    you_list.clear()
    win_list.clear()
    for row in range(0,nrow):
        for col in range(0,ncol):
            if "is" in state[row][col]:
                if(row>0 and row<nrow-1):
                    ll,rl = state[row-1][col],state[row+1][col]
                    for lv in ll:
                        for rv in rl:
                            if lv in lValues and rv in rValues:
                                rule_list.append([lv,"is",rv])
                if(col>0 and col<ncol-1):
                    ll,rl = state[row][col-1],state[row][col+1]
                    for lv in ll:
                        for rv in rl:
                            if lv in lValues and rv in rValues:
                                rule_list.append([lv,"is",rv])
    for rule in rule_list:
        if rule[2] == "wt":
            if rule[0] not in win_list:
                win_list.append(rule[0].replace("t","o"))
        if rule[2] == "yt":
            if rule[0] not in you_list:
                you_list.append(rule[0].replace("t","o"))

def printRules(state):
    rule_list = []
    you_list = []
    win_list = []
    updateRules(state,rule_list,you_list,win_list)
    print("Rule list :")
    for rule in rule_list:
        print(rule[0]+" "+rule[1]+" "+rule[2])
    print("You list :")
    print(you_list)
    print("Win list :")
    print(win_list)

#       0
#      3  1
#       2

def isStop(list):
    for elem in list:
        if elem in stopList:
            return True
    return False

def isPush(list):
    for elem in list:
        if elem in pushList:
            return True
    return False

def isYou(list, you_list):
    for elem in list:
        if elem in you_list:
            return True
    return False

def isWin(list, win_list):
    for elem in list:
        if elem in win_list:
            return True
    return False

# Return (observation, reward, done)
def step(state,action):
    rule_list = []
    you_list = []
    win_list = []
    updateRules(state,rule_list,you_list,win_list)
    first = True
    newState = stateToUp(state,action)
    nrow,ncol = len(newState),len(newState[0])
    for col in range(1,ncol):
        for row in range(nrow):
            if isYou(newState[row][col],you_list):
                emptyFound = False
                dist = 1
                while True:
                    if col-dist < 0 or isStop(newState[row][col-dist]):
                        break
                    elif isPush(newState[row][col-dist]):
                        dist+=1
                        continue
                    else:
                        emptyFound = True
                        break
                if emptyFound:
                    for i in range(1,dist):
                        j=dist-i
                        size= len(newState[row][col-j])
                        for i in range(size):
                            k = size-1-i
                            if newState[row][col-j][k] in pushList:
                                newState[row][col-j-1].append(newState[row][col-j][k])
                                del newState[row][col-j][k]
                    size= len(newState[row][col])
                    for i in range(size):
                        k = size-1-i
                        if newState[row][col][k] in you_list:
                            newState[row][col-1].append(newState[row][col][k])
                            del newState[row][col][k]
    
    state = stateToUp(newState,(4-action)%4)
    simplify(state)
    
    reward = 1. if isWinState(newState) else 0.
    return newState, reward, isFinalState(newState)

def stateToUp(state,action):
    newState = []
    nrow,ncol = len(state),len(state[0])
    if action%2==1:
        ncol,nrow=nrow,ncol
    for row in range(0,nrow):
        newRow = []
        for col in range(0,ncol):
            i,j=row,col
            if action == 3:
                i,j=col,nrow-1-row
            elif action == 2:
                i,j=nrow-1-row,ncol-1-col
            elif action == 1:
                i,j=ncol-1-col,row
            newRow.append(state[i][j])
        newState.append(newRow)
    return newState

def simplify(state):
    for row in state:
        for tile in row:
            size=len(tile)
            for i in range(size):
                k = size-1-i
                if tile[k] == "no":
                    del tile[k]
                else:
                    for j in range(k):
                        if tile[j]==tile[k]:
                            del tile[k]
                            break

def isDeathState(state):
    rule_list = []
    you_list = []
    win_list = []
    updateRules(state,rule_list,you_list,win_list)
    if len(you_list)==0:
        return True
    for row in state:
        for elem in row:
            if isYou(elem,you_list):
                return False
    return True

def isWinState(state):
    rule_list = []
    you_list = []
    win_list = []
    updateRules(state,rule_list,you_list,win_list)
    for row in state:
        for elem in row:
            if isYou(elem,you_list) and isWin(elem,win_list):
                return True
    return False
                
def isFinalState(state):
    return isWinState(state) or isDeathState(state)

def getActionList():
    return [0,1,2,3]
