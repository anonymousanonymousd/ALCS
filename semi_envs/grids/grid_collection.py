import gym
from gym import spaces
import numpy as np
from envs.grids.game_objects import Actions


def merge_sort(s, l, r):
    if l >= r: return 0
    mid = l + r >> 1
    res = merge_sort(s,l,mid) + merge_sort(s,mid + 1,r)
    i = l
    j = mid + 1
    tmp = []
    while(i <= mid and j <= r):
        if(s[i] <= s[j]):
            tmp.append(s[i])
            i += 1
        else:
            res += (mid - i + 1)
            tmp.append(s[j])
            j += 1
            pass
        pass
    while(i <= mid):
        tmp.append(s[i])
        i += 1
        pass
    while(j <= r):
        tmp.append(s[j])
        j += 1
        pass
    s[l:r + 1] = tmp
    return res


class CollectionEnv(gym.Env):

    def __init__(self):
        self.mode = 7
        self._load_map()
        self.map_height, self.map_width = 12,9
        N, M = self.map_height, self.map_width
        self.action_space = spaces.Discrete(4) # up, right, down, left
        self.observation_space = spaces.Box(low=0, high=max([N,M]), shape=(2,), dtype=np.uint8)
        self._hand = []


    def reset(self):
        self.agent = (4,4)
        self._hand = []
        return self.get_features()

    def step(self, action):
        self.execute_action(action)
        obs = self.get_features()
        label = ''
        if not (self.get_true_propositions()==""):
            if self.get_true_propositions() not in self._hand:
                label = self.get_true_propositions()
                # self.agent = (4, 4)
                self._hand.append(label)
        reward, done = self.reward_func()
        info = {'label':label}
        return obs, reward, done, info

    def reward_func(self):
        if self._hand == []:
            return 0, False
        if self._hand[-1] != 'g':
            return 0, False
        else:
            hand = self._hand[:-1]
            hand = [int(i) for i in hand]
            hand = hand[::-1]
            res = merge_sort(hand, 0, len(hand)-1)
            return res, True


    def execute_action(self, a):
        """
        We execute 'action' in the game
        """
        x,y = self.agent
        self.agent = self._get_new_position(x,y,a)

    def _get_new_position(self, x, y, a):
        action = Actions(a)
        # executing action
        if (x,y,action) not in self.forbidden_transitions:
            if action == Actions.up   : y+=1
            if action == Actions.down : y-=1
            if action == Actions.left : x-=1
            if action == Actions.right: x+=1
        return x,y


    def get_true_propositions(self):
        """
        Returns the string with the propositions that are True in this state
        """
        ret = ""
        if self.agent in self.objects and self.objects[self.agent] not in self._hand:
            ret += str(self.objects[self.agent])
        return ret

    def get_features(self):
        """
        Returns the features of the current state (i.e., the location of the agent)
        """
        x,y = self.agent
        return np.array([x,y])

    def show(self):
        for y in range(8,-1,-1):
            if y % 3 == 2:
                for x in range(12):
                    if x % 3 == 0:
                        print("_",end="")
                        if 0 < x < 11:
                            print("_",end="")
                    if (x,y,Actions.up) in self.forbidden_transitions:
                        print("_",end="")
                    else:
                        print(" ",end="")
                print()
            for x in range(12):
                if (x,y,Actions.left) in self.forbidden_transitions:
                    print("|",end="")
                elif x % 3 == 0:
                    print(" ",end="")
                if (x,y) == self.agent:
                    print("A",end="")
                elif (x,y) in self.objects:
                    print(self.objects[(x,y)],end="")
                else:
                    print(" ",end="")
                if (x,y,Actions.right) in self.forbidden_transitions:
                    print("|",end="")
                elif x % 3 == 2:
                    print(" ",end="")
            print()
            if y % 3 == 0:
                for x in range(12):
                    if x % 3 == 0:
                        print("_",end="")
                        if 0 < x < 11:
                            print("_",end="")
                    if (x,y,Actions.down) in self.forbidden_transitions:
                        print("_",end="")
                    else:
                        print(" ",end="")
                print()


    def _load_map(self):
        # Creating the map
        self.objects = {}
        self.objects_p = [(1,1), (7,7), (10,1), (1,7), (4,7), (7,1), (4,1)]
        for i in range(self.mode):
            self.objects[self.objects_p[i]] = i + 1

        self.objects[(10,7)] = "g"  # OFFICE
        # Adding walls
        self.forbidden_transitions = set()
        # general grid
        for x in range(12):
            for y in [0,3,6]:
                self.forbidden_transitions.add((x,y,Actions.down))
                self.forbidden_transitions.add((x,y+2,Actions.up))
        for y in range(9):
            for x in [0,3,6,9]:
                self.forbidden_transitions.add((x,y,Actions.left))
                self.forbidden_transitions.add((x+2,y,Actions.right))
        # adding 'doors'
        for y in [3, 4, 5]:
            for x in [2,5,8]:
                self.forbidden_transitions.remove((x,y,Actions.right))
                self.forbidden_transitions.remove((x+1,y,Actions.left))
        for x in [1,4,7,10]:
            self.forbidden_transitions.remove((x,5,Actions.up))
            self.forbidden_transitions.remove((x,6,Actions.down))
        for x in [1,4,7,10]:
            self.forbidden_transitions.remove((x,2,Actions.up))
            self.forbidden_transitions.remove((x,3,Actions.down))
        # Adding the agent
        self.actions = [Actions.up.value,Actions.right.value,Actions.down.value,Actions.left.value]
