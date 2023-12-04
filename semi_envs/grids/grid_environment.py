import random

import gym
from gym import spaces
import numpy as np
from envs.grids.game_objects import Actions


class OfficeEnv(gym.Env):

    def __init__(self):
        self.map_height, self.map_width = 12,9
        N, M = self.map_height, self.map_width
        self.action_space = spaces.Discrete(4) # up, right, down, left
        self.observation_space = spaces.Box(low=0, high=max([N,M]), shape=(2,), dtype=np.uint8)
        self._hand = []
        self.is_random_start = False
        self.bag = True
        

    def init_map(self, task):
        self.mode = task
        self._load_map()

    def reset(self):
        if self.is_random_start:
            i = random.randint(0, self.map_height - 1)
            j = random.randint(0, self.map_width - 1)
            while (i, j) in self.objects.keys():
                i = random.randint(0, 12)
                j = random.randint(0, 9)
            self.agent = (i, j)
        else:
            self.agent = (2, 1)  # (10, 7)
        self._hand = []
        return self.get_features()

    def reset_withp(self, p=(2, 1)):
        self.agent = p
        return self.get_features()

    def step(self, action):
        self.execute_action(action)
        obs = self.get_features()
        label = ''
        if self.bag == True:
            if not (self.get_true_propositions()==""):
                if self.get_true_propositions() not in self._hand:
                    label = self.get_true_propositions()
                    # self.agent = (4, 4)
                    self._hand.append(label)
        else:
            if not (self.get_true_propositions()==""):
                label = self.get_true_propositions()
                # self.agent = (4, 4)
                self._hand.append(label)


        reward, done = self.reward_func()
        info = {'label':label}
        return obs, reward, done, info

    def reward_func(self):
        if self.mode == 'g':
            if self.agent == (4, 4):
                reward = 1
            else:
                reward = 0
            done = False if reward == 0 else True
        elif self.mode == 'e':
            if self.agent == (7, 4):
                reward = 1
            else:
                reward = 0
            done = False if reward == 0 else True
        elif self.mode == 'eg':
            if self.agent == (4, 4) and 'e' in self._hand:
                reward = 1
                done = True
            else:
                reward = 0
                done = False
        elif self.mode == 'fg-orig':
            if self.agent == (4, 4) and 'f' in self._hand:
                reward = 1
                done = True
            else:
                reward = 0
                done = False
        elif self.mode == 'fg':
            if self.agent == (4, 4):
                if 'f' in self._hand:
                    reward = 1
                    done = True
                else:
                    reward = 0
                    done = True
            else:
                reward = 0
                done = False
        elif self.mode == 'efg':
            if self.agent == (4, 4) and 'e' in self._hand and 'f' in self._hand:
                reward = 1
                done = True
            else:
                reward = 0
                done = False
        elif self.mode == 'c4':
            if self.agent == (4, 4):
                if 'a' in self._hand and 'b' in self._hand and 'c' in self._hand and 'd' in self._hand:
                    reward = 1
                    done = True
                else:
                    reward = 0
                    done = True
            else:
                reward = 0
                done = False
        elif self.mode == 'bonus':
            if self.agent == (4, 4):
                reward = 0
                if 'a' in self._hand:
                    reward += 1
                if 'b' in self._hand:
                    reward += 1
                if 'c' in self._hand:
                    reward += 1
                if 'd' in self._hand:
                    reward += 1
                if 'a' in self._hand and 'b' in self._hand and 'c' in self._hand and 'd' in self._hand:
                    reward += 5
                done = True
            else:
                reward = 0
                done = False
        else:
            raise NotImplementedError
        return reward, done



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
        if self.agent in self.objects:
            ret += self.objects[self.agent]
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

    def get_model(self):
        """
        This method returns a model of the environment.
        We use the model to compute optimal policies using value iteration.
        The optimal policies are used to set the average reward per step of each task to 1.
        """
        S = [(x,y) for x in range(12) for y in range(9)] # States
        A = self.actions.copy() # Actions
        L = self.objects.copy() # Labeling function
        T = {}                  # Transitions (s,a) -> s' (they are deterministic)
        for s in S:
            x,y = s
            for a in A:
                T[(s,a)] = self._get_new_position(x,y,a)
        return S,A,L,T # SALT xD

    def _load_map(self):
        # Creating the map
        self.objects = {}
        # self.objects[(1,1)] = "a"
        # self.objects[(1,7)] = "b"
        # self.objects[(10,7)] = "c"
        # self.objects[(10,1)] = "d"
        self.objects[(4, 4)] = "g"  # OFFICE
        if  self.mode == 'efg':
            self.objects[(7, 4)] = "e"  # MAIL
            self.objects[(8,2)] = "f"  # COFFEE
            self.objects[(3,6)] = "f"  # COFFEE
        elif self.mode == 'fg':
            self.objects[(3,6)] = "f"  # COFFEE
        elif self.mode == 'eg':
            self.objects[(7, 4)] = "e"  # MAIL
        elif self.mode == 'c3' or self.mode == 'c4' or self.mode == 'bonus':
            self.objects[(0, 0)] = "d"
            self.objects[(0, 8)] = "a"
            self.objects[(11, 8)] = "b"
            self.objects[(11, 0)] = "c"
            
        # self.objects[(4,1)] = "n"  # PLANT
        # self.objects[(7,1)] = "n"  # PLANT
        # self.objects[(4,7)] = "n"  # PLANT
        # self.objects[(7,7)] = "n"  # PLANT
        # self.objects[(1,4)] = "n"  # PLANT
        # self.objects[(10,4)] = "n" # PLANT
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
        for y in [1,7]:
            for x in [2,5,8]:
                self.forbidden_transitions.remove((x,y,Actions.right))
                self.forbidden_transitions.remove((x+1,y,Actions.left))
        for x in [1,4,7,10]:
            self.forbidden_transitions.remove((x,5,Actions.up))
            self.forbidden_transitions.remove((x,6,Actions.down))
        for x in [1,10]:
            self.forbidden_transitions.remove((x,2,Actions.up))
            self.forbidden_transitions.remove((x,3,Actions.down))
        # Adding the agent
        self.actions = [Actions.up.value,Actions.right.value,Actions.down.value,Actions.left.value]
