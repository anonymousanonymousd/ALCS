from envs.grids.game_objects import *
import random, math, os
import numpy as np
import gym
from gym import spaces

class CraftWorld(gym.Env):

    def __init__(self):
        ###
        # task 1: ab ~ @
        # task 2: ac ~
        # task 3: de ~
        # task 4: db ~
        # task 5: afe ~ @
        # task 6: abdc ~ @
        # task 7: acfb ~
        # task 8:
        # task 9: afeg ~
        # task 10: afcbh @
        self._hand = []
        self.action_space = spaces.Discrete(4)
        self.label = ''
        self.env_game_over = False

    def init_map(self, task):
        if task == 'plant':
            self.file_map = './envs/grids/maps/map_ab.txt'
            self.mode = 'ab'
        elif task == 'bridge':
            self.file_map = './envs/grids/maps/map_afe.txt'
            self.mode = 'afe'
        elif task == 'bed':
            self.file_map = './envs/grids/maps/map_abdc.txt'
            self.mode = 'abdc'
        elif task == 'gem':
            self.file_map = './envs/grids/maps/map_afcbh.txt'
            self.mode = 'afcbh'
        else:
            raise NotImplementedError

        self._load_map(self.file_map)

    def reset(self):
        self.agent.reset()
        self._hand = []
        return self.get_features()

    def get_hand(self):
        return self._hand

    def step(self, action):
        self.label = ''
        self.execute_action(action)
        obs = self.get_features()
        label = ''
        if not (self.get_true_propositions()=="") and self.get_true_propositions() not in self._hand:
            label = self.get_true_propositions()
            if self.mode == 'selling' and label=='b':
                if 'a' in self._hand:
                    self._hand.remove('a')
                    self._hand.append('k')
                self.label = label
            if self.mode == 'selling' and label == 'a':
                self._hand.append('a')
                self.label = label

            if self.mode != 'selling':
                # self.agent = (4, 4)
                self._hand.append(label)
                self.label = label
        reward, done = self.reward_func()
        info = {'label':label}
        return obs, reward, done, info

    def reward_func(self):
        if self.mode == 'ab':
            if self.label == 'b':
                if 'a' in self._hand:
                    reward = 1
                    done = True
                else:
                    reward = 0
                    done = True
            else:
                reward = 0
                done = False
            return reward, done
        if self.mode == 'ac':
            if self.label == 'c':
                if 'a' in self._hand:
                    reward = 1
                    done = True
                else:
                    reward = 0
                    done = True
            else:
                reward = 0
                done = False
            return reward, done
        if self.mode == 'de':
            if self.label == 'e':
                if 'd' in self._hand:
                    reward = 1
                    done = True
                else:
                    reward = 0
                    done = True
            else:
                reward = 0
                done = False
            return reward, done
        if self.mode == 'db':
            if self.label == 'b':
                if 'd' in self._hand:
                    reward = 1
                    done = True
                else:
                    reward = 0
                    done = True
            else:
                reward = 0
                done = False
            return reward, done
        if self.mode == 'afe':
            if self.label == 'e':
                if 'a' in self._hand and 'f' in self._hand:
                    reward = 1
                    done = True
                else:
                    reward = 0
                    done = True
            else:
                reward = 0
                done = False
            return reward, done
        if self.mode == 'abdc':
            if self.label == 'c':
                if 'a' in self._hand and 'b' in self._hand and 'd' in self._hand:
                    if self._hand.index('a') < self._hand.index('b'):
                        reward = 1
                        done = True
                    else:
                        reward = 0
                        done = True
                else:
                    reward = 0
                    done = True
            else:
                reward = 0
                done = False
            return reward, done
        if self.mode == 'acfb':
            if self.label == 'b':
                if 'a' in self._hand and 'c' in self._hand and 'f' in self._hand:
                    if self._hand.index('a') < self._hand.index('c'):
                        reward = 1
                        done = True
                    else:
                        reward = 0
                        done = True
                else:
                    reward = 0
                    done = True
            else:
                reward = 0
                done = False
            return reward, done
        if self.mode == 'afeg':
            if self.label == 'g':
                if 'a' in self._hand and 'f' in self._hand and 'e' in self._hand:
                    if self._hand.index('a') < self._hand.index('e') and self._hand.index('f') < self._hand.index('e'):
                        reward = 1
                        done = True
                    else:
                        reward = 0
                        done = True
                else:
                    reward = 0
                    done = True
            else:
                reward = 0
                done = False
            return reward, done
        if self.mode == 'afcbh':
            if self.label == 'h':
                if 'a' in self._hand and 'f' in self._hand and 'c' in self._hand and 'b' in self._hand:
                    if self._hand.index('f') < self._hand.index('b'):
                        reward = 1
                        done = True
                    else:
                        reward = 0
                        done = True
                else:
                    reward = 0
                    done = True
            else:
                reward = 0
                done = False
            return reward, done
        if self.mode == 'selling':
            if self.label == 's':
                if 'k' in self._hand:
                    reward = self._hand.count('k')
                    done = True
                else:
                    reward = 0
                    done = True
            else:
                reward = 0
                done = False
            return reward, done
        else:
            raise NotImplementedError

    def execute_action(self, a):
        """
        We execute 'action' in the game
        """
        agent = self.agent
        ni, nj = agent.i, agent.j

        # Getting new position after executing action
        ni, nj = self._get_next_position(ni, nj, a)

        # Interacting with the objects that is in the next position (this doesn't include monsters)
        action_succeeded = self.map_array[ni][nj].interact(agent)

        # So far, an action can only fail if the new position is a wall
        if action_succeeded:
            agent.change_position(ni, nj)

    def _get_next_position(self, ni, nj, a):
        """
        Returns the position where the agent would be if we execute action
        """
        action = Actions(a)

        # OBS: Invalid actions behave as NO-OP
        if action == Actions.up: ni -= 1
        if action == Actions.down: ni += 1
        if action == Actions.left: nj -= 1
        if action == Actions.right: nj += 1

        return ni, nj

    def get_true_propositions(self):
        """
        Returns the string with the propositions that are True in this state
        """
        ret = str(self.map_array[self.agent.i][self.agent.j]).strip()
        return ret

    def get_features(self):
        """
        Returns the features of the current state (i.e., the location of the agent)
        """
        return np.array([self.agent.i, self.agent.j])

    def show(self):
        """
        Prints the current map
        """
        r = ""
        for i in range(self.map_height):
            s = ""
            for j in range(self.map_width):
                if self.agent.idem_position(i, j):
                    s += str(self.agent)
                else:
                    s += str(self.map_array[i][j])
            if (i > 0):
                r += "\n"
            r += s
        print(r)

    def _load_map(self, file_map):
        """
        This method adds the following attributes to the game:
            - self.map_array: array containing all the static objects in the map (no monsters and no agent)
                - e.g. self.map_array[i][j]: contains the object located on row 'i' and column 'j'
            - self.agent: is the agent!
            - self.map_height: number of rows in every room
            - self.map_width: number of columns in every room
        The inputs:
            - file_map: path to the map file
        """
        # contains all the actions that the agent can perform
        self.actions = [Actions.up.value, Actions.right.value, Actions.down.value, Actions.left.value]
        # loading the map
        self.map_array = []
        self.class_ids = {}  # I use the lower case letters to define the features
        f = open(file_map)
        i, j = 0, 0
        for l in f:
            # I don't consider empty lines!
            if (len(l.rstrip()) == 0): continue

            # this is not an empty line!
            row = []
            j = 0
            for e in l.rstrip():
                if e in "abcdefghijklmnopqrstuvwxyzH":
                    entity = Empty(i, j, label=e)
                    if e not in self.class_ids:
                        self.class_ids[e] = len(self.class_ids)
                if e in " A":  entity = Empty(i, j)
                if e == "X":   entity = Obstacle(i, j)
                if e == "A":   self.agent = Agent(i, j, self.actions)
                row.append(entity)
                j += 1
            self.map_array.append(row)
            i += 1
        f.close()
        # height width
        self.map_height, self.map_width = len(self.map_array), len(self.map_array[0])

