###
# 一些尚不清晰的点：
#    尚不知道任务终止条件和输入goal的关系，源代码里任务终止是由model决定的
#    尚不知道reward计算方式，是否是二进制的reward
#    能输出背包中物品的index，尚不清楚如何将index与具体物品关联
#    尚不清楚环境中减少物品和背包中增加物品的具体运算过程，因为
#        —— 需要变更代码是环境中资源非一次性消耗（有量）
#        —— 需要弄清背包运算方式，在环境中添加商店以卖掉物品（或游戏结束时结算物品总价值作为reward）
###

WINDOW_WIDTH = 5
WINDOW_HEIGHT = 5

N_WORKSHOPS = 3

DOWN = 0
UP = 1
LEFT = 2
RIGHT = 3
USE = 4
N_ACTIONS = USE + 1

class CraftWapper:
    def __init__(self, world, goal_arg):
        # for single task, world+goal_arg is a task
        self.WIDTH = 10
        self.HEIGHT = 10
        self.world = world
        self.scenario = self.world.sample_scenario_with_goal(goal_arg)
        self.state = None
        self.state_str = []
        self.state_str_x = []
        for y in range(self.HEIGHT):
            for x in range(self.WIDTH):
                self.state_str_x.append('-'+'-')
            self.state_str.append(self.state_str_x)
            self.state_str_x = []


    def reset(self):
        self.state = self.scenario.init()
        return self.state

    def initize_str(self):
        for y in range(self.HEIGHT):
            for x in range(self.WIDTH):
                self.state_str_x.append('-'+'-')
            self.state_str.append(self.state_str_x)
            self.state_str_x = []

    def step(self, action):
        reward, next_state = self.state.step(action)
        self.state = next_state
        done = self._is_terminate()
        info = {}
        return self.state, reward, done, info

    def _is_terminate(self):
        return False

    def render(self, action):
        self.initize_str()

        for y in range(self.HEIGHT):
            for x in range(self.WIDTH):
                # if not (self.state.grid[x, y, :].any() or (x, y) == self.state.pos):
                #     continue
                thing = self.state.grid[x, y, :].argmax()
                # ch1 = ch2 = '-'
                # print(self.state.pos)
                if (x, y) == self.state.pos:
                    if action == LEFT:
                        ch1 = "<"
                        ch2 = "@"
                    elif action == RIGHT:
                        ch1 = "@"
                        ch2 = ">"
                    elif action == UP:
                        ch1 = "^"
                        ch2 = "@"
                    elif action == DOWN:
                        ch1 = "@"
                        ch2 = "v"
                    else:
                        ch1 = "@"
                        ch2 = "@"
                    # color = curses.color_pair(mstate.arg or 0)
                    # color = curses.color_pair(0)
                elif thing == self.world.cookbook.index["boundary"]:
                    ch1 = ch2 = '|'
                    # color = curses.color_pair(10 + thing)
                else:
                    name = self.world.cookbook.index.get(thing)
                    ch1 = name[0]
                    ch2 = name[-1]
                    # color = curses.color_pair(10 + thing)

                self.state_str[y][x] = ch1+ch2

        self._print()
        print('current inventory: {}'.format(self.state.inventory))


    def _print(self):
        for y in range(self.HEIGHT):
            print(self.state_str[self.HEIGHT-1-y])






