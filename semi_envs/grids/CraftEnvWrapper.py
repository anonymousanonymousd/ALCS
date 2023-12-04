
class CraftWapper:
    def __init__(self, env, task, max_time=1000):
        self.env = env
        self.env.init_map(task)
        self.max_time = max_time
        self.current_t = 0
        self.action_space = env.action_space
        self.label = ''

    def reset(self):
        self.current_t = 0
        self.label = ''
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.label = info['label']
        self.current_t = self.current_t + 1
        if self.current_t >= self.max_time:
            done = True
        return obs, reward, done, info

    def get_dict_prop(self, all_prop):
        i = 0
        dict_prop = {}
        chtoint = {}
        for prop in all_prop:
            dict_prop[i] = prop
            chtoint[prop] = i
            i+=1
        return dict_prop, chtoint


    def get_all_prop(self):
        if self.env.mode == 'selling':
            return ['a', 'b', 's']
        all_prop = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        # for v in self.env.objects.values():
        #     if v not in all_prop:
        #         all_prop.append(v)
        return all_prop

    def label_function(self):
        # print(self.env.get_true_propositions())
        return self.label

    def render(self, mode='human'):
        if mode == 'human':
            # commands
            str_to_action = {"w": 0, "d": 1, "s": 2, "a": 3}

            # play the game!
            done = True
            while True:
                if done:
                    print("New episode --------------------------------")
                    obs = self.reset()
                    self.env.show()
                    print("Features:", obs)

                print('Label function: {}'.format(self.label_function()))
                print(self.env.get_hand())
                print("\nAction? (WASD keys or q to quite) ", end="")
                a = input()
                print()
                if a == 'q':
                    break
                # Executing action
                if a in str_to_action:
                    obs, rew, done, _ = self.step(str_to_action[a])
                    self.env.show()
                    print("Features:", obs)
                    print("Reward:", rew)
                else:
                    print("Forbidden action")
        else:
            raise NotImplementedError