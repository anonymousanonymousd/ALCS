import gym
from collections import defaultdict
import re
import argparse


from gym.envs.registration import register
register(
    id='Office-v1',
    entry_point='semi_envs.grids.grid_environment:OfficeEnv',
    max_episode_steps=1000,
    kwargs={}
)
register(
    id='Craft-v1',
    entry_point='semi_envs.grids.craft_env:CraftWorld',
    max_episode_steps=1000,
    kwargs={}
)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', default='office', type=str, help='domain')
    parser.add_argument('--task', default='fg', type=str, help='task')
    args = parser.parse_args()
    if args.world == 'office':
        env = gym.make("Office-v1")
        from semi_envs.grids.OfficeEnvWrapper import OfficeWapper
        env = OfficeWapper(env, args.task)
    elif args.world == 'craft':
        env = gym.make("Craft-v1")
        from semi_envs.grids.CraftEnvWrapper import CraftWapper
        env = CraftWapper(env, args.task)
    else:
        raise NotImplementedError

    #env.render()
    #exit()
    from algorithms.knowledge_Lnew import learnp
    log_list_step = []
    log_list_reward = []
    print('here')
    log_list_win = []
    for i in range(20):
        print("===========================EPOCH: {}=============================".format(i))
        step, reward, win = learnp(env, args, total_timesteps=1500000, C=0.0)  # 1500000
        log_list_step.append(step)
        log_list_reward.append(reward)
        log_list_win.append(win)

    exit()
    import numpy as np
    log_list_step = np.array(log_list_step)
    log_list_reward = np.array(log_list_reward)
    log_list_win = np.array(log_list_win)




