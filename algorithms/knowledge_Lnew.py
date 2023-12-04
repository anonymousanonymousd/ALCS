import random

import gym
from baselines import logger
from semi_envs.grids.OfficeEnvWrapper import OfficeWapper, CollectionWrapper
from semi_envs.grids.CraftEnvWrapper import CraftWapper
import numpy as np
import math


class Node:
    def __init__(self, varp, parent=None):
        self.visits = 0
        self.varp = varp
        self.children = []
        self.parent = parent
        self.achieve_set = self.getAchieveSet()

    def visit(self):
        self.visits += 1

    def append_children(self, all_var):
        for v in all_var:
            n = Node(v, parent=self)
            self.children.append(n)

    def getAchieveSet(self):
        achieved = set()
        tmp = self
        while not tmp.parent is None:
            achieved.add(tmp.varp)
            tmp = tmp.parent
        return achieved

    def select_for_explore(self, a_v_dict, C=0.7):
        if self.children == []:
            raise NotImplementedError
        else:
            visit_num = [n.visits for n in self.children]
            visit_num = [0.9 if n==0 else n for n in visit_num]
            eevalue = []
            for i in range(len(visit_num)):
                try:
                    eevalue.append(a_v_dict[i]+C*math.sqrt(math.log(self.visits)/visit_num[i]))
                except KeyError:
                    print(i)
                    print(a_v_dict.keys())
                    exit(0)


            # for i in range(len(eevalue)):
            #     if self.children[i].varp in self.achieve_set:
            #         eevalue[i] = -99999
            visit_log = np.array(eevalue)
            min_id = np.argmax(visit_log)
            return self.children[min_id]

    def go_to_child(self, child_varp):
        if self.children == []:
            raise NotImplementedError
        else:
            child_name = [n.varp for n in self.children]
            index_child = child_name.index(child_varp)
            return self.children[index_child]

    def print_tree(self):
        str = ""
        if not self.parent is None:
            tmp = self.parent
            str += self.parent.varp
            while not tmp.parent is None:
                str += tmp.parent.varp
                tmp = tmp.parent

        print("Node:{}; visits: {}; parent: {}".format(self.varp, self.visits,str))
        if self.children != []:
            for i in self.children:
                i.print_tree()


class Dict_Count:
    def __init__(self):
        self.key_num_dict = {}

    def count_key(self, akey):
        if akey not in self.key_num_dict.keys():
            self.key_num_dict[akey] = 1
        else:
            self.key_num_dict[akey] = self.key_num_dict[akey] + 1

    def print_dict(self):
        for k,v in self.key_num_dict:
            print("key: {}, value: {}".format(k, v))

    def print_v_avg(self):
        values = []
        for _,v in self.key_num_dict:
            values.append(v)
        values = np.array(values)
        print(values.mean())


def get_qmax(Q,s,actions,q_init):
    # s = <s, g> for low, <s> for high
    if s not in Q:
        Q[s] = dict([(a,q_init) for a in actions])
    return max(Q[s].values())


def get_best_action(Q,s,actions,q_init):
    # s = <s, g> for low, <s> for high
    qmax = get_qmax(Q,s,actions,q_init)
    best = [a for a in actions if Q[s][a] == qmax]
    return random.choice(best)


def get_best_action_h(Q,s,actions,q_init=-1.0):
    # s = <s, g> for low, <s> for high
    qmax = get_qmax(Q,s,actions,q_init)
    try:
        best = [a for a in actions if Q[s][a] == qmax]
        return random.choice(best)
    except KeyError:
        print(s)
        print(actions)
        exit(0)
    except IndexError:
        print(actions)
        print(qmax)
        print(Q[s].values())
        print(Q[s].keys())
        exit(0)


def evaluate_er(Q_H, Q_L, args, total_e=10000, q_init=2.0):
    if args.world == 'office':
        env = gym.make("Office-v1")
        env = OfficeWapper(env, args.task)
    elif args.world == 'craft':
        env = gym.make("Craft-v1")
        env = CraftWapper(env, args.task)
    actions = list(range(env.action_space.n))
    props = env.get_all_prop()
    goals = list(range(len(props)))
    gs = []
    r_tol = 0
    i = 0

    step_tol = 0

    while i <= total_e:
        done = False
        s = tuple(env.reset())
        stat_his = ('n', )
        while not done:
            s_h = tuple(stat_his)
            g = get_best_action_h(Q_H, (s, s_h), goals)

            s_l = s + (g, )
            a = get_best_action(Q_L, s_l, actions, q_init)
            sn, r, done, info = env.step(a)
            r_tol += r
            i += 1
            l = env.label_function()
            if not l == '' and l not in stat_his:
                stat_his += (l, )
            sn = tuple(sn)
            if done:
                break
            s = sn
    # print(gs)
    return r_tol


def evaluate_win_rate(Q_H, Q_L, args, total_e=50, q_init=2.0):
    env = gym.make(args.world)
    if args.world=="Office-v1":
        env = OfficeWapper(env, args.task)
    elif args.world=="Craft-v1":
        env = CraftWapper(env, args.task)
    actions = list(range(env.action_space.n))
    props = env.get_all_prop()
    goals = list(range(len(props)))
    gs = []
    r_tol = 0

    step_tol = 0

    for i in range(total_e):
        done = False
        s = tuple(env.reset())
        stat_his = ('n', )
        while not done:
            s_h = tuple(stat_his)
            # print(goals)
            # try:
            g = get_best_action_h(Q_H, (s, s_h), goals)
            # except IndexError:
            #     print(props)
            #     print(goals)
            #     exit(0)

            s_l = s + (g, )
            if i==0:
                gs.append(goals[g])
            a = get_best_action(Q_L, s_l, actions, q_init)
            sn, r, done, info = env.step(a)
            step_tol += 1
            l = env.label_function()
            if not l == '' and l not in stat_his:
                stat_his += (l, )
            sn = tuple(sn)
            if done:
                r_tol += r
                break
            s = sn
    # print(gs)
    return r_tol/total_e

def r_for_p(prop, ln):
    if prop == ln:
        return 1
    else:
        return 0

def r_for_p_avoid(prop, ln):
    if ln == '': return 0
    elif ln == prop: return 1
    else: return -1

def phi(g, l, props_dict):
    if l == props_dict[g]:
        return 1
    else:
        return 0


def shape_low_reward(g, gn, l, ln, r, gamma, props_dict):
    r_l = r + gamma * phi(gn, ln, props_dict) - phi(g, l, props_dict)
    return r_l


def learnp(env,
          args,
          lr=0.1,
          total_timesteps=100000,
          epsilon=0.2,
          print_freq=10000,
          gamma=0.9,
          q_init=2.0,
          q_init_h=-1.0,
          use_crm=False,
          use_rs=False,
          C=1.0,
           gamma_h=0.99):

    # get props from env
    props = env.get_all_prop()

    props_dict, props_chartoint = env.get_dict_prop(props)
    # print(props)
    # print(props_dict)
    # print(props_chartoint)
    # exit(0)

    # Running Q-Learning
    reward_total = 0
    step = 0
    num_episodes = 0
    Q_L = {}
    Q_H = {}
    actions = list(range(env.action_space.n))
    goals = list(range(len(props)))
    # print(goals)
    # exit(0)

    node_root = Node('n')
    node_root.append_children(props)

    step_list = []
    episode_list = []
    reward_tot_list = []
    win_list = []
    experiences_h = []

    while step < total_timesteps:
        s = tuple(env.reset())

        stat_his = ('n', )
        current_node = node_root
        current_node.visit()

        s_h = tuple(stat_his)
        s_his = [(s, )]
        t_his = [(env.current_t, )]
        if (s, s_h) not in Q_H: Q_H[(s, s_h)] = dict([(g, q_init_h) for g in goals])
        g_list = []

        showfig = False
        # if step >1400000:
        #    showfig = True

        while True:
            # Select the goal
            s_h = tuple(stat_his)
            if (s, s_h) not in Q_H: Q_H[(s, s_h)] = dict([(g, q_init_h) for g in goals])

            # g type: int
            # g = random.choice(goals) if random.random() < epsilon else get_best_action_h(Q_H, s_h, goals)
            g_node = current_node.select_for_explore(Q_H[(s, s_h)], C=C)
            g_chr = g_node.varp

            g = props_chartoint[g_chr]
            g_list.append(g)

            s_l = s + (g, )
            if s_l not in Q_L: Q_L[s_l] = dict([(a, q_init) for a in actions])

            # Selecting and executing the action
            a = random.choice(actions) if random.random() < epsilon else get_best_action(Q_L, s_l, actions, q_init)
            sn, r, done, info = env.step(a)

            sn = tuple(sn)
            ln = env.label_function()
            if not ln == '' and ln not in stat_his:
                stat_his += (ln, )
                s_his.append(())
                t_his.append(())

                current_node = current_node.go_to_child(ln)
                current_node.visit()
                if current_node.children == []:
                    current_node.append_children(props)
                # print(current_node.children)
                # current_node.print_tree()
                # exit(0)

            s_his[-1] = s_his[-1] + (sn, )
            t_his[-1] = t_his[-1] + (env.current_t, )
            r_h = r

            sn_h = tuple(stat_his)
            if (s, sn_h) not in Q_H: Q_H[(s, sn_h)] = dict([(g, q_init_h) for g in goals])

            gn = random.choice(goals) if random.random() < epsilon else get_best_action_h(Q_H, (s, sn_h), goals)
            sn_l = sn + (gn, )
            if sn_l not in Q_L: Q_L[sn_l] = dict([(a, q_init) for a in actions])

            experiences_l = []

            for prop in props_dict.keys():
                _r_l = r_for_p_avoid(props_dict[prop], ln)
                _s_l = s + (prop,)
                _sn_l = sn + (prop,)
                experiences_l.append((_s_l, a, _r_l, _sn_l, done))


            if done:
                his_tup = tuple(stat_his) # (ln, )
                for i in range(1, len(his_tup)):
                    done_h = True if i==len(his_tup)-1 else False
                    experiences_h.append(((s_his[i - 1][len(s_his[i-1])-1], his_tup[:i]),
                                          props_chartoint[his_tup[i]],
                                          r_h,
                                          (s_his[i][0], his_tup[:i + 1]),
                                          done_h,
                                          t_his[i - 1][len(s_his[i-1])-1],
                                          t_his[i][0]))
                    for j in range(len(s_his[i-1])-1):
                        experiences_h.append(((s_his[i-1][j], his_tup[:i]),
                                              props_chartoint[his_tup[i]],
                                              r_h,
                                              (s_his[i][0], his_tup[:i]),
                                              done_h,
                                              t_his[i-1][j],
                                              t_his[i][0]))
                # if done and r > 0:
                #     for exp in experiences_h:
                #         print(exp)
                #     exit()

                for _s, _a, _r, _sn, _done, _t1, _t2 in experiences_h:
                    if _s not in Q_H: Q_H[_s] = dict([(b, q_init) for b in actions])
                    if _done:
                        _delta = _r - Q_H[_s][_a]
                    else:
                        # _delta = _r +  pow(gamma_h, _t2-_t1) * get_qmax(Q_H, _sn, actions, q_init) - Q_H[_s][_a]
                        _delta = _r + gamma_h * get_qmax(Q_H, _sn, actions, q_init) - Q_H[_s][_a]
                    Q_H[_s][_a] += lr * _delta
                experiences_h = []

            for _s,_a,_r,_sn,_done in experiences_l:
                if _s not in Q_L: Q_L[_s] = dict([(b,q_init) for b in actions])
                if _done: _delta = _r - Q_L[_s][_a]
                else:     _delta = _r + gamma*get_qmax(Q_L,_sn,actions,q_init) - Q_L[_s][_a]
                Q_L[_s][_a] += lr*_delta


            if showfig:
                print("current goal: {}: {}".format(g, props_dict[g]))
                print("current feature: ")
                env.env.show()
                if done:
                    print(g_list)
                    for prop in props_dict.keys():
                        print("{}:{}".format(prop, props_dict[prop]))
                    exit()

            reward_total += r
            step += 1
            if step%print_freq == 0:
                step_tol = evaluate_er(Q_H, Q_L, args)
                logger.record_tabular("steps", step)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("total reward", reward_total)
                logger.record_tabular("step_test_avg", step_tol)
                logger.dump_tabular()
                step_list.append(step)
                reward_tot_list.append(reward_total)
                episode_list.append(num_episodes)
                win_list.append(step_tol)
                reward_total = 0
            if done:
                num_episodes += 1
                break
            s = sn
    return step_list, reward_tot_list, win_list
