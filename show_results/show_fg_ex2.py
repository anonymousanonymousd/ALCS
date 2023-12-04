import matplotlib.pyplot as plt
import numpy as np
str = '..\\data\\office_final\\'

font = {'size': 18}

def delete_max_min_two_in(arr):
    # Find the indices of the two maximum and two minimum elements
    max_indices = np.argpartition(arr, -2)[-2:]
    min_indices = np.argpartition(arr, 2)[:2]

    # Sort the indices in descending order for max values and ascending order for min values
    max_indices = max_indices[np.argsort(arr[max_indices])[::-1]]
    min_indices = min_indices[np.argsort(arr[min_indices])]

    # Combine the indices to be deleted
    indices_to_delete = np.concatenate((max_indices, min_indices))

    # Delete the four selected elements by their indices
    arr = np.delete(arr, indices_to_delete)

    return arr

def delete_max_min_two(win):
    win = win.T
    win_list = []
    for win_t in win:
        win_list.append(delete_max_min_two_in(win_t))
    win_list = np.array(win_list)
    win_list = win_list.T
    return win_list

step = np.load(str+'bouns\\ours_c0\\step.npy')
# episode = np.load('..\data\qlearn\qlearn_fg_episode.npy')
win = np.load(str+'bouns\\ours_c0\\reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0], win_mean, alpha=1, c='b', label='ours')
plt.fill_between(x=step[0], y1=win_min, y2=win_max, alpha=0.2, color='b')

step = np.load(str+'bouns\\deepsynth\\deepsynth_efg_step.npy')
win = np.load(str+'bouns\\deepsynth\\deepsynth_efg_reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0], win_mean, alpha=1, c='m', label='DeepSynth')
plt.fill_between(x=step[0], y1=win_min, y2=win_max, alpha=0.2, color='m')

step = np.load(str+'bouns\\jirp\\jirp_bonus_step.npy')
win = np.load(str+'bouns\\jirp\\jirp_bonus_reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0], win_mean, alpha=1, c='r', label='JIRP')
plt.fill_between(x=step[0], y1=win_min, y2=win_max, alpha=0.2, color='r')

step = np.load(str+'bouns\\hrl\\step.npy')
win = np.load(str+'bouns\\hrl\\reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0], win_mean, alpha=1, c='y', label='HRL')
plt.fill_between(x=step[0], y1=win_min, y2=win_max, alpha=0.2, color='y')

step = np.load(str+'bouns\\qlearning\\step.npy')
win = np.load(str+'bouns\\qlearning\\reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0], win_mean, alpha=1, c='g', label='Q-Learning')
plt.fill_between(x=step[0], y1=win_min, y2=win_max, alpha=0.2, color='g')

plt.xlabel('Steps',font)
plt.ylabel('Reward',font)
plt.grid(color = 'black', linestyle = '-.', linewidth = 0.5)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()



step = np.load(str+'c4\\ours_c0\\step.npy')
# episode = np.load('..\data\qlearn\qlearn_fg_episode.npy')
win = np.load(str+'c4\\ours_c0\\reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0], win_mean, alpha=1, c='b', label='ours')
plt.fill_between(x=step[0], y1=win_min, y2=win_max, alpha=0.2, color='b')

step = np.load(str+'c4\\c4\\jirp_afcbh_step.npy')
win = np.load(str+'c4\\c4\\jirp_afcbh_reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0], win_mean, alpha=1, c='r', label='JIRP')
plt.fill_between(x=step[0], y1=win_min, y2=win_max, alpha=0.2, color='r')

step = np.load(str+'c4\\deepsynth\\deepsynth_efg_step.npy')
win = np.load(str+'c4\\deepsynth\\deepsynth_efg_reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0], win_mean, alpha=1, c='m', label='DeepSynth')
plt.fill_between(x=step[0], y1=win_min, y2=win_max, alpha=0.2, color='m')

step = np.load(str+'c4\\hrl\\step.npy')
win = np.load(str+'c4\\hrl\\reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0], win_mean, alpha=1, c='y', label='HRL')
plt.fill_between(x=step[0], y1=win_min, y2=win_max, alpha=0.2, color='y')

step = np.load(str+'c4\\qlearning\\step.npy')
win = np.load(str+'c4\\qlearning\\reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0], win_mean, alpha=1, c='g', label='Q-Learning')
plt.fill_between(x=step[0], y1=win_min, y2=win_max, alpha=0.2, color='g')

plt.xlabel('Steps',font)
plt.ylabel('Reward',font)
plt.grid(color = 'black', linestyle = '-.', linewidth = 0.5)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()



step = np.load(str+'efg\\ours_C0\\step.npy')
# episode = np.load('..\data\qlearn\qlearn_fg_episode.npy')
win = np.load(str+'efg\\ours_C0\\reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0], win_mean, alpha=1, c='b', label='ours')
plt.fill_between(x=step[0], y1=win_min, y2=win_max, alpha=0.2, color='b')

step = np.load(str+'efg\\jirp\\jirp_efg_step.npy')
win = np.load(str+'efg\\jirp\\jirp_efg_reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0], win_mean, alpha=1, c='r', label='JIRP')
plt.fill_between(x=step[0], y1=win_min, y2=win_max, alpha=0.2, color='r')

step = np.load(str+'efg\\deepsynth\\deepsynth_efg_step.npy')
win = np.load(str+'efg\\deepsynth\\deepsynth_efg_reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0], win_mean, alpha=1, c='m', label='DeepSynth')
plt.fill_between(x=step[0], y1=win_min, y2=win_max, alpha=0.2, color='m')

step = np.load(str+'efg\\hrl\\step.npy')
win = np.load(str+'efg\\hrl\\reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0], win_mean, alpha=1, c='y', label='HRL')
plt.fill_between(x=step[0], y1=win_min, y2=win_max, alpha=0.2, color='y')

step = np.load(str+'efg\\ql\\step.npy')
win = np.load(str+'efg\\ql\\reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0], win_mean, alpha=1, c='g', label='Q-Learning')
plt.fill_between(x=step[0], y1=win_min, y2=win_max, alpha=0.2, color='g')

plt.xlabel('Steps',font)
plt.ylabel('Reward',font)
plt.grid(color = 'black', linestyle = '-.', linewidth = 0.5)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()


step = np.load(str+'fg\\ours_C0\\step.npy')
# episode = np.load('..\data\qlearn\qlearn_fg_episode.npy')
win = np.load(str+'fg\\ours_C0\\reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0], win_mean, alpha=1, c='b', label='ours')
plt.fill_between(x=step[0], y1=win_min, y2=win_max, alpha=0.2, color='b')

step = np.load(str+'fg\\jirp\\jirp_fg_step.npy')
win = np.load(str+'fg\\jirp\\jirp_fg_reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0], win_mean, alpha=1, c='r', label='JIRP')
plt.fill_between(x=step[0], y1=win_min, y2=win_max, alpha=0.2, color='r')

step = np.load(str+'fg\\deepsynth\\deepsynth_fg_step.npy')
win = np.load(str+'fg\\deepsynth\\deepsynth_fg_reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0], win_mean, alpha=1, c='m', label='DeepSynth')
plt.fill_between(x=step[0], y1=win_min, y2=win_max, alpha=0.2, color='m')

step = np.load(str+'fg\\hrl\\step.npy')
win = np.load(str+'fg\\hrl\\reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0], win_mean, alpha=1, c='y', label='HRL')
plt.fill_between(x=step[0], y1=win_min, y2=win_max, alpha=0.2, color='y')

step = np.load(str+'fg\\ql\\step.npy')
win = np.load(str+'fg\\ql\\reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0], win_mean, alpha=1, c='g', label='Q-Learning')
plt.fill_between(x=step[0], y1=win_min, y2=win_max, alpha=0.2, color='g')

plt.xlabel('Steps',font)
plt.ylabel('Reward',font)
plt.grid(color = 'black', linestyle = '-.', linewidth = 0.5)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()



