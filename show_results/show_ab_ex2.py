import matplotlib.pyplot as plt
import numpy as np
str = '..\\data\\craft_final\\'

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


step = np.load(str+'ab\\ours_c0\\step.npy')
# episode = np.load('..\data\qlearn\qlearn_fg_episode.npy')
win = np.load(str+'ab\\ours_c0\\reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0][:150], win_mean[:150], alpha=1, c='b', label='ours')
plt.fill_between(x=step[0][:150], y1=win_min[:150], y2=win_max[:150], alpha=0.2, color='b')

step = np.load(str+'ab\\jirp\\jirp_ab_step.npy')
win = np.load(str+'ab\\jirp\\jirp_ab_reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0], win_mean, alpha=1, c='r', label='JIRP')
plt.fill_between(x=step[0], y1=win_min, y2=win_max, alpha=0.2, color='r')

# step = np.load(str+'afe\\ours_c0177\\step.npy')
# # episode = np.load('..\data\qlearn\qlearn_fg_episode.npy')
# win = np.load(str+'afe\\ours_c0177\\reward.npy')
# win_mean = np.mean(win, axis=0)
# win_max = np.max(win, axis=0)
# win_min = np.min(win, axis=0)
# plt.plot(step[0], win_mean, alpha=1, c='y', label='ours0177')
# plt.fill_between(x=step[0], y1=win_min, y2=win_max, alpha=0.2, color='y')

step = np.load(str+'ab\\deepsynth\\ab_step.npy')
win = np.load(str+'ab\\deepsynth\\ab_reward.npy')
win = delete_max_min_two(win)
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0][:150], win_mean[:150], alpha=1, c='m', label='DeepSynth')
plt.fill_between(x=step[0][:150], y1=win_min[:150], y2=win_max[:150], alpha=0.2, color='m')
#
step = np.load(str+'ab\\hrl\\step.npy')
win = np.load(str+'ab\\hrl\\reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0][:150], win_mean[:150], alpha=1, c='y', label='HRL')
plt.fill_between(x=step[0][:150], y1=win_min[:150], y2=win_max[:150], alpha=0.2, color='y')
# #
step = np.load(str+'ab\\qlearning\\step.npy')
win = np.load(str+'ab\\qlearning\\reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0][:300], win_mean[:300], alpha=1, c='g', label='Q-Learning')
plt.fill_between(x=step[0][:300], y1=win_min[:300], y2=win_max[:300], alpha=0.2, color='g')

plt.xlabel('Steps', font)
plt.ylabel('Reward', font)
plt.grid(color = 'black', linestyle = '-.', linewidth = 0.5)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()


step = np.load(str+'afe\\ours_c0\\step.npy')
# episode = np.load('..\data\qlearn\qlearn_fg_episode.npy')
win = np.load(str+'afe\\ours_c0\\reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0][:150], win_mean[:150], alpha=1, c='b', label='ours')
plt.fill_between(x=step[0][:150], y1=win_min[:150], y2=win_max[:150], alpha=0.2, color='b')

step = np.load(str+'afe\\deepsynth\\ab_step.npy')
win = np.load(str+'afe\\deepsynth\\ab_reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0][:150], win_mean[:150], alpha=1, c='m', label='DeepSynth')
plt.fill_between(x=step[0][:150], y1=win_min[:150], y2=win_max[:150], alpha=0.2, color='m')

step = np.load(str+'afe\\jirp\\jirp_afe_step.npy')
win = np.load(str+'afe\\jirp\\jirp_afe_reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0], win_mean, alpha=1, c='r', label='JIRP')
plt.fill_between(x=step[0], y1=win_min, y2=win_max, alpha=0.2, color='r')
#
step = np.load(str+'afe\\hrl\\step.npy')
win = np.load(str+'afe\\hrl\\reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0][:150], win_mean[:150], alpha=1, c='y', label='HRL')
plt.fill_between(x=step[0][:150], y1=win_min[:150], y2=win_max[:150], alpha=0.2, color='y')
#
step = np.load(str+'afe\\qlearning\\step.npy')
win = np.load(str+'afe\\qlearning\\reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0][:300], win_mean[:300], alpha=1, c='g', label='Q-Learning')
plt.fill_between(x=step[0][:300], y1=win_min[:300], y2=win_max[:300], alpha=0.2, color='g')



plt.xlabel('Steps', font)
plt.ylabel('Reward', font)
plt.grid(color = 'black', linestyle = '-.', linewidth = 0.5)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()


step = np.load(str+'abdc\\ours_c0\\step.npy')
# episode = np.load('..\data\qlearn\qlearn_fg_episode.npy')
win = np.load(str+'abdc\\ours_c0\\reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0][:150], win_mean[:150], alpha=1, c='b', label='ours')
plt.fill_between(x=step[0][:150], y1=win_min[:150], y2=win_max[:150], alpha=0.2, color='b')

step = np.load('..\\data\\er\\deepsynth\\ab\\ab_step.npy')
win = np.load('..\\data\\er\\deepsynth\\ab\\ab_reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0][:150], win_mean[:150], alpha=1, c='m', label='DeepSynth')
plt.fill_between(x=step[0][:150], y1=win_min[:150], y2=win_max[:150], alpha=0.2, color='m')

step = np.load(str+'abdc\\jirp\\jirp_abdc_step.npy')
win = np.load(str+'abdc\\jirp\\jirp_abdc_reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0], win_mean, alpha=1, c='r', label='JIRP')
plt.fill_between(x=step[0], y1=win_min, y2=win_max, alpha=0.2, color='r')
#
step = np.load(str+'abdc\\hrl\\step.npy')
win = np.load(str+'abdc\\hrl\\reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0][:150], win_mean[:150], alpha=1, c='y', label='HRL')
plt.fill_between(x=step[0][:150], y1=win_min[:150], y2=win_max[:150], alpha=0.2, color='y')
#
step = np.load(str+'abdc\\qlearning\\step.npy')
win = np.load(str+'abdc\\qlearning\\reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0][:300], win_mean[:300], alpha=1, c='g', label='Q-Learning')
plt.fill_between(x=step[0][:300], y1=win_min[:300], y2=win_max[:300], alpha=0.2, color='g')



plt.xlabel('Steps', font)
plt.ylabel('Reward', font)
plt.grid(color = 'black', linestyle = '-.', linewidth = 0.5)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()


step = np.load(str+'afcbh\\ours_c0\\step.npy')
# episode = np.load('..\data\qlearn\qlearn_fg_episode.npy')
win = np.load(str+'afcbh\\ours_c0\\reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0][:150], win_mean[:150], alpha=1, c='b', label='ours')
plt.fill_between(x=step[0][:150], y1=win_min[:150], y2=win_max[:150], alpha=0.2, color='b')

step = np.load('..\\data\\er\\deepsynth\\afcbh\\ab_step.npy')
win = np.load('..\\data\\er\\deepsynth\\afcbh\\ab_reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0][:150], win_mean[:150], alpha=1, c='m', label='DeepSynth')
plt.fill_between(x=step[0][:150], y1=win_min[:150], y2=win_max[:150], alpha=0.2, color='m')

step = np.load(str+'afcbh\\jirp\\jirp_afcbh_step.npy')
win = np.load(str+'afcbh\\jirp\\jirp_afcbh_reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0], win_mean, alpha=1, c='r', label='JIRP')
plt.fill_between(x=step[0], y1=win_min, y2=win_max, alpha=0.2, color='r')
#
step = np.load(str+'afcbh\\hrl\\step.npy')
win = np.load(str+'afcbh\\hrl\\reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0][:150], win_mean[:150], alpha=1, c='y', label='HRL')
plt.fill_between(x=step[0][:150], y1=win_min[:150], y2=win_max[:150], alpha=0.2, color='y')
#
step = np.load(str+'afcbh\\qlearning\\step.npy')
win = np.load(str+'afcbh\\qlearning\\reward.npy')
win = delete_max_min_two(win)
win_mean = np.mean(win, axis=0)
win_max = np.max(win, axis=0)
win_min = np.min(win, axis=0)
plt.plot(step[0][:300], win_mean[:300], alpha=1, c='g', label='Q-Learning')
plt.fill_between(x=step[0][:300], y1=win_min[:300], y2=win_max[:300], alpha=0.2, color='g')



plt.xlabel('Steps', font)
plt.ylabel('Reward', font)
plt.grid(color = 'black', linestyle = '-.', linewidth = 0.5)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()
