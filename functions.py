import numpy as np
import math

K = 10 # Number of SUs
L = 10 # Number of primary channels
Nk = 4 # Number of neighbors
Nk_U_k = Nk + 1 # SUk and its neghbors
rate_exp = 4

# epsilon greedy policy to select an action
def epsilon_greedy(Qu, eps):
    if np.random.random() < (1 - eps):
        a_opt = Qu.popitem()[0]
        return a_opt
    else:
        a = np.random.choice(list(Qu.keys()))
        del Qu[a]
        return a    

# update the number and duration of calls for PUs and the channels states
def update_PU_channels(num_calls, state_c, t_c):
    state_c = 0
    t_c = np.ceil(np.random.default_rng().exponential(scale = rate_exp, size=None))
    num_calls -= 1
    return num_calls, state_c, t_c

# calculate the reward for the Q-learning algorithm
def Q_reward(neighbors, bandits, W, tj, s, sense_chan, su, t):
    temp = 0
    i = 0
    for suj in neighbors:
        p_chapeau = bandits[i].p_estimate
        x = tj[suj] - t
        W[suj] = np.exp(x)
        temp += ((1 - s[suj])*W[suj]*p_chapeau)/Nk_U_k
        i += 1
    return temp

# generate neighbour groups that guarantee unique elements and symmetry with other groups
def generate_neighbors(n, k):
    flag = False
    while flag == False:
        # Create an empty dictionary to store the groups
        groups = {i: set([i]) for i in range(1, n + 1)}
        # Generate initial groups with one member each
        for i in range(1, n + 1):
            available_numbers = list(range(1, n + 1))
            available_numbers.remove(i)
            # Choose k-1 members ensuring the symmetric property
            chosen_members = set(np.random.choice(available_numbers, k, replace=False))
            for member in chosen_members:
                if len(groups[member]) <= k and len(groups[i]) <= k:
                    groups[member].add(i)
                    groups[i].add(member)
        # Convert sets to sorted lists for consistency
        final_groups = {key: sorted(list(value)) for key, value in groups.items()}
        mini_flag = False
        for x in final_groups:
            if len(final_groups[x]) != Nk_U_k:
                mini_flag = True
                break
        if mini_flag == False:
            flag = True
        
    for x in final_groups:
        for i in range(Nk_U_k):
            final_groups[x][i] = 'SU' + str(final_groups[x][i])
    final_groups = {'SU' + str(k): v for k, v in final_groups.items()}

    return final_groups