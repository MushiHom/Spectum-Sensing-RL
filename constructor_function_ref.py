import math
from arm import bandit, initilize_bandit, play_bandit, set_p
import random
import numpy as np
from functions import *


def normal_dur_fun(alpha, tau, c, M, K, L, Nk, rate_exp, rate_su, epsilon, pui, time_duration):
    Nk_U_k = Nk + 1 # SUk and its neghbors
    channels = []
    SUs = []
    bandit_probabilities = {}
    for i in range(K):
        item_s = 'SU' + str(i+1)
        item_b = list(np.random.random(Nk_U_k))
        SUs.append(item_s)    
        bandit_probabilities[item_s] = item_b

    for i in range(L):
        item_c = 'c' + str(i+1)
        channels.append(item_c)

    # create empty lists to hold the performance metrics values and Q-values
    Q_sum_su_dict = {}
    Q_sum_ch_dict = {}
    PU_avg_att_list = []
    PU_det_prob_list = []
    PU_blk_rate_list = []



    for rate_pu in pui:
    
        # Generate neighbors for each SU as a dictionary which includes the SU as well
        neighbors = generate_neighbors(K, Nk)
        # Create bandit objects for each SU with Nk_U_k arms
        bandits = {su: bandit(range(Nk_U_k)) for su in SUs}
        # Intilize bandits by playing each arm of each bandit once at time t = 1
        total_plays = {su: initilize_bandit(bandits[su], np.random.random(Nk_U_k)) for su in SUs}

        print('* for rate_pu =', rate_pu)
        # Initilize Q-values for all channels states held by each SU
        Q = {su: {c: -10. for c in channels} for su in SUs}
        Q_old = {su: {c: 0. for c in channels} for su in SUs}
        # Initilize state values held by SUs for all channels
        s = {su: {c: {suj: 1 for suj in neighbors[su]} for c in channels} for su in SUs}
        temp_s = {su: {c: {suj: 1 for suj in neighbors[su]} for c in channels} for su in SUs}
        # Initilize aging weights and tj for all channels states held by each SU to 0
        W = {su: {c: {suj: 0 for suj in neighbors[su]} for c in channels} for su in SUs}
        tj = {su: {c: {suj: 0 for suj in neighbors[su]} for c in channels} for su in SUs}
        temp_tj = {su: {c: {suj: 0 for suj in neighbors[su]} for c in channels} for su in SUs}
        # Initilize the number of times the SU held each channel
        matep = {su: {c: 0 for c in channels} for su in SUs}
        # time vector
        time = range(2,time_duration + 1)
        # Initilize actual channels state values
        state = {c: 1 for c in channels}
        # Initilize the duration of PU calls (duration of channel holding)
        t_pu = {c: 0 for c in channels}
        # Initilize the duration of SU calls (duration of channel holding)
        t_su = {su: {c: -5 for c in channels} for su in SUs}

        avg_att_list = []
        det_prob_list = []
        det_prob_count_list = []
        blk_rate_list = []
        Q_list_su = []
        Q_list_ch = []

        for t in time:
            # Initilize number of successes and attempts for each SU
            success = {su: 0 for su in SUs}
            attempt = {su: 0 for su in SUs}
            # generate the number of calls for PUs following a poisson process
            num_calls_pu = np.random.default_rng().poisson(rate_pu)

            # a loop to assign PU calls to channels
            for c in channels:
                if num_calls_pu == 0:
                    break
                elif t_pu[c] > 0:
                    continue
                elif np.random.random() < rate_pu*0.1 and state[c] == 1:
                    num_calls_pu, state[c], t_pu[c] = update_PU_channels(num_calls_pu, state[c], t_pu[c])
                else:
                    num_calls_pu -= 1

            # change detection probabilities for neighbours every 10 TUs
            if (t-1)%10 == 0:
                set_p(bandits, SUs, Nk_U_k) 

            # generate the number of calls for SUs following a poisson process
            num_calls_su = np.random.default_rng().poisson(rate_su)
            total_calls = num_calls_su
            avg_att_sum = 0
            avg_att_count = 0
            bloc_sum = 0

            # shuffle SU list to randomize the SU call processing order
            SUh = np.random.choice(SUs, K, replace=False)
            # a loop to assign SU calls to channels
            for su in SUh:
                if sum(t_su[su].values()) != -5*K:
                    continue
                sensed_channels = []
                # Check if all channels are cyrrently held by PUs
                if sum(state.values()) == 0:
                    ALL_PU_OCCU = True
                else:
                    ALL_PU_OCCU = False
                if num_calls_su == 0:
                    break
                else:
                    num_calls_su -= 1
                    sorted_Q = dict(sorted(Q[su].items(), key = lambda item: item[1]))
                    while True:
                        attempt[su] += 1
                        sense_chan = epsilon_greedy(sorted_Q, epsilon)

                        sensed_channels.append(sense_chan)

                        R = play_bandit(bandits, t, su, total_plays)

                        # Update aging weight and Check if SUk access the channel correctly
                        tj[su][sense_chan][su] = t

                        if state[sense_chan] == 1:
                            t_su[su][sense_chan] = np.ceil(np.random.default_rng().exponential(scale = rate_su, size=None))
                            state[sense_chan] = 0
                            s[su][sense_chan][su] = 1
                            x = Q_reward(neighbors[su], bandits[su], W[su][sense_chan], tj[su][sense_chan], s[su][sense_chan], sense_chan, su, t)
                            r = 1 - x
                            success[su] += 1
                        else:
                            s[su][sense_chan][su] = 0
                            x = Q_reward(neighbors[su], bandits[su], W[su][sense_chan], tj[su][sense_chan], s[su][sense_chan], sense_chan, su, t)
                            r = -x
                        # Update Q-value
                        if Q[su][sense_chan] < -5:
                            Q_old[su][sense_chan] = 0
                        else:
                            Q_old[su][sense_chan] = Q[su][sense_chan]
                        matep[su][sense_chan] += 1
                        Q[su][sense_chan] = (1 - alpha)*Q_old[su][sense_chan] + alpha*(r - math.exp(-tau*matep[su][sense_chan]))


                        if attempt[su] == M and success[su] == 0:
                            bloc_sum += 1
                        if success[su] == 1:
                            avg_att_sum += attempt[su]
                            avg_att_count += 1
                        if success[su] == 1 or attempt[su] == M:
                            for suj in neighbors[su]:
                                for c in sensed_channels:
                                    temp_tj[suj][c][su] = t
                                    temp_s[suj][c][su] = 0
                            break

            tj = temp_tj
            s = temp_s

            # update the duration of PU and SU calls and channel states
            for c in channels:
                if t_pu[c] != 0:
                    t_pu[c] -= 1
                su_count = 0
                for su in SUs:
                    if t_su[su][c] > 0:
                        t_su[su][c] -= 1
                    elif t_su[su][c] <= 0:
                        t_su[su][c] = -5
                        su_count += 1

                if t_pu[c] == 0 and su_count == K:
                    state[c] = 1

            succe = sum(success.values())
            atte = sum(attempt.values())
            if avg_att_sum!= 0 and succe!=0:       
                avg_att_list.append(avg_att_sum/succe)
            if succe!= 0 and atte!=0:
                det_prob_count_list.append(succe/atte)


            if total_calls!= 0:
                blk_rate_list.append(bloc_sum/total_calls)

            Q_avg_su = 0
            for su in SUs:
                Q_temp_su = 0
                for c in channels:
                    if Q[su][c] != -10.:
                        Q_temp_su += Q[su][c]
                Q_avg_su += Q_temp_su/L

            Q_avg_ch = 0
            for c in channels:
                Q_temp_ch = 0
                for su in SUs:
                    if Q[su][c] != -10.:
                        Q_temp_ch += Q[su][c]
                Q_avg_ch += Q_temp_ch/K  

            Q_list_su.append(Q_avg_su)
            Q_list_ch.append(Q_avg_ch)
        Q_sum_su_dict[rate_pu] = Q_list_su
        Q_sum_ch_dict[rate_pu] = Q_list_ch

        PU_avg_att_list.append(sum(avg_att_list)/len(avg_att_list))
        PU_det_prob_list.append(sum(det_prob_count_list)/len(det_prob_count_list))
        PU_blk_rate_list.append(sum(blk_rate_list)/len(blk_rate_list))

    return PU_avg_att_list, PU_det_prob_list, PU_blk_rate_list, Q_sum_su_dict, Q_sum_ch_dict