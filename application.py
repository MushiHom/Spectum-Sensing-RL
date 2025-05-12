%reset -f
from constructor_function_ref import *
from constructor_function_prop import *
import matplotlib.pyplot as plt

# System parameters
tau = 1
c = 0.8
M = 5  # Maximum number of attempts before a call is considered blocked
K = 10 # Number of SUs
L = 10 # Number of primary channels
Nk = 4 # Number of neighbors
rate_exp = 4
rate_su = 2
epsilon = 0.3
alpha = 0.5
pui = [4, 5, 6, 7, 8, 9]
time_duration = 100000
time = range(2,time_duration + 1)

PU_avg_att_list = []
PU_det_prob_list = []
PU_blk_rate_list = []
Q_sum_su_dict = {}
Q_sum_ch_dict = {}
PU_avg_att_list_dur = []
PU_det_prob_list_dur = []
PU_blk_rate_list_dur = []
Q_sum_su_dict_dur = {}
Q_sum_ch_dict_dur = {}

PU_avg_att_list, PU_det_prob_list, PU_blk_rate_list, Q_sum_su_dict, Q_sum_ch_dict = normal_dur_fun(alpha, tau, c, M, K, L, Nk, rate_exp, rate_su, epsilon, pui, time_duration)
PU_avg_att_list_dur, PU_det_prob_list_dur, PU_blk_rate_list_dur, Q_sum_su_dict_dur, Q_sum_ch_dict_dur = updated_dur_fun(alpha, tau, c, M, K, L, Nk, rate_exp, 1, epsilon, pui, time_duration)

pui_percentage = [x*10 for x in pui]
PU_blk_rate_percentage = [x*100 for x in PU_blk_rate_list]
PU_blk_rate_percentage_dur = [x*100 for x in PU_blk_rate_list_dur]

f = plt.figure()
f.set_figwidth(8)
f.set_figheight(6)
plt.plot(pui_percentage,PU_avg_att_list, label= "Ref")
plt.plot(pui_percentage,PU_avg_att_list_dur, label= "Proposed")
plt.legend(loc="upper left")
plt.xlabel("PU usage rate in %")
plt.ylabel("Avg. number of attempts")
plt.grid()
plt.show()

f = plt.figure()
f.set_figwidth(8)
f.set_figheight(6)
plt.plot(pui_percentage,PU_det_prob_list, label= "Ref")
plt.plot(pui_percentage,PU_det_prob_list_dur, label= "Proposed")
plt.legend(loc="upper right")
plt.xlabel("PU usage rate in %")
plt.ylabel("Avg. detection probility")
plt.grid()
plt.show()

f = plt.figure()
f.set_figwidth(8)
f.set_figheight(6)
plt.plot(pui_percentage,PU_blk_rate_percentage, label= "Ref")
plt.plot(pui_percentage,PU_blk_rate_percentage_dur, label= "Proposed")
plt.legend(loc="upper left")
plt.xlabel("PU usage rate in %")
plt.ylabel("Avg. call block rate in %")
plt.grid()
plt.show()