import numpy as np

c = 0.8
gamma = 0.8
m = 100000
B = np.ones(m)
B = B*gamma
K = np.arange(m)
C = B**K

class Arm:
    def __init__(self, iden):
        self.iden = iden # arm identifier 
        self.p = 0 # the actual win rate
        self.p_estimate = 0. # the estimated win rate
        self.N = 0. # the number of times the arm was pulled
        self.n = 0. # the discounted number of times the arm was pulled
        self.At = np.zeros(m) # the times when the arm was pulled
        self.rewards = np.zeros(m)

    def pull(self):
        # draw a 1 with probability p
        if np.random.random() < self.p:
            return 1
        else:
            return 0

    def update(self, x, curr_t):
        self.N += 1
        self.At[curr_t - 1] = 1
        self.rewards[curr_t - 1] = x
        temp_array = C*self.At
        sum_array = np.sum(temp_array)
        self.n = sum_array
        self.p_estimate = np.sum(temp_array*self.rewards)/sum_array

def set_p(bandits, SUs, Nk_U_k):
    for su in SUs:
        gf = np.random.random(Nk_U_k)
        for i in range(Nk_U_k):
            bandits[su][i].p = gf[i]
        
def value(mean, n, nj):
    return mean + c*np.sqrt(np.log(n) / nj)

def bandit(bandit_identifiers):
    return [Arm(iden) for iden in bandit_identifiers]

def initilize_bandit(arms, p):
    rewards = {i: [] for i in range(len(arms))}
    total_plays = 0
    for j in range(len(arms)):
        arms[j].p = p[j]
        x = arms[j].pull()
        arms[j].update(x, 1)
        total_plays += arms[j].n
    
    return total_plays

def play_bandit(bandits, curr_t, su, total_plays):
    i = np.argmax([value(a.p_estimate, total_plays[su], a.n) for a in bandits[su]])
    x = bandits[su][i].pull()
    bandits[su][i].update(x, curr_t)
    total_plays[su] += bandits[su][i].n
    return x