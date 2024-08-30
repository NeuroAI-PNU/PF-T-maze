# agents.py

import numpy as np
import library.utils as utils


class SFAgent():
    def __init__(self, featvec_size, action_size, \
                alpha_r = 0.1, alpha_w = 0.1, gamma = 0.95, weight_init = "eye"):
        self.featvec_size = featvec_size
        self.action_size = action_size
        self.sf_size = featvec_size
        self.r_vector = np.zeros(self.featvec_size) # expected position of reward
        self.alpha_r = alpha_r # learning rate for reward vector W
        self.alpha_w = alpha_w # learning rate for M matrix
        self.gamma = gamma # discount rate
        if weight_init == "eye":
            self.w_matrix = np.eye(self.featvec_size)
        elif weight_init == "zero":
            self.w_matrix = np.zeros((self.featvec_size, self.sf_size))
        elif weight_init == "random":
            self.w_matrix = utils.weight_init(self.featvec_size, self.sf_size)
        else:
            print("weight initialization problem.")
        #self.algo = algo
        #if self.algo == "SARSA":
        #    self.sr_matrix = np.stack([np.eye(featvec_size) \
        #        for i in range(action_size)])
        #elif self.algo == "V":
        #    self.sr_matrix = np.eye(featvec_size)
        #    pass

    def estimated_sf_vec(self, featvec):
        # Geerts original paper used W.T here, and outer(delta_in, s_t)
        # but it was not working as expected.
        # you have to use W and outer(delta_in, s_t) - I used it here.
        # or use W.T and outer(s_t, delta_in) - which is used by original code
        # of Geerts.
        est_sf_vec = self.w_matrix @ featvec
        #est_sf_vec[est_sf_vec<0] = 0
        return est_sf_vec
    
    @property
    def estimated_SR(self):
        feature_matrix = np.eye(self.featvec_size)
        return np.matmul(self.w_matrix, feature_matrix).T
    
    def update_w(self, current_exp):
        s_t = current_exp[0]
        s_t_1 = current_exp[2]
        sf_s_t = self.estimated_sf_vec(s_t)
        #print(sf_s_t)
        sf_s_t_1 = self.estimated_sf_vec(s_t_1)
        #print(sf_s_t_1)
        done = current_exp[4]
        if done:
            delta_in = self.alpha_w * (s_t + self.gamma*s_t_1 - sf_s_t) 
            # 여기 SR code와 차이가 있엇음. gamma 값을 곱하지 않았음. 
        else:
            delta_in = self.alpha_w * (s_t + self.gamma*sf_s_t_1 - sf_s_t)
        delta_W = np.outer(delta_in, s_t)
        
        #print("w_matrix", self.w_matrix)
        self.w_matrix += delta_W
        #self.w_matrix[self.w_matrix<0] = 0
        
        #print("sf_s_t", sf_s_t)
        return delta_W

    def update_r_vector(self, current_exp):
        s_t_1 = current_exp[2]
        reward = current_exp[3]
        delta_in = self.alpha_r * (reward - np.matmul(self.r_vector, s_t_1))
        delta_r_vector = delta_in * s_t_1

        self.r_vector += delta_r_vector

        return delta_r_vector


    def V_estimates(self, featvec, goal = None):
        goal = self.r_vector
        sf_vec = self.estimated_sf_vec(featvec)
        V_state = np.matmul(sf_vec, goal)
        return V_state
        #    return V_matrix
        #if self.algo == "SARSA":
        #   return np.matmul(self.sr_matrix[action, :, :], goal) # 4 X 49 X 49 matmul 49 X 1
        #elif self.algo == "V":
        #    V_matrix = np.matmul(self.sr_matrix[:, :], goal)
        #    return V_matrix # 49 X 49 matmul 49 X 1
        #else:
        #    pass
    
    @property
    def V_vector_estimated(self):
        return np.matmul(self.estimated_SR, self.r_vector)

    def Q_estimates(self, next_state, reward):
        '''
        estimate Q value depends on estimated value of next state
        '''
        V = self.V_estimates(next_state)
        Qvalue = reward + self.gamma * V
        return Qvalue
    


class PRAgent():
    def __init__(self, featvec_size, action_size, \
                alpha_r = 0.1, alpha_w = 0.1, gamma = 0.95, lambda_ = 0.95, weight_init = "eye"):
        self.featvec_size = featvec_size
        self.action_size = action_size
        self.sf_size = featvec_size
        self.r_vector = np.zeros(self.featvec_size) # expected position of reward
        self.alpha_r = alpha_r # learning rate for reward vector W
        self.alpha_w = alpha_w # learning rate for M matrix
        self.lambda_ = lambda_ # TD lambda
        self.gamma = gamma
        self.eligibility_state = np.zeros(self.featvec_size)
        #self.next_state = np.zeros(self.featvec_size)
        if weight_init == "eye":
            self.w_matrix = np.eye(self.featvec_size)
        elif weight_init == "zero":
            self.w_matrix = np.zeros((self.featvec_size, self.sf_size))
        elif weight_init == "random":
            self.w_matrix = utils.weight_init(self.featvec_size, self.sf_size)
        else:
            print("weight initialization problem.")

    def estimated_pf_vec(self, featvec):
        est_pf_vec =  self.w_matrix @ featvec
        return est_pf_vec
    
    @property
    def estimated_SR(self):
        feature_matrix = np.eye(self.featvec_size)
        return np.matmul(self.w_matrix, feature_matrix).T

    def update_w(self, current_exp):
        current_state = current_exp[0]
        next_state = current_exp[2]
        pf_s_t = self.estimated_pf_vec(current_state)
        pf_s_t_1 = self.estimated_pf_vec(next_state)
        done = current_exp[4]
        if done:
            delta_in = current_state + self.gamma * next_state - pf_s_t
        else:
            delta_in = current_state + self.gamma * pf_s_t_1 - pf_s_t
        delta_W = self.alpha_w * np.outer(delta_in, self.eligibility_state)

        self.w_matrix += delta_W
        return delta_W

    def input(self, current_exp):
        current_state = current_exp[0]
        self.eligibility_state += current_state
        return self.eligibility_state
    
    def update_eligibility(self):
        self.eligibility_state = self.lambda_ * self.gamma * self.eligibility_state
        #print("update eligibility")
        return self.eligibility_state

    
    
    def eligibility_reset(self):
        self.eligibility_state = np.zeros(self.featvec_size)
        return self.eligibility_state
    
    def update_r_vector(self, current_exp):
        next_state = current_exp[2]
        reward = current_exp[3]
        delta_in = self.alpha_r * (reward - np.matmul(self.r_vector, next_state))
        delta_r_vector = delta_in * next_state            
        self.r_vector += delta_r_vector

        return delta_r_vector
    
    def V_estimates(self, featvec, goal = None):
        goal = self.r_vector
        sf_vec = self.estimated_pf_vec(featvec)
        V_state = np.matmul(sf_vec, goal)
        return V_state

    @property
    def V_vector_estimated(self):
        return np.matmul(self.estimated_SR, self.r_vector)
    
    def Q_estimates(self, next_state, reward):
        '''
        estimate Q value depends on estimated value of next state
        '''
        V = self.V_estimates(next_state)
        Qvalue = reward + self.gamma * V
        return Qvalue