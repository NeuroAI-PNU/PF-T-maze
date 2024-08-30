# utils.py
import random
import numpy as np
from math import sqrt

def onehot(value, length_of_vec):
    '''
    making one hot vector, two arg needs
    value : position of a hot element
    length_of_vec : length of one hot vector
    '''
    vec = np.zeros(length_of_vec)
    vec[value] = 1
    return vec

def noise(value, length_of_vec):
    '''
    making noise vector, one arg need
    value : current position of agents
    length_of_vec : length of noise vector to agent observation
    return random noise vector of current position
    '''
    vec = random_env(length_of_vec, length_of_vec)
    return vec[value][0]

def random_env(length_x, length_y):
    '''
    make sure fixed random variable return to env
    '''
    random.seed(0)
    vec = np.zeros((length_x, length_y))
    for idx in range(length_x):
        for idy in range(length_y):
            vec[idx, idy] = random.random()
    return vec

def noisy_onehot(value, length_of_vec, noise_level = 0):
    '''
    this function is different from noise env, which return fixed random noise value
    this function is based on one-hot vector, but give noisy signal with gaussian 
    args
        value: current position of agent
        length_of_vec : size of maze
        noise_level : 0-1 value to define the noise level of signal
    '''
    onehot_vec = np.zeros(length_of_vec)
    onehot_vec[value] = 1
    noisy_vec = np.random.normal(0, noise_level, size = length_of_vec)
    vec = onehot_vec + noisy_vec
    return vec


# Xavier Weight Initilazation 
# https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
def weight_init(n_input, n_output):
    lower = -(1.0 /sqrt(n_input))
    upper = 1.0 / sqrt(n_input)
    #random.seed(42) 시드를 추가하면 모든 random agent는 똑같은 결과를 냄.
    numbers = np.random.rand(n_input * n_output)
    scaled = lower + numbers*(upper - lower)
    w_matrix = np.array(scaled).reshape([n_input, n_output])
    return np.abs(w_matrix) # Uniform distribution [0, 1/sqrt(n_i)]

def V_error_calculation(V_ground_truth, V_estimates):
    V_true = np.array(V_ground_truth)
    V_est = np.array(V_estimates)
    errors = (V_true - V_est) ** 2
    return np.mean(errors[:-1]) # exclude error from last state from 1D maze

def my_argmax(Qarray):
    max_index = np.where(Qarray == Qarray.max())[0]
    if len(max_index) == 1:
        return np.argmax(Qarray)
    else:
        return np.random.randint(len(Qarray))
