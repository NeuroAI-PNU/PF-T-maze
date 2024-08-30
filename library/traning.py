
# load library
import copy
import numpy as np
from tqdm import tqdm

# custom library
from library.env import Simple1DMaze, Grid2DMaze
from library.agents import SFAgent, PRAgent
import library.utils as utils


def training_2D(agent_ = None, 
                episodes = 500,
          alpha_w = 0.1,
          alpha_r = 0.1,
          gamma = 0.95,
          grid_size = 5,
          goal_position = None,
          max_step_length = 100,
          obs_mode = "onehot",
          noise_level = None, 
          weight_init = "eye",
          lambda_ = None,
          explora = False, 
          epsilon_dic = None,
          goal_switch = None,
          num_success_threshold = 5):
    experiences = []
    step_lengths = []
    lifetime_R_errors = []
    sf_mat_history = []
    V_vector_estimated_history = []
    rewards = []
    adaptation_rate_history = []

    max_step_length = max_step_length

    maze = Grid2DMaze(grid_size, obs_mode=obs_mode, noise_level=noise_level)
    if agent_ == SFAgent:
        agent = agent_(maze.state_size, maze.action_size, alpha_r, alpha_w, gamma, weight_init=weight_init)
    elif agent_ == PRAgent:
        agent = agent_(maze.state_size, maze.action_size, alpha_r, alpha_w, gamma, lambda_=lambda_, weight_init=weight_init)

    goal_pos = [0, grid_size - 1]
    change_goal_pos = [0, 0]
    num_success = 0
    adaptation_rate = 0
    for episode in range(episodes):
        agent_start = [grid_size -1, (grid_size//2) ]
        
        if goal_switch is not None:
            if episode % goal_switch == 0:
                goal_pos, change_goal_pos = change_goal_pos, goal_pos
                changed_goal = True
                last_reward_pos_changed = episode
            else:
                pass
        else:
            pass

        maze.reset(agent_pos=agent_start, goal_pos=goal_pos)
        if agent_ == SFAgent:
            pass
        elif agent_ == PRAgent:
            agent.eligibility_reset()
        
        state = maze.observation
        reward_error = []

        if explora:
            #epsilon decay
            epsilon = 0.9 * (epsilon_dic ** episode) + 0.1
        else:
            epsilon = epsilon_dic
        
        # step_idx = 0 for while loop
        #while True:
        for step_idx in range(max_step_length):
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(maze.action_size)
            else:
                Qvalues = np.zeros(maze.action_size)
                for action in maze.action_set:
                    simulated_next_state, simulated_reward = maze.simulate(action)
                    Qvalues[action] = agent.Q_estimates(simulated_next_state, \
                                                        simulated_reward)
                action = utils.my_argmax(Qvalues)
            #action = 1
            reward = maze.step(action)
            state_next = maze.observation
            done = maze.done
            experiences.append([state, action, state_next, reward, done])
            if agent_ == SFAgent:
                pass
            elif agent_ == PRAgent:
                agent.input(experiences[-1])
            
            state = state_next
            if step_idx >= 0:
                agent.update_w(experiences[-1])
                delta_r_vector = agent.update_r_vector(experiences[-1])
                reward_error.append(np.mean(np.abs(delta_r_vector)))
            if maze.done:
                agent.update_w(experiences[-1])
                reward_error.append(np.mean(np.abs(delta_r_vector)))
                break
            if agent_ == SFAgent:
                pass
            elif agent_ == PRAgent:
                agent.update_eligibility()
            #step_idx += 1
        #print(agent.V_estimates([0,0,1,0,0]))
        
        if changed_goal:
            if reward == 1:
                num_success += 1
            else:
                num_success = 0

            if num_success >= num_success_threshold:
                adaptation_rate = 1
                changed_goal = False
                num_success = 0
            if (episode - last_reward_pos_changed) >= (goal_switch - num_success_threshold):
                #adaptation_rate = 0
                changed_goal = False
                num_success = 0

        step_lengths.append(step_idx+1)
        lifetime_R_errors.append(np.mean(reward_error))
        V_vector_estimated_history.append(agent.V_vector_estimated)
        sf_mat_history.append(copy.deepcopy(agent.estimated_SR))
        rewards.append(experiences[-1][3])
        adaptation_rate_history.append(adaptation_rate)
        adaptation_rate = 0
    return step_lengths, sf_mat_history, V_vector_estimated_history, rewards, adaptation_rate_history


def SF_1D(episodes = 500,
          alpha_w = 0.1,
          alpha_r = 0.1,
          gamma = 0.95,
          corridor_size = 5,
          max_step_length = 100,
          obs_mode = "onehot",
          noise_level = None, 
          weight_init = "eye",
          explora = False, 
          epsilon_dic = None):
    experiences = []
    step_lengths = []
    lifetime_R_errors = []
    sf_mat_history = []
    V_vector_estimated_history = []
    V_error_history = []
    rewards = []

    max_step_length = max_step_length
    V_ground_truth = [gamma ** (corridor_size - (i+1)) for i in range(corridor_size)]

    maze = Simple1DMaze(corridor_size, obs_mode=obs_mode, noise_level=noise_level)
    agent = SFAgent(maze.corridor_size, maze.action_size, alpha_r, alpha_w, gamma, weight_init=weight_init)
    for episode in tqdm(range(episodes), desc="episodes"):
        agent_start = [0]
        goal_pos = [maze.corridor_size - 1]
        
        maze.reset(agent_pos=agent_start, goal_pos=goal_pos)
        state = maze.observation
        reward_error = []

        if explora:
            #epsilon decay
            epsilon = 0.9 * (epsilon_dic ** episode) + 0.1
        else:
            epsilon = epsilon_dic
        
        # step_idx = 0 for while loop
        #while True:
        for step_idx in range(max_step_length):
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(maze.action_size)
            else:
                Qvalues = np.zeros(maze.action_size)
                for action in maze.action_set:
                    simulated_next_state, simulated_reward = maze.simulate(action)
                    Qvalues[action] = agent.Q_estimates(simulated_next_state, \
                                                        simulated_reward)
                action = utils.my_argmax(Qvalues)
            #action = 1
            reward = maze.step(action)
            state_next = maze.observation
            done = maze.done
            experiences.append([state, action, state_next, reward, done])
            state = state_next
            if step_idx >= 0:
                agent.update_w(experiences[-1])
                delta_r_vector = agent.update_r_vector(experiences[-1])
                reward_error.append(np.mean(np.abs(delta_r_vector)))
            if maze.done:
                agent.update_w(experiences[-1])
                reward_error.append(np.mean(np.abs(delta_r_vector)))
                break
            #step_idx += 1
        #print(agent.V_estimates([0,0,1,0,0]))
        
        step_lengths.append(step_idx+1)
        lifetime_R_errors.append(np.mean(reward_error))
        V_vector_estimated_history.append(agent.V_vector_estimated)
        V_error_history.append(utils.V_error_calculation(V_ground_truth,
                                             agent.V_vector_estimated))
        sf_mat_history.append(copy.deepcopy(agent.estimated_SR))
        rewards.append(experiences[-1][3])

    return step_lengths, sf_mat_history, V_vector_estimated_history, \
        V_error_history, rewards


def PF_1D(episodes = 500,
          alpha_w = 0.1,
          alpha_r = 0.1,
          gamma = 0.95,
          lambda_ = 0.8,
          corridor_size = 5,
          max_step_length = 100,
          obs_mode = "onehot",
          noise_level = None,
          weight_init = "eye",
          explora = False, 
          epsilon_dic = None):
    experiences = []
    step_lengths = []
    lifetime_R_errors = []
    pf_mat_history = []
    V_vector_estimated_history = []
    V_error_history = []
    eligibility_history = []
    rewards = []

    max_step_length = max_step_length
    V_ground_truth = [gamma ** (corridor_size - (i+1)) for i in range(corridor_size)]

    maze = Simple1DMaze(corridor_size, obs_mode=obs_mode, noise_level=noise_level)
    agent = PRAgent(maze.corridor_size, maze.action_size, alpha_r, alpha_w, gamma, lambda_= lambda_, weight_init=weight_init)
    for episode in tqdm(range(episodes), desc="episodes"):
        agent_start = [0]
        goal_pos = [maze.corridor_size - 1]
        
        maze.reset(agent_pos=agent_start, goal_pos=goal_pos)
        agent.eligibility_reset()
        state = maze.observation
        reward_error = []

        if explora:
            #epsilon decay
            epsilon = 0.9 * (epsilon_dic ** episode) + 0.1
        else:
            epsilon = epsilon_dic
        
        # step_idx = 0 for while loop
        #while True:
        for step_idx in range(max_step_length):
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(maze.action_size)
            else:
                Qvalues = np.zeros(maze.action_size)
                for action in maze.action_set:
                    simulated_next_state, simulated_reward = maze.simulate(action)
                    Qvalues[action] = agent.Q_estimates(simulated_next_state, \
                                                        simulated_reward)
                action = utils.my_argmax(Qvalues)
            
            reward = maze.step(action)
            state_next = maze.observation
            done = maze.done
            experiences.append([state, action, state_next, reward, done])
            agent.input(experiences[-1])
            state = state_next
            if step_idx >= 0:
                agent.pf_w_update(experiences[-1])
                delta_r_vector = agent.update_r_vector(experiences[-1])
                reward_error.append(np.mean(np.abs(delta_r_vector)))
            if maze.done:
                agent.pf_w_update(experiences[-1])
                reward_error.append(np.mean(np.abs(delta_r_vector)))
                break
            eligibility = agent.update_eligibility()
            #step_idx += 1
        #print(agent.V_estimates([0,0,1,0,0]))
        
        step_lengths.append(step_idx+1)
        lifetime_R_errors.append(np.mean(reward_error))
        V_vector_estimated_history.append(agent.V_vector_estimated)
        V_error_history.append(utils.V_error_calculation(V_ground_truth,
                                             agent.V_vector_estimated))
        pf_mat_history.append(copy.deepcopy(agent.estimated_PR))
        eligibility_history.append(eligibility)
        rewards.append(experiences[-1][3])

    return step_lengths, pf_mat_history, V_vector_estimated_history, \
        V_error_history, eligibility_history, rewards