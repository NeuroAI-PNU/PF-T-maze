# env.py
import numpy as np
import library.utils as utils

class Simple1DMaze():
    def __init__(self, size, obs_mode = "onehot", noise_level = None):
        self.corridor_size = size

        # for 1D maze only 2 actions allowed
        self.action_size = 2
        self.ACTION_LT = 0
        self.ACTION_RT = 1
        self.action_set = [self.ACTION_LT, self.ACTION_RT]

        # enviroment properties
        self.goal_pos = []
        self.agent_pos = []
        self.done = None
        self.observations = None
        self.obs_mode = obs_mode
        self.noise_level = noise_level

        # gonna use feature vector only(not index for naive SR)
        self.obs_size = self.corridor_size
        self.goal_size = self.corridor_size


    def reset(self, goal_pos = None, agent_pos = None):
        self.done = False
        if goal_pos != None:
            try:
                self.goal_pos = goal_pos
            except IndexError as e:
                print(e)
        else:
            raise NotImplementedError("goal_pos")
        if agent_pos != None:
            self.agent_pos = agent_pos
        else:
            raise NotImplementedError("agent_pos")

    @property
    def corridor(self):
        '''
        return position of agent and goal in one-hot code 
        2D numpy array
        '''
        corridor = np.zeros([self.corridor_size,  2])
        corridor[self.agent_pos, 0] = 1
        corridor[self.goal_pos, 1] = 1
        return corridor

    @property
    def all_positions(self):
        '''
        return position index of goal and agent in list 
        '''
        all_positions = [self.goal_pos] + [self.agent_pos]
        return all_positions

    def step(self, action):
        '''
        arg: action
        self.ACTION_LT = 0
        self.ACTION_RT = 1
        return: reward
        '''
        
        move_array = np.array([0])
        if action == self.ACTION_LT:
            move_array = np.array([-1])
        if action == self.ACTION_RT:
            move_array = np.array([1])
        self.move_agent(move_array)
        if self.agent_pos == self.goal_pos:
            self.done = True
            return 1.0
        else:
            return 0.0

    def move_agent(self, direction):
        '''
        subroutine of self.step
        '''
        new_pos = self.agent_pos + direction
        if self.check_target(new_pos):
            self.agent_pos = list(new_pos)
    
    def check_target(self, target):
        '''
        subroutine of self.move_agent
        prevent agent from going out of maze
        '''
        check = target > -1 and target < self.corridor_size
        if check:
            return True
        else:
            return False

    def simulate(self, action):
        '''
        Q learning style simulating the action of agent
        return simulated next state and simulated reward
        '''
        agent_old_pos = self.agent_pos
        simulated_reward = self.step(action)
        simulated_next_state = self.observation
        #simulated_flag
        self.agent_pos = agent_old_pos
        return simulated_next_state, simulated_reward

    @property
    def observation(self):
        '''
        return obsevation property for agent
        state
        '''
        agent_pos_index = self.agent_pos
        if self.obs_mode == "onehot":
            return utils.onehot(agent_pos_index, self.corridor_size)
        elif self.obs_mode == "noise":
            return utils.noise(agent_pos_index, self.corridor_size)
        elif self.obs_mode == "noisy_onehot":
            return utils.noisy_onehot(agent_pos_index, self.corridor_size, noise_level=self.noise_level)
        else:
            print("check observation mode")



class Grid2DMaze():
    '''
    python Class for making various mazes
    '''
    def __init__(self, size, maze_pattern = "t_maze", obs_mode = "index", noise_level = None):
        self.grid_size = size
        self.state_size = size * size

        # for 2D maze
        self.action_size = 4
        self.ACTION_LT = 2
        self.ACTION_RT = 3
        self.ACTION_UP = 0
        self.ACTION_DW = 1
        self.action_set = [self.ACTION_UP, self.ACTION_DW, self.ACTION_LT, \
            self.ACTION_RT]

        self.blocks = self.make_blocks(maze_pattern)

        self.goal_pos = []
        self.agent_pos = []
        self.done = None
        self.observations = None
        self.noise_level = noise_level
        self.obs_mode = obs_mode

        # gonna use feature vector only(not index for naive SR)
        self.obs_size = self.state_size
        self.goal_size = self.state_size

    def reset(self, goal_pos = None, agent_pos = None):
        self.done = False
        if goal_pos != None:
            self.goal_pos = goal_pos
        else:
            self.goal_pos = self.get_free_spot()
        if agent_pos != None:
            self.agent_pos = agent_pos
        else:
            self.agent_pos = self.get_free_spot()

    def get_free_spot(self):
        free = False
        possible_x = np.arange(0, self.grid_size)
        possible_y = np.arange(0, self.grid_size)
        while not free:
            try_x = np.random.choice(possible_x, replace = False)
            try_y = np.random.choice(possible_y, replace = False)
            try_position = [try_x, try_y]
            if try_position not in self.all_positions:
                return try_position


    def make_blocks(self, pattern):
        if pattern == "t_maze":
            blocks = []
            mid = int(self.grid_size // 2)
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    if row != 0 and col != mid:
                        blocks.append([row, col])
            #self.bottlenecks = []
        elif pattern == "no":
            blocks = []
        else:
            print("Define correct maze pattern!")

        return blocks

    @property
    def grid(self):
        grid = np.zeros([self.grid_size, self.grid_size, 3])
        grid[self.agent_pos[0], self.agent_pos[1], 0] = 1
        grid[self.goal_pos[0], self.goal_pos[1], 1] = 1
        for block in self.blocks:
            grid[block[0], block[1], 2] = 1
        return grid
    
    def move_agent(self, direction):
        new_pos = self.agent_pos + direction
        if self.check_target(new_pos):
            self.agent_pos = list(new_pos)

    def check_target(self, target):
        x_check = target[0] > -1 and target[0] < self.grid_size
        y_check = target[1] > -1 and target[1] < self.grid_size
        block_check = list(target) not in self.blocks
        if x_check and y_check and block_check:
            return True
        else:
            return False

    def simulate(self, action):
        agent_old_pos = self.agent_pos
        simulated_reward = self.step(action)
        simulated_next_state = self.observation
        #simulated_flag
        self.agent_pos = agent_old_pos
        return simulated_next_state, simulated_reward

    @property
    def observation(self):
        agent_pos_index = self.agent_pos[0] * self.grid_size + self.agent_pos[1]
        if self.obs_mode == "onehot":
            return utils.onehot(agent_pos_index, self.state_size)
        elif self.obs_mode == "noisy_onehot":
            return utils.noisy_onehot(agent_pos_index, self.state_size, noise_level=self.noise_level)
        elif self.obs_mode == "noise":
            return utils.noise(agent_pos_index, self.state_size)
        else:
            print("check observation mode")
        

    @property
    def goal(self):
        goal_pos_index = self.goal_pos[0] * self.grid_size + self.goal_pos[1]
        if self.obs_mode == "onehot":
            return utils.onehot(goal_pos_index, self.state_size)
        if self.obs_mode == "index":
            return goal_pos_index

    @property
    def all_positions(self):
        all_positions = self.blocks + [self.goal_pos] + [self.agent_pos]
        return all_positions

    def state_to_grid(self, state):
        vec_state = np.zeros([self.state_size])
        vec_state[state] = 1
        vec_state = np.reshape(vec_state, [self.grid_size, self.grid_size])
        return vec_state

    def state_to_goal(self, state):
        return utils.onehot(state, self.state_size)

    def state_to_point(self, state):
        state_in_2d_grid = self.state_to_grid(state)
        state_pos = np.where(state_in_2d_grid == 1)
        xy_loc_state = [state_pos[0][0], state_pos[1][0]]
        return xy_loc_state

    def state_to_obs(self, state):
        if self.obs_mode == "onehot":
            xy_loc_state = self.state_to_point(state)
            state_index = xy_loc_state[0] * self.grid_size + xy_loc_state[1]
            return utils.onehot(state_index, self.state_size)
        if self.obs_mode == "index":
            return state

    def step(self, action):
        '''
        self.ACTION_LT = 2
        self.ACTION_RT = 3
        self.ACTION_UP = 0
        self.ACTION_DW = 1
        '''
        
        move_array = np.array([0,0])
        if action == self.ACTION_LT:
            move_array = np.array([0,-1])
        if action == self.ACTION_RT:
            move_array = np.array([0,1])
        if action == self.ACTION_UP:
            move_array = np.array([-1,0])
        if action == self.ACTION_DW:
            move_array = np.array([1,0])
        self.move_agent(move_array)
        if self.agent_pos == self.goal_pos:
            self.done = True
            return 1.0
        else:
            return 0.0