import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt


class Gridworld:
    def __init__(self):
        self.states_ = None
        self.rewards_ = {}
        self.gamma_ = 0
        self.starting_grid_ = None
        self.terminal_grid_ = None
        self.blocked_grids_ = []
        self.dim_ = None

    def set_states(self, dim):
        """
        Creates the Gridworld states
        """
        if not isinstance(dim, int):
            raise Exception("Dimensions of the Grid must be an Integer")

        elif dim < 0:
            raise Exception("Dimensions must be positive")

        else:
            self.states_ = [(a, b) for a in range(dim) for b in range(dim)]
            self.dim_ = dim

    def set_rewards(self, val):
        """
        Creates the set of rewards and stores into a dictionary based on event
        """
        if type(val) is not tuple and len(val) != 4:
            raise Exception("Rewards must be a tuple of dimension 3")

        for v in val:
            if type(v) is not int and type(v) is not float:
                raise Exception("Rewards value must be Integer or Float")
        else:
            self.rewards_['move'] = val[0]
            self.rewards_['wall'] = val[1]
            self.rewards_['blocked'] = val[2]
            self.rewards_['terminal'] = val[3]

    def set_discount(self, val):
        """
        Sets the discount factor of each step in the environment
        """
        if type(val) is not int and type(val) is not float:
            raise Exception("Discount factor must be an Integer or Float")

        elif val < 0 or val > 1:
            raise Exception("Invalid Discounting factor must be between 0 and 1")

        else:
            self.gamma_ = val

    def set_starting_grid(self, state):
        """
        Sets the starting position in the grid
        """
        if type(state) is not tuple and len(state) != 2:
            raise Exception("Starting State must be a Tuple of dimension 2")

        elif state[0] >= self.get_dim() or state[1] >= self.get_dim():
            raise Exception("Starting State must be within the Gridworld")

        elif state in self.blocked_grids_:
            raise Exception("Starting State cannot be within a blocked grid")

        elif state == self.get_terminal_grid():
            raise Exception("Starting State cannot be Terminal State")

        else:
            self.starting_grid_ = state

    def set_terminal_grid(self, state):
        """
        Sets the terminal position in the grid
        """
        if type(state) is not tuple and len(state) != 2:
            raise Exception("Terminal State must be a Tuple of dimension 2")

        elif state[0] >= self.get_dim() or state[1] >= self.get_dim():
            raise Exception("Terminal State must be within the Gridworld")

        elif state in self.blocked_grids_:
            raise Exception("Terminal State cannot be within a blocked grid")

        elif state == self.get_starting_grid():
            raise Exception("Terminal State cannot be Starting State")

        else:
            self.terminal_grid_ = state

    def set_blocked_grid(self, state):
        """
        Append a blocked grid to the gridworld
        """
        if type(state) is not tuple and len(state) != 2:
            raise Exception("Blocked Grid must be a Tuple of dimension 2")

        elif state[0] >= self.get_dim() or state[1] >= self.get_dim():
            raise Exception("Blocked Grid must be within the Gridworld")

        elif state in [self.get_starting_grid(), self.get_terminal_grid()]:
            raise Exception("Blocked Grid cannot be Starting or Terminal State")

        elif state in self.blocked_grids_:
            raise Exception("State is already in Blocked Grids")

        else:
            self.blocked_grids_.append(state)

    def get_rewards(self):
        """
        Returns list of reward values
        """
        return list(set([x for x in self.rewards_.values()]))

    def get_states(self):
        """
        Returns list of all the states
        """
        return self.states_

    #     def get_states_index(self, state):
    #         return self.states_.index(state)

    def get_discount(self):
        """
        Return discount value
        """
        return self.gamma_

    def get_starting_grid(self):
        """
        Return Starting Point of Grid
        """
        return self.starting_grid_

    def get_terminal_grid(self):
        """
        Return Terminal Point of Grid
        """
        return self.terminal_grid_

    def get_blocked_grids(self):
        """
        Return list of blocked grids
        """
        return self.blocked_grids_

    def get_dim(self):
        """
        Return Dimension of the Gridworld
        """
        return self.dim_


class Agent(Gridworld):
    def __init__(self):
        super().__init__()
        self.actions_ = ['N', 'W', 'S', 'E']
        self.v_ = None
        self.policy_ = None

    def init_v_and_policy(self):
        """
        Initialize the matrix of policy and value function from Gridworld dimensions
        """
        self.set_v(np.zeros(int(self.get_dim() ** 2)))
        self.set_policy(np.array(['X' for _ in range(int(self.get_dim() ** 2))]))

    def set_v(self, vector):
        """
        Modify Value for each States
        """
        self.v_ = vector

    def set_policy(self, vector):
        """
        Modify Policy for each States
        """
        self.policy_ = vector

    def get_actions(self):
        """
        Returns list of Actions
        """
        return self.actions_

    def get_v(self):
        """
        Returns Value for each States
        """
        return self.v_

    def get_policy(self):
        """
        Returns Policy for each States
        """
        return self.policy_

    def pi(self, a, s):
        """
        Initial Uniformly Distributed Policy
        """
        if a == 'N':
            return 0.25
        elif a == 'S':
            return 0.25
        elif a == 'W':
            return 0.25
        elif a == 'E':
            return 0.25
        else:
            raise Exception("Invalid Action")

    def move_conditions(self, s, a):
        """
        Returns the next state based on action and booleans if hit side wall or move
        """
        if a == 'N':
            temp_s = (s[0] - 1, s[1])
            wall_cond = s[0] == 0
            move_cond = s[0] > 0

        elif a == 'S':
            temp_s = (s[0] + 1, s[1])
            wall_cond = s[0] == self.get_dim() - 1
            move_cond = s[0] < self.get_dim() - 1

        elif a == 'W':
            temp_s = (s[0], s[1] - 1)
            wall_cond = s[1] == 0
            move_cond = s[1] > 0

        elif a == 'E':
            temp_s = (s[0], s[1] + 1)
            wall_cond = s[1] == self.get_dim() - 1
            move_cond = s[1] < self.get_dim() - 1

        else:
            raise Exception("Invalid Action")

        return temp_s, wall_cond, move_cond

    def prob(self, s_prime, r, s, a):
        """
        Returns probability of transitioning to state s_prime with reward r given action a in state s
        """
        # If we are in terminal state or inside a terminal grid we stop moving
        if s in self.get_blocked_grids() + [self.get_terminal_grid()]:
            return 0.0

        temp_s, wall_cond, move_cond = self.move_conditions(s, a)

        # Check if the potential moves hit a blocked grid
        if temp_s in self.get_blocked_grids():
            if s_prime == s and r == self.rewards_['blocked']:
                return 1.0
            else:
                return 0.0

        # Check if we hit wall
        if wall_cond and s_prime == s and r == self.rewards_['wall']:
            return 1.0

        # Moves in the direction of action
        elif move_cond and s_prime == temp_s and r == self.rewards_['move']:
            return 1.0
        else:
            return 0.0

    def BellmanOperator(self):
        """
        Bellman Operator to compute the new value function for each states
        """
        states, actions, rewards = self.get_states(), self.get_actions(), self.get_rewards()
        gamma = self.get_discount()

        vector = np.array(
            [np.sum([self.pi(a, s) * self.prob(s_prime, r, s, a) * (r + gamma * self.get_v()[states.index(s_prime)]) \
                     for s_prime in states for a in actions for r in rewards]) for s in states])
        vector[states.index(self.get_terminal_grid())] = self.rewards_['terminal']
        # scale factor for plot
        scale_min_factor = 1.1
        # We fill the values of blocked cell with a low value to plot it black later on
        min_val = -abs(np.min(vector)) * scale_min_factor
        for s in self.get_blocked_grids():
            vector[states.index(s)] = min_val

        self.set_v(vector)

    def compute_value_policy(self, thresh):
        """
        Compute the value of the initial policy given a threshold to stop iterating
        """
        prev_v = self.get_v().copy()
        eps = 1
        states = self.get_states()
        while eps > thresh:
            self.BellmanOperator()
            #  temp_v = self.get_v().copy()
            eps = (np.array([(self.get_v()[states.index(s)] - prev_v[states.index(s)]) ** 2 for s in states])).sum()
            prev_v = self.get_v().copy()

    def plot_value_function(self):
        """
        Print the Value for each State
        """
        val = self.get_v().copy().reshape((self.get_dim(), self.get_dim()))
        self.plot_state_matrix(30, val)

    def plot_state_matrix(self, fs, matrix, add_data=None):
        """
        Plot Colored State Matrix
        """
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111)
        if add_data is None:
            data = matrix
        else:
            data = add_data
        fs_ = fs
        for (i, j), z in np.ndenumerate(data):

            if (i, j) in self.get_blocked_grids():
                f = 'X'
                fs = 40
            elif isinstance(z, np.str_):
                f = str(z)
            else:
                f = '{:0.1f}'.format(z)

            ax.text(j + 0.5, i + 0.5, f, ha='center', va='center', fontsize=fs,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
            fs = fs_

        extent = (0, data.shape[1], data.shape[0], 0)
        ax.grid(color='k', lw=2)
        ax.imshow(matrix, extent=extent, cmap='magma')

    def state_action_val(self, s, a):
        """
        Computes the Q-value given a state-action pair
        """
        states, rewards = self.get_states(), self.get_rewards()

        return np.sum(
            [self.prob(s_prime, r, s, a) * (r + self.get_v()[states.index(s_prime)]) for s_prime in states for r in
             rewards])

    def compute_optimized_policy(self, thresh):
        """
        Computes the optimal policy for each state given a threshold to stop iterating
        """
        eps = 1
        states, actions = self.get_states(), self.get_actions()

        while eps > thresh:
            # scale factor for plot
            scale_min_factor = 1.25
            new_v = np.array([np.max([self.state_action_val(s, a) for a in actions]) for s in states])
            new_v[states.index(self.get_terminal_grid())] = self.rewards_['terminal']
            # We fill the values of blocked cell with a low value to plot it black later on
            min_val = -abs(np.min(new_v)) * scale_min_factor
            for s in self.get_blocked_grids():
                new_v[states.index(s)] = min_val

            new_policy = np.array([actions[np.argmax([self.state_action_val(s, a) for a in actions])] for s in states])
            eps = np.sum((self.get_v() - new_v) ** 2)
            self.set_v(new_v)

        for i in range(len(new_policy)):
            if new_policy[i] == 'W':
                new_policy[i] = u'\u2190'
            elif new_policy[i] == 'N':
                new_policy[i] = u'\u2191'
            elif new_policy[i] == 'E':
                new_policy[i] = u'\u2192'
            elif new_policy[i] == 'S':
                new_policy[i] = u'\u2193'

        for s in self.get_blocked_grids():
            new_policy[states.index(s)] = 'X'
        new_policy[states.index(self.get_terminal_grid())] = 'T'
        self.set_policy(new_policy)

    def plot_optimized_policy(self):
        """
        Print the Optimized Policy for each State
        """
        policy = self.get_policy().copy().reshape((self.get_dim(), self.get_dim()))
        v = self.get_v().copy().reshape((self.get_dim(), self.get_dim()))
        self.plot_state_matrix(40, v, policy)

    def print_optimaL_set_directions(self):
        """
        Print the Optimal Policy starting at the Starting Grid
        """
        directions = []
        curr_grid = self.get_starting_grid()
        while curr_grid != self.get_terminal_grid():
            index = self.get_states().index(curr_grid)
            action = self.get_policy()[index]
            directions.append(action)
            if action == u'\u2191':
                curr_grid = (curr_grid[0] - 1, curr_grid[1])
            elif action == u'\u2190':
                curr_grid = (curr_grid[0], curr_grid[1] - 1)
            elif action == u'\u2193':
                curr_grid = (curr_grid[0] + 1, curr_grid[1])
            elif action == u'\u2192':
                curr_grid = (curr_grid[0], curr_grid[1] + 1)

        print(directions)

    def plot_optimaL_set_directions(self):
        """
        Plot Optimal Direction Matrix
        """
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111)
        data = np.empty((self.get_dim(), self.get_dim())) * np.nan
        cax = ax.matshow(data, cmap='binary')
        curr_grid = self.get_starting_grid()

        while curr_grid != self.get_terminal_grid():
            index = self.get_states().index(curr_grid)
            action = self.get_policy()[index]
            ax.text(curr_grid[1] + 0.5, curr_grid[0] + 0.5, action, color='red', ha='center', va='center', fontsize=40,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='r'))

            if action == u'\u2191':
                curr_grid = (curr_grid[0] - 1, curr_grid[1])
            elif action == u'\u2190':
                curr_grid = (curr_grid[0], curr_grid[1] - 1)
            elif action == u'\u2193':
                curr_grid = (curr_grid[0] + 1, curr_grid[1])
            elif action == u'\u2192':
                curr_grid = (curr_grid[0], curr_grid[1] + 1)

        for s in self.get_blocked_grids():
            ax.text(s[1] + 0.5, s[0] + 0.5, 'X', ha='center', va='center', fontsize=40,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='k'))
        ax.text(self.get_terminal_grid()[1] + 0.5, self.get_terminal_grid()[0] + 0.5, 'T', color='blue', ha='center',
                va='center', fontsize=40,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='blue'))
        extent = (0, data.shape[1], data.shape[0], 0)
        ax.grid(color='k', lw=2)
        ax.imshow(data, extent=extent)
        plt.show()
