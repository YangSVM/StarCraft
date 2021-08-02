import copy
import numpy as np

class matrix_game:
    env_info = {
        "n_actions":3,
        #只有action和task对应的时候才会做任务i
        "n_agents":2,
        "state_shape":3,
        "obs_shape":3,
        "episode_limit":10
    }
    def __init__(self):
        self.done = 0
        self.count = 0
        self.state = [0, 0, 0]
        self.win = False

    def step(self, action):
        self.count += 1
        t1 = 4
        t2 = 3
        reward = 0                  # 这句话没用
        self.count += 1
        p_state = copy.deepcopy(self.state)
        state = self.state
        if state[0] == t1 and state[1] == t2:
            if action[0] == 2 or action[1] == 2: 
                state[2] = 1
                self.done = 1
                self.win = True
                reward = 1                  # 这句话没用
        else: 
            if action[0] == 0: 
                if action[1] == 0:
                    state[0] = t1
                else: 
                    state[0] = min(t1, state[0] + 1)
  
            elif action[0] == 1:
                if action[1] != 1: 
                    state[1] = min(t2, state[1] + 1)
            if action[1] == 1:
                if action[0] == 1:
                    state[1] = t2
                else: 
                    state[1] = min(t2, state[1] + 1)
            elif action[1] == 0: 
                if action[0] != 0: 
                    state[0] = min(t1, state[0] + 1) 

        reward = [state[0] - p_state[0], state[1] - p_state[1], state[2] - p_state[2]]
        self.state = state
        if self.count == 10: 
            self.done = True
        return reward, self.done, self.win
    
    def reset(self):
        self.done = 0
        self.count = 0
        self.state = [0, 0, 0]
        self.win = False

    def get_obs(self):
        obs = []
        obs.append(self.state)
        obs.append(self.state)
        return np.array(obs)

    def get_state(self):
        return np.array(self.state)

    def get_state1(self):
        return self.state
    
    def close(self):
        pass

    def save_replay(self):
        pass

    def get_avail_agent_actions(self, agent_id):
        return [1, 1, 1]

    def get_obs_enemy_feats_size(self):
        return[3, 3]
    
    def get_obs_enemy_feats_size(self):
        return [3]

