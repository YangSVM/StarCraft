import copy
import numpy as np

'''
复杂matrix game
有三个task，两个agent
任务0和1完成之后才能做任务2
任务0任务量为4，任务1任务量为3，任务2任务量为1
合作完成任务0有4的收益，单人完成收益为1
合作完成任务1有3的收益，单人完成收益为1
任务2只要完成就有收益
任务2完成，游戏结束
'''
class matrix_game:
    env_info = {
        "n_actions":3,
        #只有action和task对应的时候才会做任务i
        "n_agents":2,
        "state_shape":3,
        "obs_shape":3,
        "episode_limit":10,
        "n_tasks":3
    }
    def __init__(self):
        self.done = 0
        self.count = 0
        self.state = [0, 0, 0]
        self.win = False

    def step(self, action, qmix=False):
        self.count += 1
        t1 = 4
        t2 = 3
        p_state = copy.deepcopy(self.state)
        state = self.state
        if state[0] == t1 and state[1] == t2:
            if action[0] == 2 or action[1] == 2: 
                state[2] = 1
                self.done = 1
                self.win = True
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
        if qmix: 
            reward = [state[0] - p_state[0] + state[1] - p_state[1] + state[2] - p_state[2]]
        self.state = state
        if self.count == self.env_info["episode_limit"]: 
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
        
    def get_env_info(self): 
        return self.env_info 

