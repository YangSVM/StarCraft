from torch._C import qscheme
import torch.nn as nn
import torch.nn.functional as f
import torch

class TaskRNNAll(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(TaskRNNAll, self).__init__()
        self.args = args
        self.n_tasks = args.n_tasks
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # args.n_actions +1 ：动作维数+task selector. 每个任务不同的策略。
        self.fc2 = nn.Linear(args.rnn_hidden_dim, (args.n_actions +1) * args.n_tasks)

    def forward(self, obs, hidden_state):
        ''' 不传0。直接返回整个
        Params:
            obs: (n_episode,  obs_shape+n_actions+n_agents). or (n_episode*n_agent, obs_shape+n_actions+n_agents)
            hidden_state: (n_episode, n_agent, rnn_hidden_dim)
        Return:
            q: (n_episode,  n_actions), or (n_episode*n_agent,  n_actions)
            i_task: (n_episode) or (n_episode*n_agent)
        '''

        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        # 保证所有的q都是正数
        q = torch.sigmoid(q)

        '''选择对应的task
        '''
        q_shape = list(q.shape)         # shape: (n_episode, (n_actions +1) * n_tasks)
        q_shape.append(-1)
        q_shape[-2] = self.n_tasks

        # 选择对应的动作：q_shape: (n_episode, n_tasks, n_actions+1)
        q = q.view(q_shape)

        return q, h



class TaskRNNAllwoTask(nn.Module):
    # 不传0。选择累加动作最优的一列
    def __init__(self, input_shape, args):
        super(TaskRNNAllwoTask, self).__init__()
        self.args = args
        self.n_tasks = args.n_tasks
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # args.n_actions +1 ：动作维数+task selector. 每个任务不同的策略。
        self.fc2 = nn.Linear(args.rnn_hidden_dim, (args.n_actions) * args.n_tasks)

    def forward(self, obs, hidden_state):
        ''' 不传0。选择
        Params:
            obs: (n_episode,  obs_shape+n_actions+n_agents). or (n_episode*n_agent, obs_shape+n_actions+n_agents)
            hidden_state: (n_episode, n_agent, rnn_hidden_dim)
        Return:
            q: (n_episode,  n_actions), or (n_episode*n_agent,  n_actions)
            i_task: (n_episode) or (n_episode*n_agent)
        '''

        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        # 保证所有的q都是正数
        q = torch.sigmoid(q)

        '''选择对应的task
        '''
        q_shape = list(q.shape)         # shape: (n_episode, (n_actions ) * n_tasks)
        q_shape.append(-1)
        q_shape[-2] = self.n_tasks

        # 选择对应的动作：q_shape: (n_episode, n_tasks, n_action)
        q = q.view(q_shape)


        
        return q, h