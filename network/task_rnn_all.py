from torch._C import qscheme
import torch.nn as nn
import torch.nn.functional as f
import torch

class TaskRNNAll(nn.Module):
    # 直接输出rnn表格。包括task score。
    def __init__(self, input_shape, args):
        # input_shape: (obs_shape + n_actions[last action] + n_agent).
        super(TaskRNNAll, self).__init__()
        self.args = args
        self.n_tasks = args.n_tasks
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # args.n_actions +1 ：动作维数+task selector. 每个任务不同的策略。
        self.fc2 = nn.Linear(args.rnn_hidden_dim, (args.n_actions +1) * args.n_tasks)

    def forward(self, obs, hidden_state):
        ''' 不传0。直接返回整个rnn表格。包括task score。
        Params:
            obs: (n_episode,  obs_shape+n_actions+n_agents). or (n_episode*n_agent, obs_shape+n_actions+n_agents)
            hidden_state: (n_episode, n_agent, rnn_hidden_dim)
        Return:
            q: (n_episode,  n_tasks, n_actions+1), or (n_episode*n_agent,  n_tasks, n_actions+1)
        '''

        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        # 保证所有的q都是正数
        q = torch.sigmoid(q)

        '''将q值reshape成合适的矩阵进行输出。
        '''
        q_shape = list(q.shape)         # shape: (n_episode, (n_actions +1) * n_tasks)
        q_shape.append(-1)
        q_shape[-2] = self.n_tasks

        # q_shape: (n_episode, n_tasks, n_actions+1)
        q = q.view(q_shape)

        return q, h



class TaskRNNAllwoTask(nn.Module):
    # 直接输出rnn表格。没有task score。
    def __init__(self, input_shape, args):
        super(TaskRNNAllwoTask, self).__init__()
        self.args = args
        self.n_tasks = args.n_tasks
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # args.n_actions：动作维数。没有task score。
        self.fc2 = nn.Linear(args.rnn_hidden_dim, (args.n_actions + 1) * args.n_tasks)

    def forward(self, obs, hidden_state):
        ''' 不传0。输出整个task表格。没有task score。
        Params:
            obs: (n_episode,  obs_shape+n_actions+n_agents). or (n_episode*n_agent, obs_shape+n_actions+n_agents)
            hidden_state: (n_episode, n_agent, rnn_hidden_dim)
        Return:
            q: (n_episode,  n_actions), or (n_episode*n_agent,  n_actions)
        '''

        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        # 保证所有的q都是正数
        q = torch.sigmoid(q)

        '''将q值reshape成合适的矩阵进行输出。
        '''
        q_shape = list(q.shape)         # shape: (n_episode, (n_actions +1) * n_tasks)
        q_shape.append(-1)
        q_shape[-2] = self.n_tasks

        # q_shape: (n_episode, n_tasks, n_action+1)
        q = q.view(q_shape)
        q = q[..., 0].unsqueeze(-1) * q[..., 1:]

        return q, h