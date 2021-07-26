from torch._C import qscheme
import torch.nn as nn
import torch.nn.functional as f
import torch

class TaskRNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(TaskRNN, self).__init__()
        self.args = args
        self.n_tasks = args.n_tasks
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # args.n_actions +1 ：动作维数+task selector. 每个任务不同的策略。
        self.fc2 = nn.Linear(args.rnn_hidden_dim, (args.n_actions +1) * args.n_tasks)

    def forward(self, obs, hidden_state):
        '''
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
        q_shape = q.shape
        # 1. 取最后一个维度第一个数作为task selector, 选择最擅长的task
        task_score, i_task = q[...,0].max(dim = -1)             # i_task shape: (n_episode)
        # 2. 取对应项相乘，计算
        i_task = i_task.unsqueeze(-1).unsqueeze(-1)     # i_task shape: (n_episode, 1, 1)
        i_task_shape = list(q_shape)
        i_task_shape[-2] =  1                       
        i_task = i_task.expand(i_task_shape)                   #  i_task_shape: (n_episode, 1, n_action + 1)
        q = torch.gather(q, dim=-2, index=i_task)           # q shape (n_episode, 1, n_action + 1)。 选择q中仅与最高分任务的一行
        q = (q[...,0].unsqueeze(-1) *q[...,1:]).squeeze(-2)     # task_score * 对应动作q值。 q shape : (n_episode,  n_action)
        i_task = i_task.squeeze(1)                                          # 将 i_task task维度去掉(因为该维度值必为1)
        i_task = i_task[..., 0]                                                         # i_task shape (n_episode)

        
        return q, h, i_task

