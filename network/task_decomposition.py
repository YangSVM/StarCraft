from numpy.core.numeric import zeros_like
from torch.functional import istft
import torch.nn as nn
import torch
import torch.nn.functional as F


class TaskDecomposition(nn.Module):
    def __init__(self, args):
        super(TaskDecomposition, self).__init__()
        self.args = args
        # 因为生成的hyper_w1需要是一个矩阵，而pytorch神经网络只能输出一个向量，
        # 所以就先输出长度为需要的 矩阵行*矩阵列 的向量，然后再转化成矩阵
        self.n_tasks = args.n_tasks
        self.hypers_w1, self.hypers_w2, self.hypers_b1, self.hypers_b2 = [], [], [], []
        
        for i in range(self.n_tasks):
            # args.n_agents是使用hyper_w1作为参数的网络的输入维度，args.qmix_hidden_dim是网络隐藏层参数个数
            # 从而经过hyper_w1得到(经验条数，args.n_agents * args.qmix_hidden_dim)的矩阵
            if args.two_hyper_layers:
                hyper_w1 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(args.hyper_hidden_dim, args.n_agents * args.qmix_hidden_dim))
                # 经过hyper_w2得到(经验条数, 1)的矩阵
                hyper_w2 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim))
            else:
                hyper_w1 = nn.Linear(args.state_shape, args.n_agents * args.qmix_hidden_dim)
                # 经过hyper_w2得到(经验条数, 1)的矩阵
                hyper_w2 = nn.Linear(args.state_shape, args.qmix_hidden_dim * 1)

            # hyper_w1得到的(经验条数，args.qmix_hidden_dim)矩阵需要同样维度的hyper_b1
            hyper_b1 = nn.Linear(args.state_shape, args.qmix_hidden_dim)
            # hyper_w2得到的(经验条数，1)的矩阵需要同样维度的hyper_b1
            hyper_b2 =nn.Sequential(nn.Linear(args.state_shape, args.qmix_hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(args.qmix_hidden_dim, 1)
                                        )
            self.hypers_w1.append(hyper_w1)
            self.hypers_w2.append(hyper_w2)
            self.hypers_b1.append(hyper_b1)
            self.hypers_b2.append(hyper_b2)
        self.hypers_w1, self.hypers_w2, self.hypers_b1, self.hypers_b2 = \
            nn.ModuleList(self.hypers_w1),nn.ModuleList(self.hypers_w2),nn.ModuleList(self.hypers_b1),nn.ModuleList(self.hypers_b2)


    def forward(self, q_values_all, states, i_task):  # states的shape为(episode_num, max_episode_len， state_shape)
        # i_task shape (n_episode, max_episode_len, n_agent)
        # q_values_all shape (n_episode, max_episode_len, n_agent)

        q_values_list = self.q_value_dec(q_values_all, i_task)
        hyper_networks = []
        # 需要根据i_task，输送到不同的网络 
        for i in range(self.n_tasks):
            # 第i个任务的网络
            hyper_w1, hyper_b1, hyper_w2, hyper_b2 = self.hypers_w1[i], self.hypers_b1[i], self.hypers_w2[i], self.hypers_b2[i]

            # 传入的q_values是三维的，shape为(episode_num, max_episode_len， n_agents)

            states = states.reshape(-1, self.args.state_shape)  # (episode_num * max_episode_len, state_shape)

            w1 = torch.abs(hyper_w1(states))  # (1920, 160)
            b1 = hyper_b1(states)  # (1920, 32)

            w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_dim)  # (1920, 5, 32)
            b1 = b1.view(-1, 1, self.args.qmix_hidden_dim)  # (1920, 1, 32)


            w2 = torch.abs(hyper_w2(states))  # (1920, 32)
            b2 = hyper_b2(states)  # (1920, 1)

            w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)  # (1920, 32, 1)
            b2 = b2.view(-1, 1, 1)  # (1920, 1， 1)


            hyper_networks.append([w1,b1,w2,b2])
        return hyper_networks, q_values_list

    
    def q_value_dec(self, q_values, i_tasks):
        '''将任务分解成
        '''
        q_list = []

        for i in range(self.n_tasks):
            q = torch.zeros(q_values.shape)
            if q_values.shape != i_tasks.shape:
                print('error. ')
            mask = i_tasks==i
            q[mask] = q_values[mask]
            q_list.append(q)
        return q_list