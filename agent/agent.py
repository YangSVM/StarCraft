import numpy as np
import torch
from torch.distributions import Categorical
import copy

# Agent no communication
class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        if args.alg == 'vdn':
            from policy.vdn import VDN
            self.policy = VDN(args)
        elif args.alg == 'iql':
            from policy.iql import IQL
            self.policy = IQL(args)
        elif args.alg == 'qmix':
            from policy.qmix import QMIX
            self.policy = QMIX(args)
        elif args.alg == 'coma':
            from policy.coma import COMA
            self.policy = COMA(args)
        elif args.alg == 'qtran_alt':
            from policy.qtran_alt import QtranAlt
            self.policy = QtranAlt(args)
        elif args.alg == 'qtran_base':
            from policy.qtran_base import QtranBase
            self.policy = QtranBase(args)
        elif args.alg == 'maven':
            from policy.maven import MAVEN
            self.policy = MAVEN(args)
        elif args.alg == 'central_v':
            from policy.central_v import CentralV
            self.policy = CentralV(args)
        elif args.alg == 'reinforce':
            from policy.reinforce import Reinforce
            self.policy = Reinforce(args)
        elif   args.alg == 'task_decomposition':
            from policy.task_decomposition import TD
            self.policy = TD(args)
        elif args.alg == 'task_decomposition_all':
            from policy.task_decomposition_all import TDAll
            self.policy = TDAll(args)
        elif args.alg == 'task_decomposition_all_without_task':
            from policy.task_decomposition_all_without_task import TDAll
            self.policy = TDAll(args)
        else:
            raise Exception("No such algorithm")
        self.args = args
        if self.args.matrix_game == True: 
            self.file_handler = open('matrix game_{}.txt'.format(args.time), mode = 'w')
            # self.q_matrix = open('q_matrix.txt', mode = 'w')
            # state0情况初始化
            inputs = [0, 0, 0]
            last_action = [0., 0., 0.]
            agent_id_0 = np.zeros(self.n_agents)
            agent_id_0[0] = 1.
            agent_id_1 = np.zeros(self.n_agents)
            agent_id_1[1] = 1.
            if self.args.last_action:
                inputs = np.hstack((inputs, last_action))
            inputs_0 = copy.deepcopy(inputs)
            inputs_1 = copy.deepcopy(inputs)
            if self.args.reuse_network:
                inputs_0 = np.hstack((inputs_0, agent_id_0))
                inputs_1 = np.hstack((inputs_1, agent_id_1))
            self.inputs_0 = torch.tensor(inputs_0, dtype=torch.float32).unsqueeze(0)
            self.inputs_1 = torch.tensor(inputs_1, dtype=torch.float32).unsqueeze(0)
            obs = torch.zeros([3])
            self.obs = obs.unsqueeze(0).unsqueeze(0)
            if self.args.alg.find('task_decomposition_all') > -1:
                self.hidden = torch.zeros([1, 192])
            elif self.args.alg == 'qmix': 
                self.hidden = torch.zeros([1, 64])


    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, maven_z=None, evaluate=False):
        ''' 选择argmax的动作，考虑epsilon探索
        Params:
            obs: np array. (obs_size)
            agent_num: 0~n_agent，表示第几个agent
        '''
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        hidden_state = self.policy.eval_hidden[:, agent_num, :]     # shape: (episode_num, self.args.rnn_hidden_dim)

        # transform the shape of inputs from (42,) to (1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # get q value
        if self.args.alg == 'maven':
            maven_z = torch.tensor(maven_z, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                maven_z = maven_z.cuda()
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state, maven_z)
        elif self.args.alg == 'task_decomposition':
            q_value, self.policy.eval_hidden[:, agent_num, :], _ = self.policy.eval_rnn(inputs, hidden_state)
        elif self.args.alg ==('task_decomposition_all'):
            q_value_all, self.policy.eval_hidden[:, agent_num, :]= self.policy.eval_rnn(inputs, hidden_state)
            q_value, _ = self.policy.find_task_q(q_value_all)
        elif self.args.alg == 'task_decomposition_all_without_task':
            q_value_all, self.policy.eval_hidden[:, agent_num, :]= self.policy.eval_rnn(inputs, hidden_state)
            q_value = q_value_all.sum(dim=-2)
        else:
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)
        if self.args.matrix_game == True and evaluate: 
            if self.args.alg == 'task_decomposition_all': 
                print(obs, ': agent', agent_num, _ , q_value)
                self.file_handler.write(str(obs) + ': agent' + str(agent_num) + str(_) + str(q_value) + '\n')
                self.file_handler.flush()
            elif self.args.alg == 'qmix' or self.args.alg == 'task_decomposition_all_without_task': 
                print(obs, ':agent', agent_num, q_value)
                self.file_handler.write(str(obs) + ':agent' + str(agent_num) + str(q_value) + '\n')
                self.file_handler.flush()


        # choose action from q value
        if self.args.alg == 'coma' or self.args.alg == 'central_v' or self.args.alg == 'reinforce':
            action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon, evaluate)
        else:
            q_value[avail_actions == 0.0] = - float("inf")
            if np.random.uniform() < epsilon:
                action = np.random.choice(avail_actions_ind)  # action是一个整数
            else:
                action = torch.argmax(q_value)
        return action

    def evaluate_TDall(self): 
        # 直接在里面存到txt里面
        q_value_0, _ = self.policy.eval_rnn(self.inputs_0, self.hidden.detach())
        q_value_1, _ = self.policy.eval_rnn(self.inputs_1, self.hidden.detach())
        matri = np.empty((3, 3, 3), dtype=float)
        mat = np.empty((3, 3), dtype=float)
        for i in range(0, 3): 
            for j in range(0, 3): 
                q_val_0 = self.policy.matrix(q_value_0, i)
                q_val_1 = self.policy.matrix(q_value_1, j)

                q = torch.cat([q_val_0, q_val_1], dim = 0)
                q = q.unsqueeze(0).unsqueeze(0) # 1*1*2*3
                hyper_networks, q_values_list = self.policy.eval_task_net(q, self.obs)
                q_tasks = self.policy.calc_Qis(q_values_list, hyper_networks, is_grad4rnn=False)
                mat[i][j] = sum(q_tasks).item()
                for k in range(0, 3): 
                    matri[i][j][k] = q_tasks[k].item()
        print(str(matri))
        print(str(mat))
        self.file_handler.write(str(matri) + '\n'+str(mat) + '\n')
        self.file_handler.flush()
    
    def evaluate_qmix(self): 
        q_value_0, _ = self.policy.eval_rnn(self.inputs_0, self.hidden.detach())
        q_value_1, _ = self.policy.eval_rnn(self.inputs_1, self.hidden.detach())
        matri = np.empty((3, 3), dtype=float)
        for i in range(0, 3): 
            for j in range(0, 3): 
                q_val_0 = self.policy.matrix(q_value_0, i)
                q_val_1 = self.policy.matrix(q_value_1, j)

                q = torch.cat([q_val_0, q_val_1], dim = 0)
                q = q.unsqueeze(0).unsqueeze(0) # 1*1*2*3
                q_total_eval = self.policy.eval_qmix_net(q, self.obs)
                matri[i][j] = q_total_eval
        print(str(matri))
        self.file_handler.write(str(matri) + '\n')
        self.file_handler.flush()
    
    def evaluate_TDall_without_task(self):
        # 直接在里面存到txt里面
        q_value_0, _ = self.policy.eval_rnn(self.inputs_0, self.hidden.detach())
        q_value_1, _ = self.policy.eval_rnn(self.inputs_1, self.hidden.detach())
        matri = np.empty((3, 3), dtype=float)
        for i in range(0, 3): 
            for j in range(0, 3): 
        
                q_val_0 = self.policy.matrix(q_value_0, i)
                q_val_1 = self.policy.matrix(q_value_1, j)

                q = torch.cat([q_val_0, q_val_1], dim = 0)
                q = q.unsqueeze(0).unsqueeze(0) # 1*1*2*3
                hyper_networks, q_values_list = self.policy.eval_task_net(q, self.obs)
                q_tasks = self.policy.calc_q_total(q_values_list, hyper_networks, is_grad4rnn=False)
                q_total_eval = sum(q_tasks).item()
                matri[i][j] = q_total_eval
        print(str(matri))
        self.file_handler.write(str(matri) + '\n')
        self.file_handler.flush()



    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon, evaluate=False):
        """
        :param inputs: # q_value of all actions
        """
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # num of avail_actions
        # 先将Actor网络的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(inputs, dim=-1)
        # add noise of epsilon
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """

        if epsilon == 0 and evaluate:
            action = torch.argmax(prob)
        else:
            action = Categorical(prob).sample().long()
        return action

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training

        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)


# Agent for communication
class CommAgents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        alg = args.alg
        if alg.find('reinforce') > -1:
            from policy.reinforce import Reinforce
            self.policy = Reinforce(args)
        elif alg.find('coma') > -1:
            from policy.coma import COMA
            self.policy = COMA(args)
        elif alg.find('central_v') > -1:
            from policy.central_v import CentralV
            self.policy = CentralV(args)
        else:
            raise Exception("No such algorithm")
        self.args = args
        print('Init CommAgents')

    # 根据weights得到概率，然后再根据epsilon选动作
    def choose_action(self, weights, avail_actions, epsilon, evaluate=False):
        weights = weights.unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # 可以选择的动作的个数
        # 先将Actor网络的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(weights, dim=-1)
        # 在训练的时候给概率分布添加噪音
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """

        if epsilon == 0 and evaluate:
            # 测试时直接选最大的
            action = torch.argmax(prob)
        else:
            action = Categorical(prob).sample().long()
        return action

    def get_action_weights(self, obs, last_action):
        obs = torch.tensor(obs, dtype=torch.float32)
        last_action = torch.tensor(last_action, dtype=torch.float32)
        inputs = list()
        inputs.append(obs)
        # 给obs添加上一个动作、agent编号
        if self.args.last_action:
            inputs.append(last_action)
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents))
        inputs = torch.cat([x for x in inputs], dim=1)
        if self.args.cuda:
            inputs = inputs.cuda()
            self.policy.eval_hidden = self.policy.eval_hidden.cuda()
        weights, self.policy.eval_hidden = self.policy.eval_rnn(inputs, self.policy.eval_hidden)
        weights = weights.reshape(self.args.n_agents, self.args.n_actions)
        return weights.cpu()

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma在训练时也需要epsilon计算动作的执行概率
        # 每次学习时，各个episode的长度不一样，因此取其中最长的episode作为所有episode的长度
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)
