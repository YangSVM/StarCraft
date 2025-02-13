''' 不传0。有task score。上传task score*最大值
修改成Q pi
'''
import torch
import os
from network.task_rnn_all import TaskRNNAll
from network.task_decomposition_all import TaskDecompositionAll
import torch.nn.functional as F
import time

class TDAll:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.n_tasks =  args.n_tasks
        input_shape = self.obs_shape
        # 根据参数决定RNN的输入维度
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        # 神经网络
        self.eval_rnn = TaskRNNAll(input_shape, args)  # rnn
        self.target_rnn = TaskRNNAll(input_shape, args)
        self.eval_task_net = TaskDecompositionAll(args)  # mixing
        self.target_task_net = TaskDecompositionAll(args)
        self.args = args
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_task_net.cuda()
            self.target_task_net.cuda()
        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/rnn_net_params.pkl'
                path_qmix = self.model_dir + '/td_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_task_net.load_state_dict(torch.load(path_qmix, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_rnn, path_qmix))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_task_net.load_state_dict(self.eval_task_net.state_dict())

        self.eval_parameters = list(self.eval_task_net.parameters()) + list(self.eval_rnn.parameters())
        if args.optimizer == "RMS":
            self.optimizer_rnn = torch.optim.RMSprop(self.eval_rnn.parameters(), lr=args.lr)
            self.optimizer_mix = torch.optim.RMSprop(self.eval_task_net.parameters(), lr=args.mix_lr)


        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print('Init alg Task Decomposition')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        '''
        在learn的时候，抽取到的数据是四维的，四个维度分别为 1——第几个episode 2——episode中第几个transition
        3——第几个agent的数据 4——具体obs维度。因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
        hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，然后一次给神经网络
        传入每个episode的同一个位置的transition
        '''
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
                                                             batch['r'],  batch['avail_u'], batch['avail_u_next'],\
                                                             batch['terminated']
        mask = 1 - batch["padded"].float()  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
        n_actions = avail_u.shape[-1]

        # q_evals (n_episode, max_episode_len， n_agents， n_tasks , n_actions )
        # i_task_target shape: (n_episode, max_episode_len， n_agents)
        q_evals, q_targets, i_task, i_task_target = self.get_q_values(batch, max_episode_len)

        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
        

        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为仅采取一个动作。(n_episode, episode_len, n_agent, n_tasks)
        u_shape = list(u.shape)
        u_shape.insert(3, self.n_tasks)
        u = u.unsqueeze(-2).expand(u_shape)                                                     # u shape: (n_episode, episode_len, n_agent, n_tasks, 1)
        q_evals = torch.gather(q_evals, dim=4, index=u).squeeze(4)          #q shape: (n_episode, episode_len, n_agent, n_tasks)


        # 得到target_q
        avail_u_shape = u_shape
        avail_u_shape[-1] = self.n_actions
        avail_u_next = avail_u_next.unsqueeze(-2).expand(avail_u_shape)
        q_targets[avail_u_next == 0.0] = - 9999999
        
        # 通过 i_task_target 选择对应行，找出最大价值的动作，选出该动作对应的所有任务的q值。
        i_task_target_shape = list(q_targets.shape)
        i_task_target_shape[-2] = 1         
        i_task_target  = i_task_target.unsqueeze(-1).unsqueeze(-1).expand(i_task_target_shape) # shape: (n_episode, episode_len, n_agents, 1(task), n_actions)

        q_targets_seletor = torch.gather(q_targets, dim=3, index=i_task_target) # shape: (n_episode, episode_len, n_agents, 1(task), n_actions)
        
        q_targets_seletor_shape = list(q_targets.shape)
        q_targets_seletor_shape[-1] = 1
        # q_targets_seletor shape : (n_episode, episode_len, n_agents, n_tasks, 1)
        q_targets_seletor = q_targets_seletor.argmax(dim=-1).unsqueeze(-1).expand(q_targets_seletor_shape)
        # add epsilon
        
        rand = torch.rand(q_targets_seletor_shape)
        action_rand_selector = rand<epsilon
        action_rand = torch.randint(n_actions, q_targets_seletor_shape)
        q_targets_seletor = q_targets_seletor.clone()
        q_targets_seletor[action_rand_selector] = action_rand[action_rand_selector]
        q_targets_seletor = q_targets_seletor      
        q_targets =  torch.gather(q_targets, dim=-1, index=q_targets_seletor).squeeze(-1)   # shape: (n_episode, episode_len, n_agents, n_tasks)


        hyper_networks, q_values_list = self.eval_task_net(q_evals, s)
        hyper_networks_target, q_targets_list  = self.target_task_net(q_targets, s_next)

        q_total_eval = sum(self.calc_Qis(q_values_list, hyper_networks))
        q_total_target = sum(self.calc_Qis(q_targets_list, hyper_networks_target))


        # q_total_target = self.target_qmix_net(q_targets, s_next, i_task_target)

        targets = r.sum(dim=-1).unsqueeze(-1) + self.args.gamma * q_total_target * (1 - terminated)

        td_error = (q_total_eval - targets.detach())
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss1 = (masked_td_error ** 2).sum() / mask.sum()

        q_tasks = self.calc_Qis(q_values_list, hyper_networks, is_grad4rnn=False)
        q_tasks_targets = self.calc_Qis(q_targets_list, hyper_networks_target, is_grad4rnn=False)
        q_tasks = torch.cat(q_tasks, dim=-1)
        q_tasks_targets = torch.cat(q_tasks_targets, dim=-1)
        q_tasks_targets = r + self.args.gamma * q_tasks_targets * (1 - terminated)
        
        td_task_error = (q_tasks - q_tasks_targets.detach())
        masked_td_task_error = mask * td_task_error
        loss2 = (masked_td_task_error ** 2).sum() / mask.sum()

        # loss = loss1+loss2
        self.optimizer_rnn.zero_grad()
        self.optimizer_mix.zero_grad()
        # loss1.backward(retain_graph=True)
        loss1.backward()
        # for parm in self.eval_task_net.parameters():
        #     x =parm.grad.data.cpu().numpy()
        loss2.backward()
        # loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer_rnn.step()
        self.optimizer_mix.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_task_net.load_state_dict(self.eval_task_net.state_dict())


    def calc_Qis(self, q_list, hyper_networks, is_grad4rnn=True):
        '''
        Params:
            q_list: n_task长list.代表第i任务输入的q。
        '''
        Qi_list = []
        # 把计算过程搬到这里
        for i in range(self.n_tasks):
            qi = q_list[i]
            w1, b1, w2, b2  = hyper_networks[i]

            # 传入的q_values是三维的，shape为(episode_num, max_episode_len， n_agents)
            episode_num = qi.size(0)
            qi = qi.view(-1, 1, self.args.n_agents)  # (episode_num * max_episode_len, 1, n_agents) = (1920,1,5)
            if is_grad4rnn:
                hidden = F.elu(torch.bmm(qi, w1.detach()) + b1.detach())  # (1920, 1, 32)
                Qi = torch.bmm(hidden, w2.detach()) + b2.detach()  # (1920, 1, 1)

            else:
                hidden = F.elu(torch.bmm(qi.detach(), w1) + b1)  # (1920, 1, 32)
                Qi = torch.bmm(hidden, w2) + b2  # (1920, 1, 1)

            Qi = Qi.view(episode_num, -1, 1)  # (32, 60, 1)
            Qi_list.append(Qi)
        return Qi_list


    def _get_inputs(self, batch, transition_idx):
        '''
        Return:
            inputs: (n_episode*n_agent, n_obs+n_actions+n_agent)
        '''
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        # obs: (n_episode, n_agent, n_obs)
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        # 给obs添加上一个动作、agent编号

        if self.args.last_action:
            # inputs append (n_episode, n_agent, n_action) tensor
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:                
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            # 因为当前的obs三维的数据，每一维分别代表(episode编号，agent编号，obs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把obs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        ''' 输出q矩阵(乘以相应的task score)。以及选择了哪个任务
        '''
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        i_tasks, i_task_targets= [], []
        for transition_idx in range(max_episode_len):
            # inputs, inputs_next：(episode_num * n_agents, n_obs+n_actions+n_agent).表示
            inputs, inputs_next = self._get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()


            # rnn输出: 整个矩阵(包括task score)
            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)  # q_eval (n_episode*n_agents, n_tasks, n_actions+1)
            q_target, self.target_hidden= self.target_rnn(inputs_next, self.target_hidden)

            _, i_task= self.find_task_q(q_eval)
            _, i_task_target= self.find_task_q(q_target)        # 找出task score最大的行。
            q_table_shape = list(q_eval.shape)
            q_table_shape[-1] -= 1 

            # 将q矩阵每一项乘以对应的task score. 
            # q_eval:  (n_episode*n_agents, n_tasks, n_actions)
            q_eval = q_eval[..., 0].unsqueeze(-1).expand(q_table_shape) * q_eval[..., 1:]
            q_target = q_target[..., 0].unsqueeze(-1).expand(q_table_shape) * q_target[..., 1:]
            
            # 把q_eval维度重新变回(episode_num, n_agents, n_tasks, n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, self.n_tasks,-1)
            q_target = q_target.view(episode_num, self.n_agents, self.n_tasks, -1)
            i_task = i_task.view(episode_num, self.n_agents, -1)
            i_task_target = i_task_target.view(episode_num, self.n_agents, -1)
            
            q_evals.append(q_eval)
            q_targets.append(q_target)
            i_tasks.append(i_task)
            i_task_targets .append(i_task_target)
        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(n_episode, n_agents，n_actions)
        # 把该列表转化成(n_episode, max_episode_len， n_agents，n_actions)的数组
        # 将for里面的每个时刻叠起来
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        i_tasks = torch.stack(i_tasks, dim=1).squeeze(-1)
        i_task_targets = torch.stack(i_task_targets, dim=1).squeeze(-1)
        
        # q_evals (episode_num, max_episode_len, n_agents, n_tasks, n_actions)
        return q_evals, q_targets, i_tasks, i_task_targets
    
    def find_task_q(self, q):
        ''' 给定带task score的q矩阵。输出选择了啥动作。以及对应的一行(乘上了相应的task score)。
        Params:
            q: (n_episode, n_tasks, n_actions+1)。带task score的q矩阵。
        Returns:
            q:  (n_episode, 1, n_actions)。 
            i_task: (n_episode, n_tasks, n_actions)
        '''
        q_shape = q.shape
        # 1. 取最后一个维度第一个数作为task selector, 选择最擅长的task
        _, i_task = q[...,0].max(dim = -1)             # i_task shape: (n_episode)
        # 2. 取对应项相乘，计算
        i_task = i_task.unsqueeze(-1).unsqueeze(-1)     # i_task shape: (n_episode, 1, 1)
        i_task_shape = list(q_shape)
        i_task_shape[-2] =  1                       
        i_task = i_task.expand(i_task_shape)                   #  i_task_shape: (n_episode, 1, n_action + 1)
        q = torch.gather(q, dim=-2, index=i_task)           # q shape (n_episode, 1, n_action + 1)。 选择q中仅与最高分任务的一行
        q = (q[...,0].unsqueeze(-1) *q[...,1:]).squeeze(-2)     # task_score * 对应动作q值。 q shape : (n_episode,  n_action)
        i_task = i_task.squeeze(1)                                          # 将 i_task task维度去掉(因为该维度值必为1)
        i_task = i_task[..., 0]                                                         # i_task shape (n_episode)
        return q, i_task

    def matrix(self, q_eval, action): 
        q_table_shape = list(q_eval.shape)
        q_table_shape[-1] -= 1 
        q_eval = q_eval[..., 0].unsqueeze(-1).expand(q_table_shape) * q_eval[..., 1:]
        q_eval = q_eval[0].gather(1, torch.tensor([[action, action, action]]).t()).squeeze(-1).unsqueeze(0)
        return q_eval

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        time_n = time.time()
        torch.save(self.eval_task_net.state_dict(), self.model_dir + '/' + num +str(time_n) + '_td_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num +str(time_n) +  '_rnn_net_params.pkl')
