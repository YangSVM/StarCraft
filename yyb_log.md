# 安装过程
1. 星际环境：最好安装在 ~/StarCraftII目录下。否则需要
   ```bash
   export SC2PATH=/home/tiecun/codes/MARL/simulation/StarCraftII
   ```
2. MPE粒子环境:注意，需要使用低版本gym. 
   ```bash
   pip install gym==0.10.5
   ```
3. install from scratch
   ```bash
   conda create -n marl python=3.6
   pip install pytorch
   <!-- scp -P 端口号 本地文件 root@ip:远程文件夹 -->
   scp -P 11730 SC2.4.6.2.69232.zip root@wx.blockelite.cn:/root/marl/StarCraftII 
   scp -P 11730 root@wx.blockelite.cn:/root/marl/StarCraft/result/task_decomposition/2s3z/plt_0.png .
   unzip
   git clone https://github.com/YangSVM/StarCraft.git
   git clone https://github.com/YangSVM/smac.git
   pip install smac/

   <!-- edit .bashrc -->
   export SC2PATH=/home/tiecun/codes/MARL/simulation/StarCraftII
   <!-- test smac installation -->
   python -m smac.examples.random_agents
   ```

# 服务器使用相关
- ssh关闭后仍然使得代码保持跑。用法：
  - ssh into the remote machine
  - start tmux by typing tmux into the shell
  - start the process you want inside the started tmux session
  - leave/detach the tmux session by typing Ctrl+b and then d
  - 重新激活： `tmux attach`
  - 在tmux环境中时，可以重命名环境，using Ctrl+b and $. 
  - 重新激活指定环境 `tmux attach-session -t <session-name>`
  - 列举已有的环境 `tmux ls`
- 已有环境
  - 小破台式机：`ssh thicv@166.111.50.91` 
  - 借用服务器： `ssh root@wx.blockelite.cn -p 11730`
  - cmar服务器：`ssh intern@222.29.136.16`

# 环境学习
## MPE
1. 没有好的教程，直接看源码：/path/to/mpe/multiagent/scenarios/simple_tag.py
2. 源码可以任意修改，达到你想要的效果
3. 使用simple_tag：红色是追逐者(我方，速度慢，源码中为 adversaries)，绿色是逃避者(敌方，速度快，源码中为good_agent)，灰色是障碍物。红方要尽快抓到绿方。
4. 智能体的总数，敌我双方的数量都可以修改
5. 环境的观测：每个智能体的观测（自己的位置、速度，障碍物位置，其他智能体的**相对**位置、逃跑者的绝对速度）.
6. 环境的输入。维度：智能体数量。每个智能体的输入维度：(2*dim_p +1)。position的维度，即xy为2，总共2\*2+1=5，输入为5维的space.Discrete（即每个智能体用一个0-4的整数进行控制）。代码写的太过于混乱，简要意思可能是用力进行控制，1,3,5维度的值可以为0，只用2,4维的值，该值可以是连续的量，乘以一定比例可以做横纵向控制。
7. 如果需要在该环境使用维诺图等传统算法，需要改动环境....环境实在是写得太糟糕了。
8. OpenAI论文中，用MADDPG算法打败DDPG算法证明优越性

## 星际争霸环境
1. 6688 https://github.com/starry-sky6688/StarCraft
2. 环境训练起来非常慢...自己的笔记本没有办法进行训练
3. 分成不同地图，不同难度的对手


# 源码阅读
## QMIX
1.  构建过程：(s,u,r,s',u')。给定轨迹，能够计算出小q，以及max q。max q经过mix网络默认得到max Q。
2. 疑问：时序过程？？？mask
3. 
4. main函数进入，使用qmix，则相当于使用get_common_args， get_mixer_args中的参数；没有evaluate
5. runner中，采用的是Agent, RolloutWorker, ReplayBuffer最简单的三个类。
6. runner.run 每次采集一个episode时，记录游戏长度(steps)，(注意返回的episode数组还是固定长度的，但是填充了0)，并将steps累加成time_step，当time_step>5000时，evaluate一次，保存并更新图片，胜率等结果。如果time_step> 2百万，则停止。
7. 上述 runner.run 被运行8次重复试验。

### 学习流程：采集数据，存入buffer，从buffer中sample样本，用样本进行train。
- 采集数据：`rollout.py`中`generate_episode`
  - Returns:
    - `episode_batch`  dict。每个key：u,s,r,u_next... 维度：*(n_episode, episode_len, n_agents, 具体维度) *，如果变量与n_agent无关，维度缩减为(n_episode, max_episode_len,  具体维度).
- 存入buffer并且从buffer中sample出mini_batch进行训练。变量维度不变
- 进入train时，首先会将mini_batch中的episode_len进行裁剪。因为可能存在一个batch中都是很快就terminated的场景。裁剪后为batch。dict中每个 **value** 都是 (n_episode, max_episode_len, n_agents, 具体维度)。max_episode_len为所有episode中最长的episode.
- 训练时，每经过args.save_cycle次训练，保存一次模型.
- 每次进行训练时policy 对应中的`learn`函数:
  - 输入batch.变量维度与上述相同*(n_episode, max_episode_len, n_agents, 具体维度)*，如果变量与n_agent无关，维度缩减为(n_episode, max_episode_len,  具体维度). 从np array转成torch。
  - 一些变量的含义：
    - terminated: (n_episode, max_episode_len, 1). 表示i轨迹在j时刻是否已经停止游戏
    - padded: (n_episode, max_episode_len, 1).表示i轨迹在j时刻是否是填充的0数据
    - o, u, s,r,o_next,s_next, avail_u,avail_u_next, u_onehot 均为字母相应意义。其中u是整数编码，其余的avail_u等与u相关的量都是独热编码。
  - init_hidden。将输入轨迹的rnn的 hidden_state 设置为0。
  - get_q_valus：计算q函数。
    - get_input：截取当前trasition_idx时刻(固定episode中某个时刻)的输入值(obs, action_onehot, agent_onehot)，self.eval_hidden 和 self.target_hidden 保存当前时刻的hidden_states。输出q_eval, self.eval_hidden,是(n_agent, n_actions).
    - 将input和hidden_states输入到rnn中。得到q_evals和q_targets。维度均为(n_episode, max_episode_len, n_agents, 动作空间维度)
    - 因为IGM，且智能体仅采取了一个动作。q_evals 和 q_targets维度缩减成(n_episode, max_episode_len, n_agents)
    - 送入eval_qmix_net 和 target_qmix_net中，计算全局的Q值。计算TD error，并且反向传播。
- QMIX网络。
  - 输入：q_evals. 维度(n_episode, max_episode_len, n_agents)；状态s，维度(n_episode, max_episode_len, 状态空间维度)
  - 输出：Q_total。维度(n_episode, max_episode_len, 1)
  - 网络设计：首先f(states)生成w1,b1,w2,b2，所以需要相应的4个超参数网络。超参数网络均为nn.Linear，hyper_w1的网络为输入(n, n_states), 输出(n, n_agents * qmix_hidden_dim), 
  - 使用torch.bmm乘法。为torch.mm的3D版本。torch.mm只能做2D矩阵的乘法。torch.bmm相当于输入维度都在第一维增加了batch维度，batch维度大小必须相同。
1.  疑问：为什么reuse_network(所有智能体共享同样的参数)时，需要输入ID作为trajectory？

## 环境解析 - SMAC
1. SMAC: 牛津oxwhirl组。https://github.com/oxwhirl/smac
2. 星际争霸属于RTS(即时战略游戏)，分为宏观和微观，宏观主要指的是做决策控制经济、生产等，微观主要是对个体的微操。SMAC主要关注微观层面
3. 共22个场景。地图大多是无障碍物平地，也有高地和障碍物，有各种不同的兵种和数量，有场景己方人数劣势。http://whirl.cs.ox.ac.uk/blog/smac/ 视频展示清楚。同时也有一些有趣的场景，必须学的一定的策略。
4. 微操常用的技术和技巧：集中火力攻击同时避免过度火力，风筝，编队。
5. 敌方单元由内置的非学习的游戏AI完成
6. 文档或博客中详细叙述了状态和观测、动作空间、奖励的相关细节
   - observation：(n_agents * 特征维度) 只能观测视野范围内的友军或者敌军的状态：距离、相对x,y,血条，护盾(护盾没有被打破时，智能体血条不会减小)，unit type，上一时刻动作(友军才能看)；周围地形8个点的walkability以及terrian height。
   - state: 训练时使用的全局信息。包含了observation，以及所有智能体的绝对位置，cooldown/energy，所有智能体上一时刻的动作；
   - Actions: 离散动作。move：上下左右，attack[enemy id], heal[agent id] (only for Medivacs),stop。攻击或者治疗。只能在shooting range内
   - Rewards: 可以提供稀疏奖励(终局胜负奖励)，也可以提供较为dense的奖励。不鼓励做奖励特殊化。
   - 未见到有文档能够详细解析观测或者state每一维度具体的含义。示例代码：smac random_agents.py
   - 每个场景状态、观测、动作、智能体数量都不相同。
7. 代码阅读random_agents.py
   - actions: 每个智能体只能采取一个动作。如移动就不能够射击。
   - 环境使用函数示意：
   ```python
   obs = env.get_obs()
   state = env.get_state()
   reward, terminated, _ = env.step(actions)
   ```
   - obs是一个list，每个智能体有一个局部观测，每个观测是一个np。state是全局的一个np。actions是一个list，每个元素是一个int代表某一个特定动作的下标。
8. SMAC代码阅读
   1. 包位置：~/anaconda3/envs/marl/lib/python3.6/site-packages/smac
   2. 主要文件为env下的 startcraft2/startcraft2.py 和multiagentenv.py
   3. obs主要使用 get_obs_agent 函数，主要包括以下4个维度(按顺序)
     - agent movement features (where it can move to, height information and pathing grid)
     - enemy features (available_to_attack, relative_x, relative_y, health, shield, unit_type)
     - ally features (visible, distance, relative_x, relative_y, shield, unit_type)
     - agent unit features (health, shield, unit_type)
   4. obs可以通过函数           functions ``get_obs_move_feats_size()``, ``get_obs_enemy_feats_size()``, ``get_obs_ally_feats_size()`` and``get_obs_own_feats_size()``.获取各个维度的大小。
   5. 环境各异：The size of the observation vector may vary, depending on the environment configuration and type of units present in the map. For instance, non-Protoss units will not have shields, movement features may or may not include terrain height and pathing grid, unit_type is not included if there is only one type of unit in the map etc.). 不是神族没有护盾，平地没有pathing grid或者高度特征，场景只有一个种族时，没有type of unit。
   6. **Observation** `get_obs_agent` 函数代码阅读
      - 首先根据函数获得各个观测的维度，并创建0数组
      - 如果该智能体血条为0，直接返回全0的数组
      - Movement features: 前4个特征代表是否能够向4个方向运动，后面根据场景，可能有get_surrounding_pathing 和 get_surrounding_height信息
      - Enemy features: 敌方个数*特征维度。敌方智能体必须在sight range并且生命值大于0.各维度意思：(available_to_attack, distance, relative_x, relative_y, health, shield, unit_type)，前面5项一般有。
      - Ally features: (我方总个数-1)*特征维度。我方智能体必须在sight range并且生命值大于0.各维度意思：(visible, distance, relative_x, relative_y, health, shield, unit_type, last_action)，前面5项一般有。
      - Own features: 一维特征：(health, sheild, unit)起码有第一个。
   7. **Reward** 奖励 `reward_battle`: 当非稀疏时，调用 `reward_battle` 函数计算reward奖励
      - 函数说明。一般仅考虑敌方，返回累加和：护盾和血条的减少量，以及每个敌方死亡奖励；如果flag reward_only_positive为True，进一步考虑我方智能体的负的reward，内容与敌方相同。
   8. **State**：`get_state` 函数阅读。以下维度按顺序
      - ally_state: 智能体数目*特征维度。我方智能体状态。(health, energy/cd, x, y) ,optional:(shield, type)
      - enemy_state：智能体数目*特征维度。敌方智能体状态。(health, x, y), optional:(shield, type)
      - (optional)state_last_action: 上一时刻动作
      - (optional)state_timestep_number：时间进度。
      将以上矩阵或向量flatten再concate起来.
9. **Action**： `get_avail_agent_actions`: 长度为n_agent的list，每个元素由以下组成
    - [0] isDead : no-op。如果死亡，为1，只能选择这个动作。
    - [1] stop : 是否停止
    - [2:5] move ： 北南东西
    - attack或者医疗兵的技能
10. `step`函数解读
    - 输出 reward, terminated, info.
      - terminated: bool。可能由于pysc2游戏bug或者错误的请求终止导致
    - 首先将得到的 actions，转化为pysc2可用的动作，并向客户端进行请求 。
    - `self.update_units()`更新所有观察，到类属性中(如self.enemies, self.agents等)。返回 `game_end_code` 如果游戏能够判定胜负或者平局，就分别为+-1或0，如果还未能判定则返回None
    - $max_reward = n_ennemies * reward_death_value + reward_win + 所有敌人的血量和护盾值之和$, $reward_scale_rate = 20$,返回的reward计算方法： $reward = reward / max_reward * reward_scale_rate$
    - 

## 代码实现：
1. 如何使用2个loss更新不同的参数：https://discuss.pytorch.org/t/how-to-have-two-optimizers-such-that-one-optimizer-trains-the-whole-parameter-and-the-other-trains-partial-of-the-parameter/62966
2. 


## 调试日志
1. 一开始采样的时候，任务的score大部分为0，且总是选择任务0，其他任务的score为0的概率大很多。
2. winning rate始终为0...大问题
3. pytorch如果需要反复经过同一个变量进行反向传播，并且需要保留中间变量的梯度时，需要 retain_graph=True
4. SMAC 3m中有奇怪的事情，有时候攻击敌方，敌方血条没有减少: 的确是环境设置的问题，可能是技能CD还是游戏的特性？检查过所有reward加和仍然是调用环境中的reward中的值，证明reward确实为0
5. QMIX在台式机上大概15分钟(65k time_steps)就能够在3m上得到较高的胜率.(最大80，平均50)
6. pytorch 和 numpy 的切片索引相同，如果a.shape (2,3), a[0,:].shape为(3,)。即单独索引减小一个维度。
7. 胜率很久都为0，问题分析：
   - local q中使用relu函数？
   - q 值相乘，导致值很小？
   - task 和 q 没有分开，用的同一个网络

## 会议任务
8. matrix game：验证方法。最好考虑dependency。matrix game代码验证
9. 为什么qplex方法存在局限性，充要的条件退化到matrix game是什么

## 工作计划
- qmix在3m中的reward最大是20
  - 查看环境代码有无改写错误：错误已修改。并且加深了对环境reward的理解。
  - 使用qmix跑跑看现在代码的情况 ：跑过了。感觉效果不错。
- task选择的时候使用 探索: 相当于整体使用了探索。已经包含. 
- 跑更多的图 (修改环境后3m的图效果还可以, 2s3z的效果感觉有点差，)
- 可视化回放。
