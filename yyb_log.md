# 安装过程
1. 星际环境：最好安装在 ~/StarCraftII目录下。否则需要
   ```bash
   export SC2PATH=/home/tiecun/codes/MARL/simulation/StarCraftII
   ```
2. MPE粒子环境:注意，需要使用低版本gym. 
   ```bash
   pip install gym==0.10.5
   ```

# 环境学习
## MPE ----弃用该环境。改写比较麻烦
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
6. 

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
   6. `get_obs_agent` 函数代码阅读
      - 首先根据函数获得各个观测的维度，并创建0数组
      - 如果该智能体血条为0，直接返回全0的数组
      - Movement features: 前4个特征代表是否能够向4个方向运动，后面根据场景，可能有get_surrounding_pathing 和 get_surrounding_height信息
      - Enemy features: 敌方个数*特征维度。敌方智能体必须在sight range并且生命值大于0.各维度意思：(available_to_attack, distance, relative_x, relative_y, health, shield, unit_type)，前面5项一般有。
      - Ally features: (我方总个数-1)*特征维度。我方智能体必须在sight range并且生命值大于0.各维度意思：(visible, distance, relative_x, relative_y, health, shield, unit_type, last_action)，前面5项一般有。
      - Own features: 一维特征：(health, sheild, unit)起码有第一个。
   7. reward奖励: 当非稀疏时，调用 `reward_battle` 函数计算reward奖励
      - 函数说明。一般仅考虑敌方，返回累加和：护盾和血条的减少量，以及每个敌方死亡奖励；如果flag reward_only_positive为True，进一步考虑我方智能体的负的reward，内容与敌方相同。
   8. state：`get_state` 函数阅读。以下维度按顺序
      - ally_state: 智能体数目*特征维度。我方智能体状态。(health, energy/cd, x, y) ,optional:(shield, type)
      - enemy_state：智能体数目*特征维度。敌方智能体状态。(health, x, y), optional:(shield, type)
      - (optional)state_last_action: 上一时刻动作
      - (optional)state_timestep_number：时间进度。
      将以上矩阵或向量flatten再concate起来.
9. QMIX源码阅读：
   - 6688


## 会议任务
8. 下一步方向：找reward和states。reward：能否分开。可以分开，所有的观测都能够拿到
9. matrix game：验证方法。最好考虑dependency。