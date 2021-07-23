from runner import Runner
from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args, get_task_decomposition_args


if __name__ == '__main__':
    args = get_common_args()

    # evalueate args
    args.load_model = True
    # 必须是绝对路径才能保存
    args.replay_dir='/home/tiecun/codes/MARL/StarCraft/replay'
    args.evaluate_epoch = 1


    if args.alg.find('coma') > -1:
        args = get_coma_args(args)
    elif args.alg.find('central_v') > -1:
        args = get_centralv_args(args)
    elif args.alg.find('reinforce') > -1:
        args = get_reinforce_args(args)
    else:
        args = get_mixer_args(args)
    if args.alg.find('commnet') > -1:
        args = get_commnet_args(args)
    if args.alg.find('g2anet') > -1:
        args = get_g2anet_args(args)

    env = StarCraft2Env(map_name=args.map,
                        step_mul=args.step_mul,
                        difficulty=args.difficulty,
                        game_version=args.game_version,
                        replay_dir=args.replay_dir,
                        reward_task_dec_type=args.task_dec_type)

    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]

    if args.alg.find('task_decomposition') > -1:
        args = get_task_decomposition_args(args, env)
        
    if args.task_dec_type !='':
        args = get_task_decomposition_args(args, env)


    runner = Runner(env, args)

    win_rate, _ = runner.evaluate()
    print('The win rate of {} is  {}'.format(args.alg, win_rate))
    
    env.close()
