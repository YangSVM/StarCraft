from runner import Runner
from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args, get_task_decomposition_args, get_multi_reward_args
import mg_complex
import mg_simple

if __name__ == '__main__':
    for i in range(8):
        args = get_common_args()
        args.time = i
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
        if args.matrix_game == False: 
            env = StarCraft2Env(map_name=args.map,
                            step_mul=args.step_mul,
                            difficulty=args.difficulty,
                            game_version=args.game_version,
                            replay_dir=args.replay_dir,
                            reward_task_dec_type=args.task_dec_type)
        elif args.matrix_difficulty == 'complex': 
            env = mg_complex.matrix_game()
        elif args.matrix_difficulty == 'simple': 
            env = mg_simple.matrix_game()

        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]


        if args.matrix_game == True: 
            args.n_tasks = env_info["n_tasks"]

        if args.task_dec_type !='':
            args = get_multi_reward_args(args, env)

        if args.alg.find('task_decomposition') > -1:
            args = get_task_decomposition_args(args)

        if args.matrix_game == True: 
            args.epsilon_anneal_scale = 'no_decay'
            args.map = 'matrix game'
            args.evaluate_cycle = 500

        runner = Runner(env, args)
        if args.multi_process_n >  -1:
            n_run = 1 * args.multi_process_n + i
        if not args.evaluate:
            runner.run(n_run)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
