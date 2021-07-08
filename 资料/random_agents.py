from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env import StarCraft2Env
import numpy as np


def main():
    env = StarCraft2Env(map_name="8m")
    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]

    n_episodes = 10

    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            obs = env.get_obs()
            state = env.get_state()

            actions = []
            # 每个智能体只能采取一个动作
            for agent_id in range(n_agents):
                # [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]。生成可以执行的动作
                avail_actions = env.get_avail_agent_actions(agent_id)
                # array([1, 2, 3, 4, 5])。获取非零值的ID。
                avail_actions_ind = np.nonzero(avail_actions)[0]
                # 从这些非零ind中选择一个作为动作
                action = np.random.choice(avail_actions_ind)
                actions.append(action)

            reward, terminated, _ = env.step(actions)
            episode_reward += reward

        print("Total reward in episode {} = {}".format(e, episode_reward))

    env.close()


if __name__ == "__main__":
    main()
