import argparse

import gymnasium as gym
import options_envs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    env = gym.make("OptionsEnv/TwoRooms-v0")

    try:
        for episode in range(args.episodes):
            observation, info = env.reset(seed=args.seed + episode)
            del observation, info

            total_reward = 0.0
            terminated = False
            truncated = False
            steps = 0

            while steps < args.max_steps and not (terminated or truncated):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                del observation, info
                total_reward += reward
                steps += 1

            print(
                f"episode={episode} steps={steps} "
                f"total_reward={total_reward:.2f} success={terminated}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
