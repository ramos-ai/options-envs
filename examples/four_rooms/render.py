import argparse

import gymnasium as gym
import options_envs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    env = gym.make("OptionsEnv/FourRooms-v0", render_mode="ansi")

    try:
        observation, info = env.reset(seed=args.seed)
        del observation, info
        print(env.render(), end="")

        for step in range(args.steps):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            del observation, reward, info
            print(f"step={step + 1} action={action}")
            print(env.render(), end="")

            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
