import cv2
import gym
from os import path
from os import makedirs

import argparse
from tqdm.auto import trange

ROOT='path/to/data'

def save_frames_to_video(frames, output_file, fps=30):
    # Get the shape of the frame to set the video width and height
    height, width, layers = frames[0].shape
    size = (width, height)
    
    # Define the codec and create VideoWriter object
    # You can use different codecs, here 'mp4v' is used for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, size)

    for frame in frames:
        out.write(frame)

    # Release the VideoWriter object
    out.release()

def main(args):
    env_name = args.env_name
    num_envs = args.num_envs
    timeout  = args.timeout

    for seed in trange(num_envs, desc=f'Generating {env_name} videos'):
        env = gym.make(
            f'procgen:procgen-{env_name.lower()}-v0',
            distribution_mode="hard",
            render_mode='rgb_array',
            start_level=seed,
            num_levels=1,
            use_sequential_levels=True,
        )

        frames = [env.reset()]
        frames.extend([
            env.step(env.action_space.sample())[0]
            for _ in range(timeout - 1)
        ])
        
        env.close()
        
        savepath = path.join(args.root, env_name, f'{str(seed).zfill(4)}.mp4')
        makedirs(path.dirname(savepath), exist_ok=True)
        
        save_frames_to_video(frames, savepath)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate videos of a gym environment')
    parser.add_argument('--env_name', type=str, default='Coinrun', help='Name of the environment')
    parser.add_argument('--num_envs', type=int, default=1, help='Number of samples to generate')
    parser.add_argument('--timeout', type=int, default=1000, help='Timeout for generating samples')
    parser.add_argument('--root', type=str, default=ROOT, help='Root folder where to save the videos')

    args = parser.parse_args()

    main(args)