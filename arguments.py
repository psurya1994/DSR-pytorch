import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--env', default='MiniGrid-Empty-Random-5x5-v0', help='algorithm to use: MiniGrid-Empty-Random-5x5-v0 | MiniGrid-Empty-Random-6x6-v0 | VizdoomBasic-v0')

    args = parser.parse_args()

    return args