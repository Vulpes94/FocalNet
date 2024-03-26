import os
import torch
import argparse
from Net.FocalNet import build_net
from train import _train
from eval import _eval

def main(args):
    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net()
    # print(model)

    model.to(device=args.device)
    
    if args.mode == 'train':
        _train(model, args)

    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='FocalNet', type=str)

    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument("--device", type=str, default='cuda:0', help="device")

    # Train
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=8e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=16)
    parser.add_argument('--save_freq', type=int, default=20)
    parser.add_argument('--valid_freq', type=int, default=20)
    parser.add_argument('--resume', type=str, default='')


    # Test
    parser.add_argument('--test_model', type=str, default='')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', 'ITS/', 'train')
    args.result_dir = os.path.join('results/', 'ITS/', 'test')

    main(args)
