import argparse
import torch
import os.path as osp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str, help='Name of experiment')
    parser.add_argument('-i', '--iteration', type=str, default='latest', help='Number in net_g_().pth, Default: latest')
    args = parser.parse_args()

    load_path = osp.join('./experiments/', args.experiment, 'models', 'net_g_' + args.iteration + '.pth')

    load_net = torch.load(load_path)
    load_net = load_net['params_ema']

    save_path = osp.join('./experiments/pretrained_models/', args.experiment[6:] + '.pth')
    torch.save(load_net, save_path, _use_new_zipfile_serialization=False)

if __name__ == '__main__':
    main()