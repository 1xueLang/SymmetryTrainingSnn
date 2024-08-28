import os
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torchvision
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

from snetx import cuend
from snetx import utils as snutils
import snetx.snn.algorithm as snnalgo
from snetx.dataset import vision as snnvds

import augment
from modules import cnn10, vgg11, resnet
import training

def execuate(device, args):
    if args.seed > 0:
        snutils.seed_all(args.seed)
        cuend.utils.seed_cupy(args.seed)

    if args.dataset == 'CIFAR10':
        tr_data, ts_data = snnvds.cifar10_dataset(args.data_dir, args.batch_size1, args.batch_size2,
                                                  augment.cifar10_transforms(False, True))
        dvs = False
    elif args.dataset == 'CIFAR100':
        tr_data, ts_data = snnvds.cifar100_dataset(args.data_dir, args.batch_size1, args.batch_size2)
        dvs = False
    elif args.dataset == 'CIFAR10DVS':
        tr_data, ts_data = snnvds.cifar10dvs_dataset(args.data_dir, args.batch_size1, args.batch_size2, args.T)
        dvs = True
    elif args.dataset == 'DVSGesture128':
        tr_data, ts_data = snnvds.dvs128gesture_dataset(args.data_dir, args.batch_size1, args.batch_size2, args.T)
        dvs = True
    elif args.dataset == 'ImageNet':
        tr_data, ts_data = snnvds.imagenet_dataset(args.data_dir, args.batch_size1, args.batch_size2)
        dvs = False
    else:
        raise ValueError(f'{args.dataset} not supported.')

    symm_config = {
        'tau': args.tau,
        'scale': args.scale,
        'symm_training': args.hs,
        'symm_connect': args.fs
    }

    if 'resnet' == args.arch:
        net = resnet.__dict__[args.arch](symm_config, num_classes=args.num_classes).to(device)
    elif 'cnn10' == args.arch:
        net = cnn10.CNN10(symm_config, 2 if dvs else 3, num_classes=args.num_classes).to(args.device)
    elif 'vgg11' == args.arch:
        net = vgg11.VGG11(
            symm_config, 2 if dvs else 3, tau=args.tau, num_classes=args.num_classes, dropout=args.drop).to(args.device)
    else:
        raise ValueError(f'{args.arch} not supported')

    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None

    if args.debug:
        writer = None
    else:
        time_str = datetime.now().strftime('%Y%m%d%H%M%S')
        writer = SummaryWriter(
            log_dir=f'{args.logs_dir}/{args.dataset}/{args.arch}/hs-{args.hs}-fs-{args.fs}/{time_str}')

    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-5)
    criteriona = torch.nn.CrossEntropyLoss()
    criterions = snnalgo.TET(torch.nn.CrossEntropyLoss(), )

    max_acc = 0.
    for e in range(args.num_epochs):
        if args.verbose:
            dataloader = tqdm(tr_data)
        else:
            dataloader = tr_data

        if dvs:
            correct, sumup, loss = training.train_dvs(net, dataloader, optimizer, criterions, criteriona, scaler, device, args)
            correct, sumup = training.validate(net, ts_data, device, args, static=False)
        else:
            correct, sumup, loss = training.train_static(net, dataloader, optimizer, criterions, criteriona, scaler, device, args)
            correct, sumup = training.validate(net, ts_data, device, args)

        s_acc = correct[0] / sumup
        a_acc = correct[1] / sumup
        h_acc = correct[2] / sumup
        if not args.debug:
            writer.add_scalar('Loss', loss, e)
            writer.add_scalar('SnnAcc', s_acc, e)
            writer.add_scalar('AnnAcc', a_acc, e)
            writer.add_scalar('Hybrid', h_acc, e)

        if args.save and max_acc <= s_acc:
            torch.save(net.state_dict(),
                       f'pths/{args.dataset}_{args.arch}_hs_{args.hf}_fs{args.fs}.pth')

        max_acc = max(max_acc, s_acc)

        print('epoch: ', e,
              f'loss: {loss:.4f}, '
              f'snn acc: {s_acc * 100:.2f}%, '
              f'ann acc: {a_acc * 100:.2f}%, '
              f'bybrid acc: {h_acc * 100:.2f}%, '
              f'Best: {max_acc * 100:.2f}%')

        scheduler.step()

    if not args.debug:
        writer.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--hs', action='store_true')
    parser.add_argument('--fs', action='store_true')
    parser.add_argument('--tau', type=float, default=2.0)
    parser.add_argument('--scale', type=float, default=0.5)

    parser.add_argument('--batch_size1', type=int, default=64, help='batch size for single device.')
    parser.add_argument('--batch_size2', type=int, default=256, help='test batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate for gradient descent.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='penal term parameter for model weight.')
    parser.add_argument('--num_epochs', type=int, default=200, help='max epochs for train process.')
    parser.add_argument('--print_intv', type=int, default=50,
                        help='train steps interval to print train mesasges: show messages after each {intv} batches.')

    parser.add_argument('--T', type=int, default=6, help='snn simulate time step.')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='')
    parser.add_argument('--data_dir', type=str, default='./dataset', help='data directory.')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--logs_dir', type=str, default='./LOGS/', help='logs directory.')

    parser.add_argument('--arch', type=str, default='resnet18', help='network architecture.')
    parser.add_argument('--drop', type=float, default=0.4, help='')

    parser.add_argument('--seed', type=int, default=2025, help='')
    parser.add_argument('--amp', '-A', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--debug', '-D', action='store_true')
    parser.add_argument('--verbose', '-V', action='store_true',
                        help='whether to display training progress in the master process.')
    parser.add_argument('--save', '-S', action='store_true', )

    cmd_args = parser.parse_args()

    execuate(torch.device(cmd_args.device), cmd_args)


if __name__ == '__main__':
    main()