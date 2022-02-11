# main

import argparse
from src.utils import *
from torch.utils.data import DataLoader
from src import train
from src import test
# ddp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

parser = argparse.ArgumentParser(description='RERD')
parser.add_argument('-f', default='', type=str)

# Settings and parameters

# Fixed
parser.add_argument('--model', type=str, default='RERD',
                    help='name of the model to use')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for ddp training')
parser.add_argument('--ddp', type=bool, default=False, help='using distributedDataParallel')

# Tasks
parser.add_argument('--vonly', action='store_true',
                    help='use the crossmodal fusion into v (default: False)')
parser.add_argument('--aonly', action='store_true',
                    help='use the crossmodal fusion into a (default: False)')
parser.add_argument('--lonly', action='store_true',
                    help='use the crossmodal fusion into l (default: False)')
parser.add_argument('--aligned', action='store_true',
                    help='consider aligned experiment or not (default: False)')
parser.add_argument('--dataset', type=str, default='mosei_senti',
                    help='dataset to use (default: mosei_senti)')
parser.add_argument('--use_bert', type=bool, default=False, help='use bert text feature')
parser.add_argument('--bert_path', type=str, default='/home1/xiaobao/OrthTransFusion/bert_data/MMSA/pretrained_model/',
                    help='path for bert pretrained model')
parser.add_argument('--train_mode', type=str, default='regression', help='classification or regression')
parser.add_argument('--data_path', type=str, default='/home1/xiaobao/OrthTransFusion/bert_data/MMSA/MOSI',
                    help='path for storing the dataset')

parser.add_argument('--emotion', type=str, default='happiness', help='iemocap emotion class')

# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                    help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                    help='attention dropout (for visual)')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.5,
                    help='output layer dropout')

# distillation encoder params
parser.add_argument('--dis_factor', type=int, default=3, help='distillation encoder factor')
parser.add_argument('--dis_d_model', type=int, default=64, help='multihead layer feature dimension')
parser.add_argument('--dis_n_heads', type=int, default=4, help='number of encoder heads')
parser.add_argument('--dis_e_layers', type=int, default=2, help='number of encoding layers')
parser.add_argument('--dis_d_ff', type=int, default=64, help='number of EncoderLayer conv channels')
parser.add_argument('--dis_dropout', type=float, default=0.2,
                    help='dropout in distillation encoder')
parser.add_argument('--dis_attn', type=str, default='prob', help='prob for distillation attention')

# Architecture
parser.add_argument('--nlevels', type=int, default=5,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=3,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')
parser.add_argument('--temp_proj', type=int, default=1, help="temporal projection number ")

# Tuning
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer to use (SGD/Adam)')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')
# ITS-n-mosei-lav-d64h6l2-e80bs32cyc-regsh-s581-01
parser.add_argument('--pretrain', type=str, default=None, help='load pretrained model name')

# regularization params
parser.add_argument('--reg_en', type=bool, default=False, help="multimodal regularization by maximizing functional "
                                                               "entropies")
parser.add_argument('--reg_lambda', type=float, default=0.1, help='lambda coefficient for regularization')

# learning rate
parser.add_argument('--schedule', type=str, default='CyclicLR', help='learning rate schedule; cycliclr or warmup or '
                                                                     'salr')
parser.add_argument('--base_lr', type=float, default=1e-4, help='base learning rate for cyclicLR')
parser.add_argument('--max_lr', type=float, default=1e-2, help='max learning rate for cyclicLR')
parser.add_argument('--step_up', type=int, default=2000, help='step size up for cyclicLR on mosei, other dataset '
                                                              'subject to change')
# warm up learning rate --schedule = 'warmup'
parser.add_argument('--lr_stepper', type=str, default='steplr', help='stepper lr after warmup; explr')
parser.add_argument('--stepper_size', type=int, default='10', help='stepLR decay step size')
parser.add_argument('--warm_epoch', type=int, default='5', help='number of warmup epochs')

# Logistics
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=581,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='ots',
                    help='name of the trial (default: "ots")')
parser.add_argument('--test', type=bool, default=False, help='test enable')

args = parser.parse_args()

torch.manual_seed(args.seed)
dataset = str.lower(args.dataset.strip())
valid_partial_mode = args.lonly + args.vonly + args.aonly

use_cuda = False

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
    'sims': 1
}

criterion_dict = {
    'mosi': 'L1Loss'
}

torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        if args.ddp:
            dist.init_process_group(backend='nccl')
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True


# Loading the dataset

print("Start loading the data....")

train_data = get_data(args, dataset, 'train')
valid_data = get_data(args, dataset, 'valid')
test_data = get_data(args, dataset, 'test')

if args.ddp:
    train_sampler = DistributedSampler(train_data)
    valid_sampler = DistributedSampler(valid_data)
    test_sampler = DistributedSampler(test_data)
    nproc = torch.cuda.device_count()
    shuffle, drop_last = False, False

else:
    train_sampler = valid_sampler = test_sampler = None
    nproc = 1
    shuffle, drop_last = True, True

train_loader = DataLoader(train_data, batch_size=int(args.batch_size / nproc), shuffle=shuffle,
                          sampler=train_sampler)
valid_loader = DataLoader(valid_data, batch_size=int(args.batch_size / nproc), shuffle=shuffle,
                          sampler=valid_sampler, drop_last=drop_last)
test_loader = DataLoader(test_data, batch_size=int(args.batch_size / nproc), shuffle=shuffle,
                         sampler=test_sampler, drop_last=drop_last)

print('Finish loading the data....')
if not args.aligned:
    print("### Note: You are running in unaligned mode.")


hyp_params = args
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_data.get_dim()
hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_data.get_seq_len()
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
hyp_params.criterion = criterion_dict.get(dataset, 'L1Loss')

if __name__ == '__main__':
    if hyp_params.test:
        test.test_initiate(hyp_params, train_loader, valid_loader, test_loader)
    else:
        test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader)
