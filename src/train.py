from torch import nn
import sys
from src import models
import torch.optim as optim
import datetime
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CyclicLR, CosineAnnealingWarmRestarts, ExponentialLR
from warmup_scheduler import GradualWarmupScheduler

from src.eval_metrics import *
from src.utils import *

from torch.utils.tensorboard import SummaryWriter
import warnings
import gc
# ddp auto sampler
import torch.distributed as dist

warnings.filterwarnings("ignore")


def get_model_size(model):
    para_num = sum([p.numel() for p in model.parameters()])
    return para_num


def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = getattr(models, hyp_params.model + 'Model')(hyp_params)

    if hyp_params.optim in ['SGD', ]:
        optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr, momentum=0.9)
    else:
        optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)

    criterion = getattr(nn, hyp_params.criterion)()

    if hyp_params.schedule in ['CyclicLR', ]:
        scheduler = CyclicLR(optimizer, base_lr=hyp_params.base_lr, max_lr=hyp_params.max_lr,
                             step_size_up=hyp_params.step_up)
    elif hyp_params.schedule in ['warmup', ]:
        # scheduler_warmup is chained with scheduler_stepper
        if hyp_params.lr_stepper in ['steplr', ]:
            scheduler_stepper = StepLR(optimizer, step_size=hyp_params.stepper_size, gamma=0.1)
        else:
            scheduler_stepper = ExponentialLR(optimizer, gamma=0.1)
        # base_lrs is defined as maximux warmup learning rate defined in optim: hyper_params.lr
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=hyp_params.warm_epoch,
                                           after_scheduler=scheduler_stepper)
    elif hyp_params.schedule in ['calr', ]:
        # CosineAnnealingLR: T_0 (int): Number of iterations for the first restart.
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)

    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


# Training and evaluation
def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    # get model type
    model = settings['model']

    if hyp_params.pretrain is not None:
        print('loading pretrained model: {}...'.format(hyp_params.pretrain))
        model = load_model(hyp_params, name=hyp_params.pretrain)
        print('Done loading..')

    # basic criterion:
    criterion = settings['criterion']

    # learning scheduler and optimizer
    scheduler = settings['scheduler']
    optimizer = settings['optimizer']

    # regularization on/off
    reg_en = hyp_params.reg_en
    print('reg_en:', reg_en)

    local_rank = 0
    if hyp_params.use_cuda:
        if torch.cuda.device_count() > 1:
            print('using ', torch.cuda.device_count(), ' gpus!')

            if hyp_params.ddp:
                local_rank = dist.get_rank()
                torch.cuda.set_device(local_rank)
                # device = torch.device("cuda", local_rank)
                model = model.cuda(local_rank)
                criterion = criterion.cuda(local_rank)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                                  output_device=local_rank,
                                                                  find_unused_parameters=True)
            else:
                model = nn.DataParallel(model.cuda())
                criterion = criterion.cuda()
        else:
            model = model.cuda()

    best_valid = 1e8
    best_acc_or_f1 = -1.0
    train_loss_ls = val_loss_ls = test_loss_ls = []

    log_dir = os.path.join('tensorboard_log', hyp_params.name, 'train')
    train_writer = SummaryWriter(log_dir=log_dir)
    log_dir = os.path.join('tensorboard_log', hyp_params.name, 'val')
    val_writer = SummaryWriter(log_dir=log_dir)
    log_dir = os.path.join('tensorboard_log', hyp_params.name, 'test')
    test_writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(1, hyp_params.num_epochs + 1):
        time1 = datetime.datetime.now()

        if epoch == 1:
            print('size of the model is {}'.format(get_model_size(model)))  # trimodal: 6,738,827

        start = time.time()
        train_loss = train_epoch(epoch, hyp_params, train_loader, model, optimizer, criterion, local_rank, reg_en)

        val_loss, _, _ = evaluate(hyp_params, model, criterion, local_rank, valid_loader, False)
        test_loss, _, _ = evaluate(hyp_params, model, criterion, local_rank, test_loader, True)

        time2 = datetime.datetime.now()
        time_diff = (time2 - time1).seconds
        print('training the whole model need :{}'.format(time_diff))

        end = time.time()
        duration = end - start
        scheduler.step(val_loss)

        if local_rank == 0:
            train_writer.add_scalar('loss', train_loss, epoch)
            val_writer.add_scalar('loss', val_loss, epoch)
            test_writer.add_scalar('loss', test_loss, epoch)

            train_loss_ls.append(train_loss)
            val_loss_ls.append(val_loss)
            test_loss_ls.append(test_loss)

            print(f'E{epoch:2d} | Time {duration:5.4f} sec | Valid {val_loss:5.4f} | Test {test_loss:5.4f}')
            # for each epoch test the best model on test set and save
            print(f'evaluating in epoch: {epoch}#####')
            # evaluating accuracy on training set, validation set and test set, respectively
            _, train_results, train_truths = evaluate(hyp_params, model, criterion, local_rank, train_loader, False)
            _, val_results, val_truths = evaluate(hyp_params, model, criterion, local_rank, valid_loader, False)
            _, results, truths = evaluate(hyp_params, model, criterion, local_rank, test_loader, True)

            if hyp_params.dataset in ["mosei_senti", "mosei"]:
                if hyp_params.train_mode is 'regression':
                    print('using new metric')
                    train_mosei_res = eval_mosei_regression(train_results, train_truths)
                    val_mosei_res = eval_mosei_regression(val_results, val_truths)
                    test_mosei_res = eval_mosei_regression(results, truths)

                    if test_mosei_res['Has0_F1_score'] > best_acc_or_f1:
                        print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
                        save_model(hyp_params, model, name=hyp_params.name)
                        best_acc_or_f1 = test_mosei_res['Has0_F1_score']
                else:
                    train_avg_f1, train_avg_acc, t_mae, t_corr = eval_mosei_senti(train_results, train_truths,
                                                                                  True)  # trainset
                    val_avg_f1, val_avg_acc, v_mae, v_corr = eval_mosei_senti(val_results, val_truths, True)  # valset
                    avg_f1, avg_acc, mae, corr = eval_mosei_senti(results, truths, True)  # testset

                    train_writer.add_scalar('accuracy', train_avg_acc, epoch)
                    val_writer.add_scalar('accuracy', val_avg_acc, epoch)
                    test_writer.add_scalar('accuracy', avg_acc, epoch)

                    train_writer.add_scalar('mae', t_mae, epoch)
                    val_writer.add_scalar('mae', v_mae, epoch)
                    test_writer.add_scalar('mae', mae, epoch)

                    train_writer.add_scalar('corr', t_corr, epoch)
                    val_writer.add_scalar('corr', v_corr, epoch)
                    test_writer.add_scalar('corr', corr, epoch)

                    if avg_f1 > best_acc_or_f1:
                        print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
                        save_model(hyp_params, model, name=hyp_params.name)
                        best_acc_or_f1 = avg_f1

            elif hyp_params.dataset in ['mosi', 'sims', ]:
                if hyp_params.train_mode is 'regression':
                    print('using new metric')
                    train_mosi_res = eval_mosei_regression(train_results, train_truths)
                    val_mosi_res = eval_mosei_regression(val_results, val_truths)
                    test_mosi_res = eval_mosei_regression(results, truths)

                    if test_mosi_res['Has0_F1_score'] > best_acc_or_f1:
                        print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
                        save_model(hyp_params, model, name=hyp_params.name)
                        best_acc_or_f1 = test_mosi_res['Has0_F1_score']
                else:
                    train_avg_f1, train_avg_acc, t_mae, t_corr = eval_mosi(train_results, train_truths, True)
                    val_avg_f1, val_avg_acc, v_mae, v_corr = eval_mosi(val_results, val_truths, True)
                    avg_f1, avg_acc, mae, corr = eval_mosi(results, truths, True)

                    train_writer.add_scalar('accuracy', train_avg_acc, epoch)
                    val_writer.add_scalar('accuracy', val_avg_acc, epoch)
                    test_writer.add_scalar('accuracy', avg_acc, epoch)

                    train_writer.add_scalar('mae', t_mae, epoch)
                    val_writer.add_scalar('mae', v_mae, epoch)
                    test_writer.add_scalar('mae', mae, epoch)

                    train_writer.add_scalar('corr', t_corr, epoch)
                    val_writer.add_scalar('corr', v_corr, epoch)
                    test_writer.add_scalar('corr', corr, epoch)

                    if avg_f1 > best_acc_or_f1:
                        print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
                        save_model(hyp_params, model, name=hyp_params.name)
                        best_acc_or_f1 = avg_f1
        torch.cuda.empty_cache()

    model = load_model(hyp_params, name=hyp_params.name)
    _, results, truths = evaluate(hyp_params, model, criterion, local_rank, test_loader, True)

    if hyp_params.dataset in ["mosei_senti", "mosei"]:
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == 'mosi':
        eval_mosi(results, truths, True)

    sys.stdout.flush()
    input('[Finished! Press enter ...]')

def train_epoch(epoch, hyp_params, train_loader, model, optimizer, criterion, local_rank, reg_en=False):

    epoch_loss = 0
    model.train()
    num_batches = hyp_params.n_train // hyp_params.batch_size
    proc_loss, proc_size = 0, 0
    start_time = time.time()

    for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
        sample_ind, text, audio, vision = batch_X
        eval_attr = batch_Y.squeeze(-1)  # if num of labels is 1

        model.zero_grad()
        optimizer.zero_grad()

        if hyp_params.use_cuda:
            # with torch.cuda.device(0):
            if hyp_params.ddp:
                text, audio, vision, eval_attr = text.cuda(local_rank), audio.cuda(local_rank), vision.cuda(
                    local_rank), eval_attr.cuda(local_rank)
            else:
                text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()

        batch_size = text.size(0)

        combined_loss = 0
        preds, hiddens, h_list = model(text, audio, vision)

        raw_loss = criterion(preds, eval_attr)

        if reg_en:
            # Sinkhorn Distances
            # reg_term = torch.mul(h_list[-1], 0.1)
            reg_term = h_list[-1]
            combined_loss += hyp_params.reg_lambda * torch.mean(reg_term)
        else:
            combined_loss = raw_loss

        combined_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
        optimizer.step()

        proc_loss += raw_loss.item() * batch_size
        proc_size += batch_size
        epoch_loss += combined_loss.item() * batch_size
        if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
            avg_loss = proc_loss / proc_size
            elapsed_time = time.time() - start_time
            print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                  format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss))
            proc_loss, proc_size = 0, 0
            start_time = time.time()

        del text, audio, vision, eval_attr, preds, hiddens, h_list
        gc.collect()
        torch.cuda.empty_cache()

    return epoch_loss / hyp_params.n_train

def evaluate(hyp_params, model, criterion, local_rank, loader, is_test_data):
    model.eval()
    total_loss = 0.0
    total_cnt = 0

    results = []
    truths = []
    with torch.no_grad():
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
            sample_ind, text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(dim=-1)  # if num of labels is 1

            if hyp_params.use_cuda:
                if hyp_params.ddp:
                    text, audio, vision, eval_attr = text.cuda(local_rank), audio.cuda(local_rank), vision.cuda(
                        local_rank), eval_attr.cuda(local_rank)
                else:
                    text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()

            batch_size = text.size(0)
            preds, last_h_proj, h_list = model(text, audio, vision)
            total_loss += criterion(preds, eval_attr).item() * batch_size
            total_cnt += batch_size

            # add reg on validation set to check loss change
            if not is_test_data and (hyp_params.reg_en is True):
                reg_term = h_list[-1]
                total_loss += hyp_params.reg_lambda * torch.mean(reg_term)
            # Collect the results into dictionary
            results.append(preds)
            truths.append(eval_attr)

    avg_loss = total_loss / total_cnt

    results = torch.cat(results)
    truths = torch.cat(truths)

    del text, audio, vision, eval_attr, preds, last_h_proj, h_list, total_loss
    gc.collect()
    torch.cuda.empty_cache()

    return avg_loss, results, truths
