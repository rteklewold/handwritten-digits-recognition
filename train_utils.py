import os
import torch
import tqdm
from tensorboardX import SummaryWriter


def train_1epoch(
        model: torch.nn.Module,
        model_func,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        dataloader_iter,
        accumulated_iter: int,
        tbar,  # top bar: to track stats across epochs
        total_it_each_epoch: int,
        lr_scheduler=None,
        leave_pbar: bool = False,
        tb_log: SummaryWriter = None
):
    pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train',
                     dynamic_ncols=True)  # progress bar: to track stats across iterations of an epoch
    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)

        # adjust learning rate & log its value
        if lr_scheduler is not None:
            lr_scheduler.step(accumulated_iter)
        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        # pre-training
        model.train()
        optimizer.zero_grad()

        # forward pass
        ret_dict = model_func(model, batch)
        loss = ret_dict['loss']

        # backward pass
        loss.backward()
        optimizer.step()

        # update display stats
        accumulated_iter += 1
        disp_dict = {'loss': loss.item(), 'lr': cur_lr}
        # ---
        pbar.update()
        pbar.set_postfix(dict(total_it=accumulated_iter))
        tbar.set_postfix(disp_dict)
        tbar.refresh()
        if tb_log is not None:
            tb_log.add_scalar('train/loss', loss, accumulated_iter)

    pbar.close()
    return accumulated_iter


def train_model(
        model: torch.nn.Module,
        model_fnc,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        start_epoch: int,
        total_epochs: int,
        start_iter: int,
        ckpt_save_dir: str,
        ckpt_save_interval: int = 1,
        max_ckpt_save_num: int = 50,
        lr_scheduler=None,
        tb_log: SummaryWriter = None,
):
    epoch_start_saving_ckpt = min(1, total_epochs - max_ckpt_save_num)
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=True) as tbar:
        total_it_each_epoch = len(train_loader)
        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            accumulated_iter = train_1epoch(
                model,
                model_fnc,
                optimizer,
                train_loader,
                dataloader_iter,
                accumulated_iter,
                tbar,
                total_it_each_epoch,
                lr_scheduler,
                leave_pbar=False,
                tb_log=tb_log
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and trained_epoch >= epoch_start_saving_ckpt:
                ckpt_name = os.path.join(ckpt_save_dir, f"checkpoint_epoch_{trained_epoch}")
                save_checkpoint(
                    checkpoint_state(model, optimizer, cur_epoch, accumulated_iter), ckpt_name
                )


def checkpoint_state(model, optimizer, epoch, it):
    optim_state = optimizer.state_dict()
    model_state = model.state_dict()
    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimzier_state': optim_state}


def save_checkpoint(state, filename):
    filename = f"{filename}.pth"
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None):
    ckpt = torch.load(filename)
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimzier_state'])
    return ckpt['epoch'], ckpt['it']
