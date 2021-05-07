import torch
from numpy import inf
from torchvision.utils import make_grid

def train_epoch(model, dataLoader, device, optimizer, lr_scheduler, logger, train_metrics, criterion, epoch, writer, metric_ftns):
    model.train()
    train_metrics.reset()
    len_epoch = len(dataLoader)
    log_step = 10 #int(len(dataLoader)//10 + 1)

    def _progress(batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(dataLoader, 'n_samples'):
            current = batch_idx * dataLoader.batch_size
            total = dataLoader.n_samples
        else:
            current = batch_idx
            total = len_epoch
        return base.format(current, total, 100.0 * current / total)

    for batch_idx, (data, target) in enumerate(dataLoader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        writer.set_step((epoch - 1) * len_epoch + batch_idx)
        train_metrics.update('loss', loss.item())
        for met in metric_ftns:
            train_metrics.update(met.__name__, met(output, target))

        if batch_idx % log_step == 0:
            logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                epoch,
                _progress(batch_idx),
                loss.item()))
            writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        if batch_idx == len_epoch:
            break
    log = train_metrics.result()

    if lr_scheduler is not None:
        lr_scheduler.step()
    return log


'''
def train_epoch(model, criterion, metric_ftns, optimizer, config, epoch):
    logger = config.get_logger('trainer', config['trainer']['verbosity'])

    cfg_trainer = config['trainer']
    epochs = cfg_trainer['epochs']
    save_period = cfg_trainer['save_period']
    monitor = cfg_trainer.get('monitor', 'off')

    # configuration to monitor model performance and save best
    if monitor == 'off':
        mnt_mode = 'off'
        mnt_best = 0
    else:
        mnt_mode, mnt_metric = monitor.split()
        assert mnt_mode in ['min', 'max']

        mnt_best = inf if mnt_mode == 'min' else -inf
        early_stop = cfg_trainer.get('early_stop', inf)
        if early_stop <= 0:
            early_stop = inf

    checkpoint_dir = config.save_dir

    # setup visualization writer instance                
    writer = TensorboardWriter(config.log_dir, logger, cfg_trainer['tensorboard'])

    if config.resume is not None:
        _resume_checkpoint(config.resume)

    not_improved_count = 0
    for epoch in range(start_epoch, epochs + 1):
        result = _train_epoch(epoch)

    # save logged informations into log dict
    log = {'epoch': epoch}
    log.update(result)

    # print logged informations to the screen
    for key, value in log.items():
        logger.info('    {:15s}: {}'.format(str(key), value))

    # evaluate model performance according to configured metric, save best checkpoint as model_best
    best = False
    if mnt_mode != 'off':
        try:
            # check whether model performance improved or not, according to specified metric(mnt_metric)
            improved = (mnt_mode == 'min' and log[mnt_metric] <= mnt_best) or \
                        (mnt_mode == 'max' and log[mnt_metric] >= mnt_best)
        except KeyError:
            logger.warning("Warning: Metric '{}' is not found. "
                                "Model performance monitoring is disabled.".format(mnt_metric))
            mnt_mode = 'off'
            improved = False

        if improved:
            mnt_best = log[mnt_metric]
            not_improved_count = 0
            best = True
        else:
            not_improved_count += 1

        if not_improved_count > early_stop:
            logger.info("Validation performance didn\'t improve for {} epochs. "
                                "Training stops.".format(early_stop))
            break

    if epoch % save_period == 0:
        _save_checkpoint(model, epoch, optimizer, checkpoint_dir, logger, save_best=best)

def _save_checkpoint(model, epoch, optimizer, checkpoint_dir, logger, save_best=False):
    """
    Saving checkpoints

    :param epoch: current epoch number
    :param log: logging information of the epoch
    :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
    """
    arch = type(model).__name__
    state = {
        'arch': arch,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'monitor_best': mnt_best,
        'config': config
    }
    filename = str(checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
    torch.save(state, filename)
    logger.info("Saving checkpoint: {} ...".format(filename))
    if save_best:
        best_path = str(checkpoint_dir / 'model_best.pth')
        torch.save(state, best_path)
        logger.info("Saving current best: model_best.pth ...")

def _resume_checkpoint(logger, model, config, optimizer, resume_path):
    """
    Resume from saved checkpoints

    :param resume_path: Checkpoint path to be resumed
    """
    resume_path = str(resume_path)
    logger.info("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(resume_path)
    start_epoch = checkpoint['epoch'] + 1
    mnt_best = checkpoint['monitor_best']

    # load architecture params from checkpoint.
    if checkpoint['config']['arch'] != config['arch']:
        logger.warning("Warning: Architecture configuration given in config file is different from that of "
                            "checkpoint. This may yield an exception while state_dict is being loaded.")
    model.load_state_dict(checkpoint['state_dict'])

    # load optimizer state from checkpoint only when optimizer type is not changed.
    if checkpoint['config']['optimizer']['type'] != config['optimizer']['type']:
        logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                            "Optimizer parameters not being resumed.")
    else:
        optimizer.load_state_dict(checkpoint['optimizer'])

    logger.info("Checkpoint loaded. Resume training from epoch {}".format(start_epoch))
'''