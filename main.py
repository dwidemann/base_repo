
import argparse
from parse_config import ConfigParser
import collections
import pprint
from datasets import dataset_classes 
import models.model as model_classes
from utils import prepare_device
from utils import losses
from utils import metrics as Metrics
from utils import inf_loop, MetricTracker
import torch
from logger import TensorboardWriter
from utils.train_eval_fcns import train_epoch, validation_epoch
from copy import deepcopy

def main(config):
    logger = config.get_logger('trainer', config['trainer']['verbosity'])
    # print logged informations to the screen
    pprint.pprint(config.config)

    trainDataLoader = config.init_obj('data_loader', dataset_classes)
    X,y = next(iter(trainDataLoader))
    print(X.shape)

    config_test = deepcopy(config)
    config_test.config['data_loader']['training'] = False
    valDataLoader = config_test.init_obj('data_loader', dataset_classes)

    model = config.init_obj('arch', model_classes)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    logger.info(device)
    logger.info(device_ids)

    # get function handles of loss and metrics
    criterion = getattr(losses, config['loss'])
    logger.info(criterion)
    metrics = [getattr(Metrics, met) for met in config['metrics']]
    logger.info(metrics)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    logger.info(optimizer)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    logger.info(lr_scheduler)

    cfg_trainer = config['trainer']
    train_writer = TensorboardWriter(config.log_dir, logger, config['visualization']['tensorboardX'])
    train_metrics = MetricTracker('loss', *[m.__name__ for m in metrics], writer=train_writer)

    val_writer = TensorboardWriter(config.log_dir, logger, config['visualization']['tensorboardX'])
    val_metrics = MetricTracker('loss', *[m.__name__ for m in metrics], writer=train_writer)

    for idx in range(1,cfg_trainer['epochs']+1):
        log = train_epoch(model, trainDataLoader, device, optimizer, lr_scheduler, logger, 
                        train_metrics, criterion, idx, train_writer, metrics)
        logger.info(log)

        val_log = validation_epoch(model, valDataLoader, device, logger, val_metrics,
                                criterion, idx, val_writer, metrics)
        logger.info(val_log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)