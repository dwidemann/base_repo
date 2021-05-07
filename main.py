
import argparse
from parse_config import ConfigParser
import collections
import pprint
from datasets import dataset_classes 
import models.model as model_classes

def main(config):
    logger = config.get_logger('trainer', config['trainer']['verbosity'])
    # print logged informations to the screen
    pprint.pprint(config.config)

    dataLoader = config.init_obj('data_loader', dataset_classes)
    X,y = next(iter(dataLoader))
    print(X.shape)

    model = config.init_obj('arch', model_classes)
    print(model)

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