from torch import nn
from src.dataset.root import DATASETS

# TODO: Registry
def load_dataset(dataset_args, training_args):
    if dataset_args is not None:
        type_ = dataset_args.type
        if type_ in ['COCODateset', 'VideoDatasets', 'VideoDatasetsCondition']:
            dataset = DATASETS.build(dataset_args)
            return dataset
        else:
            assert False
    else:
        return None