from torch import nn
from src.models.root import MODELS

# TODO: Registry
def load_model(model_args, training_args) -> nn.Module:
    type_ = model_args.type
    if type_ in ['VQVAEModel', 'VideoLLM']:
        model = MODELS.build(model_args)
        return model
    else:
        assert False