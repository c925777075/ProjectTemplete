import os
import sys
import logging
import pathlib
import warnings

project_path = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(project_path))
import wandb
from transformers.integrations import WandbCallback
from transformers import Trainer
from src.engine.trainer import CustomTrainer
from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
from src.utils.misc import patch_cosine_with_warmup_schedule
from src.utils import patch_transformer_logging
from src.config import prepare_args
from src.models import load_model
from src.dataset import load_dataset


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging_format = "[%(levelname)s] %(pathname)s:%(lineno)d: %(message)s"
logging.basicConfig(
    format=logging_format,
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)

def entry(cfg, training_args):
    model = load_model(cfg.model_args, training_args)
    train_dataset = load_dataset(cfg.data_args.train, training_args)
    valid_dataset = load_dataset(cfg.data_args.train, training_args)
    test_dataset = load_dataset(cfg.data_args.test, training_args)

    # 如果需要评估metric，自己加进去就好了，或者自定义一个Trainer去继承
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=train_dataset.collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[WandbCallback()]
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)  # noqa
        trainer.save_metrics("train", train_result.metrics)  # noqa

        model_save_dir = os.path.join(trainer.args.output_dir, "model_save")
        trainer.save_model(output_dir=model_save_dir)
        trainer.save_state()  # noqa

    # save cfg to output_dir
    try:
        output_dir = training_args.output_dir
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        cfg.dump(os.path.join(output_dir, "cfg.py"))
    except Exception as e:
        warnings.warn(f'try to save cfg to output_dir, but get exception {e.args}')

    # Evaluation
    if training_args.do_eval:
        if hasattr(trainer, '_test_collator') and hasattr(trainer, '_eval_collator') \
                and trainer._test_collator != trainer._eval_collator:  # noqa
            warnings.warn('[WARNING!!!] use different collator for eval and test. but do_eval and '
                          'do_predict both use trainer.predict (i.e. only test_collator is used.)')
        eval_results = trainer.predict(valid_dataset, metric_key_prefix="eval")
        trainer.log_metrics("eval", eval_results.metrics)  # noqa
        trainer.save_metrics("eval", eval_results.metrics)  # noqa

    # Predict
    if training_args.do_predict and test_dataset is not None:
        predict_results = trainer.predict(test_dataset, metric_key_prefix="test")
        trainer.log_metrics("test", predict_results.metrics)  # noqa
        trainer.save_metrics("test", predict_results.metrics)  # noqa


def main():
    patch_transformer_logging()
    cfg, training_args = prepare_args()
    patch_cosine_with_warmup_schedule(getattr(training_args, 'minimal_learning_rate', 0.0))
    if os.environ.get("WANDB_DISABLED").lower() == "false":
        wandb.login(key=training_args.wandb_key)
    os.makedirs(training_args.output_dir, exist_ok=True)

    if training_args.process_index == 0 and os.getenv('USE_SACRED', 'false').lower() == 'true':

        # use sacred
        ex = Experiment(training_args.output_dir)
        mongo_url = None
        # do not add mongo.txt into git repo
        if os.path.exists('mongo.txt'):
            with open('mongo.txt', 'r') as fin:
                mongo_url = fin.readline().strip()
        else:
            print('mongo.txt does not exists, use file observer instead')

        if mongo_url is not None:
            ex.observers.append(MongoObserver(mongo_url))
        else:
            ex.observers.append(FileStorageObserver(training_args.output_dir))

        @ex.config
        def train_cfg():
            cfg_for_sacred = None
        def do_train_sacred(cfg_for_sacred):
            entry(cfg, training_args)
        ex.main(do_train_sacred)
        ex.run(config_updates={'cfg_for_sacred': cfg.to_dict()})

    else:
        entry(cfg, training_args)

if __name__ == "__main__":
    main()
