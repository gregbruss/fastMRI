import pathlib
import sys
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything

sys.path.append("../../")  # noqa: E402

from fastmri.data.mri_data import fetch_dir
from unet_module import UnetModule

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

# Main Call with args
def main(args):
    """Main training routine."""
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    seed_everything(args.seed)
    model = UnetModule(**vars(args))

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = Trainer.from_argparse_args(args)

    # ------------------------
    # 3 START TRAINING OR TEST
    # ------------------------

    if args.mode == "train":
        trainer.fit(model)
    elif args.mode == "test":
        assert args.resume_from_checkpoint is not None
        trainer.test(model)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args():
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    path_config = pathlib.Path.cwd() / ".." / ".." / "fastmri_dirs.yaml"
    me_path = fetch_dir("me_path", path_config)
    logdir = fetch_dir("log_path", path_config) / "unet" / "multi-echo-oct10"

    # Parsing Arguments
    parent_parser = ArgumentParser(add_help=False)
    parser = UnetModule.add_model_specific_args(parent_parser)
    parser = Trainer.add_argparse_args(parser)

    num_gpus = 1
    backend = "ddp"
    batch_size = 1 if backend == "ddp" else num_gpus

    # module config
    config = dict(
        in_chans=4,
        out_chans=4,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0,
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
        data_path=me_path,
        challenge="multiecho",
        exp_dir=logdir,
        exp_name="unet_demo",
        test_split="test",
        batch_size=batch_size,
    )
    parser.set_defaults(**config)

    # trainer config
    parser.set_defaults(
        gpus=num_gpus,
        default_root_dir=logdir,
        replace_sampler_ddp=(backend != "ddp"),
        distributed_backend=backend,
        seed=42,
        deterministic=True,
        weights_summary='full',
        fast_dev_run = True
    )

    parser.add_argument("--mode", default="train", type=str)
    args = parser.parse_args()

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(args)


if __name__ == "__main__":
    run_cli()
