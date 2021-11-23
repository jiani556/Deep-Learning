# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn

from fairmotion.tasks.motion_prediction import generate, utils, test
from fairmotion.utils import utils as fairmotion_utils

import gc

logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def set_seeds():
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    fairmotion_utils.create_dir_if_absent(args.save_model_path)
    logging.info(args._get_kwargs())
    utils.log_config(args.save_model_path, args)

    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device if args.device else device
    logging.info(f"Using device: {device}")

    logging.info("Preparing dataset...")
    
    _, rep = os.path.split(args.preprocessed_path.strip("/"))
    assert rep in ["aa", "rotmat", "quat"]
    
    # precompute training data mean and std
    logging.info("Precompute training data mean and std...")
    try:
        mean, std = utils.prepare_mean_and_std(args.preprocessed_path, recalc=False)
        logging.info("Using cached mean and std")
    except:
        logging.info("Recalculate training data mean and std...")
        mean, std = utils.prepare_mean_and_std(args.preprocessed_path, recalc=True)
    
    # first process validation and test data
    dataset = {}
    for split in ["validation", "test"]:
        logging.info(f"Preparing {split} dataset...")
        dataset[split] = utils.prepare_dataloader_with_mean_std(
            os.path.join(args.preprocessed_path, f"{split}.pkl"),
            args.batch_size,
            device,
            args.shuffle,
            mean,
            std
        )
    
    # get all training pkl file names
    train_pkl_files = fairmotion_utils.files_in_dir(args.preprocessed_path, ext="pkl", keyword="train_")

    # number of predictions per time step = num_joints * angle representation
    # shape is (batch_size, seq_len, num_predictions)
    _, tgt_len, num_predictions = next(iter(dataset["test"]))[1].shape

    logging.info("Preparing model...")
    model = utils.prepare_model(
        input_dim=num_predictions,
        hidden_dim=args.hidden_dim,
        device=device,
        num_layers=args.num_layers,
        architecture=args.architecture,
    )
    
    criterion = nn.MSELoss()
    model.init_weights()
    training_losses, val_losses = [], []
    
    # fresh start
    if not args.resume:
        epoch_start = 0
    
        # load training data in batch
        # epoch_loss = 0
        # iterations = 0
        # model.eval()
        # with torch.no_grad():
        #     for filepath in train_pkl_files:
        #         logging.info(f"Pretraining evaluation with {filepath}")
        #         dataloader = utils.prepare_dataloader_with_mean_std(
        #             filepath,
        #             args.batch_size,
        #             device,
        #             args.shuffle,
        #             mean,
        #             std
        #         )
        #         for _, (src_seqs, tgt_seqs) in enumerate(dataloader):
        #             if iterations % 10 == 0:
        #                 logging.info(f"Iteration {iterations}")
        #             outputs = model(src_seqs, tgt_seqs, teacher_forcing_ratio=1,)
        #             loss = criterion(outputs, tgt_seqs)
        #             epoch_loss += loss.item()
        #             iterations += 1
        #         dataloader = None
        #         gc.collect()
            
        # epoch_loss = epoch_loss / (iterations * args.batch_size)
        # val_loss = generate.eval(
        #     model, criterion, dataset["validation"], args.batch_size, device,
        # )
        # logging.info(
        #     "Before training: "
        #     f"Training loss {epoch_loss} | "
        #     f"Validation loss {val_loss}"
        # )
    
        # specify optimizer
        opt = utils.prepare_optimizer(model, args.optimizer, args.lr)
    else:   # resume training from prev checkpoint
        saved_model_files = fairmotion_utils.files_in_dir(args.save_model_path, ext="model")
        if not len(saved_model_files):
            raise FileNotFoundError('No saved model found!')
        
        # find the file with latest modified timestamp
        saved_model_files.sort(key=os.path.getmtime)
        logging.info(f"Loading previously saved model from {saved_model_files[-1]}...")
        saved_states = torch.load(saved_model_files[-1])
        
        # load states
        epoch_start = saved_states['epoch'] + 1
        training_losses, val_losses = saved_states['training_losses'], saved_states['val_losses']
        logging.info(f'Last training epoch {epoch_start - 1}')
        
        # load model state dict
        utils.load_model(model, saved_states)
        logging.info('Model loaded')
        
        # load optimizer state dict
        opt = utils.prepare_optimizer(model, args.optimizer, args.lr)
        utils.load_optimizer(opt, saved_states)
        logging.info('Optimizer loaded')

    logging.info("Training model...")
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(epoch_start, args.epochs):
        epoch_loss = 0
        model.train()
        teacher_forcing_ratio = np.clip(
            (1 - 2 * epoch / args.epochs), a_min=0, a_max=1,
        )
        logging.info(
            f"Running epoch {epoch} | "
            f"teacher_forcing_ratio={teacher_forcing_ratio}"
        )
        
        # load training data in batch
        iterations = 0
        for filepath in train_pkl_files:
            logging.info(f"Training with {filepath}")
            dataloader = utils.prepare_dataloader_with_mean_std(
                filepath,
                args.batch_size,
                device,
                args.shuffle,
                mean,
                std
            )
            for _, (src_seqs, tgt_seqs) in enumerate(dataloader):
                if iterations % 10 == 0:
                    logging.info(f"Iteration {iterations}")
                opt.optimizer.zero_grad()
                outputs = model(
                    src_seqs, tgt_seqs, teacher_forcing_ratio=teacher_forcing_ratio
                )
                loss = criterion(
                    outputs,
                    utils.prepare_tgt_seqs(args.architecture, src_seqs, tgt_seqs),
                )
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                iterations += 1
            dataloader = None
            gc.collect()
        epoch_loss = epoch_loss / (iterations * args.batch_size)
        training_losses.append(epoch_loss)
        val_loss = generate.eval(
            model, criterion, dataset["validation"], args.batch_size, device,
        )
        val_losses.append(val_loss)
        opt.epoch_step(val_loss=val_loss)
        logging.info(
            f"Training loss {epoch_loss} | "
            f"Validation loss {val_loss} | "
            f"Iterations {iterations}"
        )
        if epoch % args.save_model_frequency == 0:
            if epoch == args.epochs - 1:
                _, mae = test.test_model(
                    model=model,
                    dataset=dataset["validation"],
                    rep=rep,
                    device=device,
                    mean=mean,
                    std=std,
                    max_len=tgt_len,
                )
                logging.info(f"Validation MAE: {mae}")
            states_to_save = {
                'epoch': epoch,
                'iterations': iterations,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': opt.optimizer.state_dict(),
                'training_losses': training_losses,
                'val_losses': val_losses
            }
            torch.save(
                states_to_save, f"{args.save_model_path}/{epoch}.model"
            )
            if len(val_losses) == 0 or val_loss <= min(val_losses):
                torch.save(
                    states_to_save, f"{args.save_model_path}/best.model"
                )
    return training_losses, val_losses


def plot_curves(args, training_losses, val_losses):
    plt.plot(range(len(training_losses)), training_losses)
    plt.plot(range(len(val_losses)), val_losses)
    plt.ylabel("MSE Loss")
    plt.xlabel("Epoch")
    plt.savefig(f"{args.save_model_path}/loss.svg", format="svg")


def main(args):
    train_losses, val_losses = train(args)
    plot_curves(args, train_losses, val_losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequence to sequence motion prediction training"
    )
    parser.add_argument(
        "--preprocessed-path",
        type=str,
        help="Path to folder with pickled files",
        required=True,
    )
    parser.add_argument(
        "--batch-size", type=int, help="Batch size for training", default=16
    )
    parser.add_argument(
        "--shuffle", action='store_false',
        help="Use this option to enable shuffling",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        help="Hidden size of LSTM units in encoder/decoder",
        default=64,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help="Number of layers of LSTM/Transformer in encoder/decoder",
        default=4,
    )
    parser.add_argument(
        "--save-model-path",
        type=str,
        help="Path to store saved models",
        required=True,
    )
    parser.add_argument(
        "--save-model-frequency",
        type=int,
        help="Frequency (in terms of number of epochs) at which model is "
        "saved",
        default=1,
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of training epochs", default=10
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Training device",
        default=None,
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--architecture",
        type=str,
        help="Seq2Seq archtiecture to be used",
        default="sttransformer",
        choices=[
            "seq2seq",
            "tied_seq2seq",
            "transformer",
            "transformer_encoder",
            "rnn",
            "sttransformer",
        ],
    )
    parser.add_argument(
        "--lr", type=float, help="Learning rate", default=None,
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        help="Torch optimizer",
        default="noamopt",
        choices=["adam", "sgd", "noamopt"],
    )
    parser.add_argument(
        "--resume", action='store_true',
        help="Use this option to resume training from prev checkpoint",
    )
    args = parser.parse_args()
    main(args)
