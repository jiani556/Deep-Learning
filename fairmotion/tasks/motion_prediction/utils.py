# Copyright (c) Facebook, Inc. and its affiliates.


import numpy as np
import os
import torch
from functools import partial
from multiprocessing import Pool

from fairmotion.models import (
    decoders,
    encoders,
    optimizer,
    rnn,
    seq2seq,
    transformer,
    sttransformer
)
from fairmotion.tasks.motion_prediction import dataset as motion_dataset
from fairmotion.utils import constants
from fairmotion.ops import conversions

from fairmotion.data import amass_dip
from fairmotion.utils import utils as fairmotion_utils
import pickle
import gc

def apply_ops(input, ops):
    """
    Apply series of operations in order on input. `ops` is a list of methods
    that takes single argument as input (single argument functions, partial
    functions). The methods are called in the same order provided.
    """
    output = input
    for op in ops:
        output = op(output)
    return output


def unflatten_angles(arr, rep):
    """
    Unflatten from (batch_size, num_frames, num_joints*ndim) to
    (batch_size, num_frames, num_joints, ndim) for each angle format
    """
    if rep == "aa":
        return arr.reshape(arr.shape[:-1] + (-1, 3))
    elif rep == "quat":
        return arr.reshape(arr.shape[:-1] + (-1, 4))
    elif rep == "rotmat":
        return arr.reshape(arr.shape[:-1] + (-1, 3, 3))


def flatten_angles(arr, rep):
    """
    Unflatten from (batch_size, num_frames, num_joints, ndim) to
    (batch_size, num_frames, num_joints*ndim) for each angle format
    """
    if rep == "aa":
        return arr.reshape(arr.shape[:-2] + (-1))
    elif rep == "quat":
        return arr.reshape(arr.shape[:-2] + (-1))
    elif rep == "rotmat":
        # original dimension is (batch_size, num_frames, num_joints, 3, 3)
        return arr.reshape(arr.shape[:-3] + (-1))


def multiprocess_convert(arr, convert_fn):
    pool = Pool(40)
    result = list(pool.map(convert_fn, arr))
    return result


def convert_fn_to_R(rep):
    ops = [partial(unflatten_angles, rep=rep)]
    if rep == "aa":
        ops.append(partial(multiprocess_convert, convert_fn=conversions.A2R))
    elif rep == "quat":
        ops.append(partial(multiprocess_convert, convert_fn=conversions.Q2R))
    elif rep == "rotmat":
        ops.append(lambda x: x)
    ops.append(np.array)
    return ops


def identity(x):
    return x


def convert_fn_from_R(rep):
    if rep == "aa":
        convert_fn = conversions.R2A
    elif rep == "quat":
        convert_fn = conversions.R2Q
    elif rep == "rotmat":
        convert_fn = identity
    return convert_fn


def unnormalize(arr, mean, std):
    return arr * (std + constants.EPSILON) + mean


def prepare_dataset(
    train_path, valid_path, test_path, batch_size, device, shuffle=False,
):
    dataset = {}
    for split, split_path in zip(
        ["train", "test", "validation"], [train_path, valid_path, test_path]
    ):
        mean, std = None, None
        if split in ["test", "validation"]:
            mean = dataset["train"].dataset.mean
            std = dataset["train"].dataset.std
        dataset[split] = motion_dataset.get_loader(
            split_path, batch_size, device, mean, std, shuffle,
        )
    return dataset, mean, std


def prepare_model(
    input_dim, hidden_dim, device, num_layers=1, architecture="seq2seq"
):
    if architecture == "rnn":
        model = rnn.RNN(input_dim, hidden_dim, num_layers)
    if architecture == "seq2seq":
        enc = encoders.LSTMEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim
        ).to(device)
        dec = decoders.LSTMDecoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            device=device,
        ).to(device)
        model = seq2seq.Seq2Seq(enc, dec)
    elif architecture == "tied_seq2seq":
        model = seq2seq.TiedSeq2Seq(input_dim, hidden_dim, num_layers, device)
    elif architecture == "transformer_encoder":
        model = transformer.TransformerLSTMModel(
            input_dim, hidden_dim, 4, hidden_dim, num_layers,
        )
    elif architecture == "transformer":
        model = transformer.TransformerModel(
            input_dim, hidden_dim, 4, hidden_dim, num_layers,
        )
    elif architecture == "sttransformer":
        num_joints = amass_dip.SMPL_NR_JOINTS
        model = sttransformer.STTransformerModel(
            num_joints = num_joints,
            rep_size = input_dim // num_joints,
            num_heads_t=4, 
            num_heads_s=4, 
            hidden_dim = hidden_dim, 
            num_layers = num_layers,
            dropout=0.1,
            use_torchMHA = False
        )
    model = model.to(device)
    model.zero_grad()
    return model

def load_model(model_init, saved_states):
    """
    model_init: initialized model, parameters have not been updated
    saved_states: previously saved model and optimizer state
    """
    model_init.load_state_dict(saved_states['model_state_dict'])
    model_init.to(next(model_init.parameters()).device)

def log_config(path, args):
    with open(os.path.join(path, "config.txt"), "w") as f:
        for key, value in args._get_kwargs():
            f.write(f"{key}:{value}\n")


def prepare_optimizer(model, opt: str, lr=None):
    kwargs = {}
    if lr is not None:
        kwargs["lr"] = lr

    if opt == "sgd":
        return optimizer.SGDOpt(model, **kwargs)
    elif opt == "adam":
        return optimizer.AdamOpt(model, **kwargs)
    elif opt == "noamopt":
        return optimizer.NoamOpt(model)

def load_optimizer(opt_init, saved_states):
    """
    opt_init: initialized optimizer, parameters have not been updated
    saved_states: previously saved model and optimizer state
    """
    if isinstance(opt_init, optimizer.NoamOpt):
        opt_init.optimizer.load_state_dict(saved_states['optim_state_dict'])
        opt_init._step = (saved_states['epoch'] + 1) * saved_states['iterations']
    else:
        raise NotImplementedError

def prepare_tgt_seqs(architecture, src_seqs, tgt_seqs):
    if architecture == "sttransformer" or architecture == "rnn":
        return torch.cat((src_seqs[:, 1:], tgt_seqs), axis=1)
    else:
        return tgt_seqs

def prepare_mean_and_std(train_path, recalc=False):
    """
    train_path: directory name that contains training pkl files
    recalc: whether to recalculate or use mean and std from local cached file
    
    calculate mean and std for streaming input to handle
    the case where data cannot be loaded into RAM in one go
    data comes as (N1, mean1, var1), (N2, mean2, var2)...
    mean = mean1 + N2 * (mean2 - mean1)/(N1 + N2)
    var = N1 * var1 / (N1 + N2) + N2 * var2 / (N1 + N2) + (N1 * N2) * (mean1 - mean2)^2/(N1 + N2)^2 

    """
    if train_path[-1] != '/':
        train_path += '/'
    mean_and_std_filepath = os.path.join(train_path, 'mean_and_std.pkl')
    # recalculate
    if recalc:
        stats = {}
        for filepath in fairmotion_utils.files_in_dir(train_path, ext="pkl", keyword="train_"):
            file_id = os.path.basename(filepath)
            print('Load {}'.format(file_id))
            with open(filepath, "rb") as f:
                src_seqs = np.array(pickle.load(f)[0])
            f = None
            stats[file_id] = {}
            stats[file_id]["mean"] = np.mean(src_seqs, axis=(0, 1))
            stats[file_id]["var"] = np.var(src_seqs, axis=(0, 1))
            stats[file_id]["len"] = src_seqs.shape[0] * src_seqs.shape[1]
            src_seqs = None
            gc.collect()
                
        mean, var, total_len = 0, 0, 0
        for _, stat in stats.items():
            new_total_len = total_len + stat["len"]
            new_mean = mean + stat["len"] * (stat["mean"] - mean) / new_total_len
            new_var = (total_len * var + stat["len"] * stat["var"]) / new_total_len\
                + total_len * stat["len"] * (mean - stat["mean"])**2 / new_total_len**2
            mean, var, total_len = new_mean, new_var, new_total_len
        mean, std = mean, np.sqrt(var)
        #save for later uses
        with open(mean_and_std_filepath, 'wb') as f:
            pickle.dump((mean, std), f)
    else:
        #load from cached pkl file
        with open(mean_and_std_filepath, 'rb') as f:
            mean, std = pickle.load(f)
    
    return mean, std
        
def prepare_dataloader_with_mean_std(
    path, batch_size, device, shuffle=False, mean=None, std=None
):
    """
    similar to prepare_dataset, except that this takes precomputed
    training mean and std

    """
    dataloader = motion_dataset.get_loader(
        path, batch_size, device, mean, std, shuffle,
    )
    return dataloader
     