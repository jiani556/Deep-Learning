# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import logging
import numpy as np
import os
import torch
from multiprocessing import Pool

from fairmotion.data import amass_dip, bvh
from fairmotion.core import motion as motion_class
from fairmotion.tasks.motion_prediction import generate, metrics, utils
from fairmotion.ops import conversions, motion as motion_ops

from fairmotion.utils import utils as fairmotion_utils

logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def prepare_model(path, num_predictions, args, device):
    model = utils.prepare_model(
        input_dim=num_predictions,
        hidden_dim=args.hidden_dim,
        device=device,
        num_layers=args.num_layers,
        architecture=args.architecture,
    )
    
    logging.info(f"Loading previously saved model from {path}...")
    map_to_cpu = (device == 'cpu')
    if map_to_cpu:
        saved_states = torch.load(path, map_location='cpu')
    else:
        saved_states = torch.load(path)
    utils.load_model(model, saved_states)
    logging.info('Model loaded')
    model.eval()
    return model


def run_model(model, data_iter, max_len, device, mean, std, idxs_to_run=None):
    pred_seqs = []
    src_seqs, tgt_seqs = [], []
    prev, cur = 0, 0  # loop through idxs_to_run
    total_seqs_num = 0
    iterations = 0
    for src_seq, tgt_seq in data_iter:
        total_seqs_num += src_seq.shape[0]
        logging.info(f"Test iteration {iterations}")
        
        if idxs_to_run is not None:
            while cur < len(idxs_to_run) and total_seqs_num >= idxs_to_run[cur]:
                cur += 1
            idxs = [idx - total_seqs_num for idx in idxs_to_run[prev:cur]]
            prev = cur
            src_seq, tgt_seq = src_seq[idxs], tgt_seq[idxs]
        
        if src_seq.shape[0]:
            max_len = max_len if max_len else tgt_seq.shape[1]
            src_seqs.extend(src_seq.to(device="cpu").numpy())
            tgt_seqs.extend(tgt_seq.to(device="cpu").numpy())
            pred_seq = (
                generate.generate(model, src_seq, max_len, device)
                .to(device="cpu")
                .numpy()
            )
            pred_seqs.extend(pred_seq)
        iterations += 1
    return [
        utils.unnormalize(np.array(l), mean, std)
        for l in [pred_seqs, src_seqs, tgt_seqs]
    ]


def save_seq(i, pred_seq, src_seq, tgt_seq, skel, output_path):
    # seq_T contains pred, src, tgt data in the same order
    motions = [
        motion_class.Motion.from_matrix(seq, skel)
        for seq in [pred_seq, src_seq, tgt_seq]
    ]
    ref_motion = motion_ops.append(motions[1], motions[2])
    pred_motion = motion_ops.append(motions[1], motions[0])
    bvh.save(
        ref_motion, os.path.join(output_path, "ref", f"{i}.bvh"),
    )
    bvh.save(
        pred_motion, os.path.join(output_path, "pred", f"{i}.bvh"),
    )


def convert_to_T(pred_seqs, src_seqs, tgt_seqs, rep):
    ops = utils.convert_fn_to_R(rep)
    seqs_T = [
        conversions.R2T(utils.apply_ops(seqs, ops))
        for seqs in [pred_seqs, src_seqs, tgt_seqs]
    ]
    return seqs_T

def save_motion_files_idx(l, num):
    return [i for i in range(0, l, l // num)]
    
def save_motion_files(seqs_T, args, idxs_to_save):
    amass_dip_motion = amass_dip.load(
        file=None, load_skel=True, load_motion=False,
    )
    fairmotion_utils.create_dir_if_absent(os.path.join(args.save_output_path, "ref"))
    fairmotion_utils.create_dir_if_absent(os.path.join(args.save_output_path, "pred"))

    pool = Pool(4)
    skels = [amass_dip_motion.skel for _ in idxs_to_save]
    dirs = [args.save_output_path for _ in idxs_to_save]
    pool.starmap(save_seq, zip(idxs_to_save, *seqs_T, skels, dirs))


def calculate_metrics(pred_seqs, tgt_seqs, metric="Euler"):
    metric_frames = [6, 12, 18, 24]
    R_pred, _ = conversions.T2Rp(pred_seqs)
    R_tgt, _ = conversions.T2Rp(tgt_seqs)
    p_pred = conversions.forward_kinematics(R_pred, amass_dip.SMPL_PARENTS, amass_dip.SMPL_OFFSETS)
    p_tgt = conversions.forward_kinematics(R_tgt, amass_dip.SMPL_PARENTS, amass_dip.SMPL_OFFSETS)
    if metric == "Euler":
        euler_error = metrics.euler_diff(
            R_pred[:, :, amass_dip.SMPL_MAJOR_JOINTS],
            R_tgt[:, :, amass_dip.SMPL_MAJOR_JOINTS],
        )
        euler_error = np.mean(euler_error, axis=0)
        mae = {frame: np.sum(euler_error[:frame]) for frame in metric_frames}
    elif metric == "JointAngle":
        angle_error = metrics.joint_angle_diff(
            R_pred[:, :, amass_dip.SMPL_MAJOR_JOINTS],
            R_tgt[:, :, amass_dip.SMPL_MAJOR_JOINTS],
        )
        angle_error = np.mean(angle_error, axis=0)
        mae = {frame: np.sum(angle_error[:frame]) for frame in metric_frames}
    elif metric == "Position":
        pos_error = metrics.positional_diff(
            p_pred[:, :, amass_dip.SMPL_MAJOR_JOINTS], 
            p_tgt[:, :, amass_dip.SMPL_MAJOR_JOINTS],
        )
        pos_error = np.mean(pos_error, axis=0)
        mae = {frame: np.sum(pos_error[:frame]) for frame in metric_frames}
    elif metric == "PCK":
        mae = {frame: metrics.pck_diff(
                p_pred[:, :frame, amass_dip.SMPL_MAJOR_JOINTS], 
                p_tgt[:, :frame, amass_dip.SMPL_MAJOR_JOINTS],
            ) for frame in metric_frames}
    elif metric == "All":
        mae = {frame:{} for frame in metric_frames}
        for m in ["Euler", "Position", "PCK"]:
            error = calculate_metrics(pred_seqs, tgt_seqs, m)
            for frame in metric_frames:
                mae[frame][m] = error.get(frame)
    return mae


def test_model(model, dataset, rep, device, mean, std, max_len=None, 
               idxs_to_run=None, metric="Euler"):
    pred_seqs, src_seqs, tgt_seqs = run_model(
        model, dataset, max_len, device, mean, std, idxs_to_run
    )
    seqs_T = convert_to_T(pred_seqs, src_seqs, tgt_seqs, rep)
    # Calculate metric only when idxs_to_run is not specified and
    # generated sequence has same shape as reference target sequence
    if idxs_to_run is None and len(pred_seqs) > 0 and pred_seqs[0].shape == tgt_seqs[0].shape:
        mae = calculate_metrics(seqs_T[0], seqs_T[2], metric=metric)
    else:
        mae = {}
    return seqs_T, mae


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Preparing dataset")
    mean, std = utils.prepare_mean_and_std(args.preprocessed_path, recalc=False)
    dataset = {}
    dataset['test'] = utils.prepare_dataloader_with_mean_std(
            os.path.join(args.preprocessed_path, "test.pkl"),
            args.batch_size,
            device,
            args.shuffle,
            mean,
            std
        )
    
    # number of predictions per time step = num_joints * angle representation
    data_shape = next(iter(dataset["test"]))[0].shape
    num_predictions = data_shape[-1]

    logging.info("Preparing model")
    model = prepare_model(
        f"{args.save_model_path}/{args.epoch if args.epoch else 'best'}.model",
        num_predictions,
        args,
        device,
    )

    logging.info("Running model")
    _, rep = os.path.split(args.preprocessed_path.strip("/"))
    
    if args.max_len is not None:
        idxs_to_run = save_motion_files_idx(len(dataset["test"].dataset), args.save_output_num)
        logging.info(f'args.max_len is {args.max_len}, not default (None)')
        logging.info(f'Only run/save model on following indices {idxs_to_run}')
    else:
        idxs_to_run = None
        logging.info('Running model for all seqs')
    
    seqs_T, mae = test_model(
        model, dataset["test"], rep, device, mean, std, args.max_len, idxs_to_run, args.metric
    )
    
    if len(mae):
        logging.info(
            "Test MAE: "
            + " | ".join([f"{frame}: {mae[frame]}" for frame in mae.keys()])
        )

    if args.save_output_path:
        logging.info("Saving results")
        if idxs_to_run is None:
            idxs_to_save = save_motion_files_idx(len(dataset["test"].dataset), args.save_output_num)
            seqs_T = [seqs[np.array(idxs_to_save)] for seqs in seqs_T]
        else:
            idxs_to_save = idxs_to_run
        save_motion_files(seqs_T, args, idxs_to_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate predictions and post process them"
    )
    parser.add_argument(
        "--preprocessed-path",
        type=str,
        help="Path to folder with pickled files from dataset",
        required=True,
    )
    parser.add_argument(
        "--save-model-path",
        type=str,
        help="Path to saved models",
        required=True,
    )
    parser.add_argument(
        "--save-output-path",
        type=str,
        help="Path to store predicted motion",
        default=None,
    )
    parser.add_argument(
        "--save-output-num",
        type=int,
        help="Number of motion files to save",
        default=10,
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
        "--max-len", 
        type=int, 
        help="Length of seqs to generate, non-default will only run selected seqs based on save-output-num", 
        default=None,
    )
    parser.add_argument(
        "--batch-size", type=int, help="Batch size for testing", default=16
    )
    parser.add_argument(
        "--shuffle", action='store_false',
        help="Use this option to enable shuffling",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        help="Model from epoch to test, will test on best"
        " model if not specified",
        default=None,
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
        "--metric",
        type=str,
        help="Metric to be tested",
        default="All",
        choices=[
            "Euler",
            "JointAngle",
            "Position",
            "PCK",
            "PS_Ent",
            "PS_KL",
            "All",
        ],
    )

    args = parser.parse_args()
    main(args)
