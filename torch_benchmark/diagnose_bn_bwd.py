#!/usr/bin/env python3
"""Diagnose a BatchNorm backward mismatch from raw NHWC dumps."""
import argparse
import os

import numpy as np


def load_tensor(path, shape):
    values = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(shape))
    if values.size != expected:
        raise ValueError(f"{path}: expected {expected} values for {shape}, got {values.size}")
    return values.reshape(shape)


def batchnorm_dx(input_tensor, grad_output, gamma, mean, inv_std, accumulation_dtype):
    batch, height, width, channels = input_tensor.shape
    sample_count = batch * height * width
    x_hat = (input_tensor.astype(accumulation_dtype) - mean) * inv_std
    grad_output = grad_output.astype(accumulation_dtype)
    gamma = gamma.astype(accumulation_dtype)
    sum_dy = grad_output.sum(axis=(0, 1, 2), dtype=accumulation_dtype)
    sum_dy_xhat = (grad_output * x_hat).sum(axis=(0, 1, 2), dtype=accumulation_dtype)
    scale = gamma * inv_std / sample_count
    return scale * (sample_count * grad_output - sum_dy - x_hat * sum_dy_xhat)


def report(reference, actual, label):
    difference = np.abs(reference - actual)
    reference_norm = np.linalg.norm(reference)
    actual_norm = np.linalg.norm(actual)
    cosine = np.dot(reference.ravel(), actual.ravel()) / (reference_norm * actual_norm)
    normalized_l2 = np.linalg.norm(difference) / reference_norm
    print(
        f"{label}: max_abs={difference.max():.6g} "
        f"p99_abs={np.percentile(difference, 99):.6g} "
        f"normalized_l2={normalized_l2:.6g} cosine={cosine:.6f}"
    )


def preceding_convolution(layer):
    prefix, batchnorm_index = layer.rsplit("_bn", 1)
    convolution_index = int(batchnorm_index) + 1
    return f"{prefix}_conv{convolution_index}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt-dir", required=True)
    parser.add_argument("--tunx-dir", required=True)
    parser.add_argument("--layer", required=True,
                        help="BatchNorm layer base name, for example layer4_block3_bn2")
    parser.add_argument("--input-shape", nargs=4, type=int, required=True,
                        metavar=("N", "H", "W", "C"))
    parser.add_argument("--float64", action="store_true")
    args = parser.parse_args()

    accumulation_dtype = np.float64 if args.float64 else np.float32
    preceding_conv = preceding_convolution(args.layer)
    input_name = f"{preceding_conv}.act.bin"
    upstream_name = f"{args.layer}.act.grad.bin"
    expected_name = f"{preceding_conv}.act.grad.bin"
    gamma_name = f"{args.layer}.weight.bin"
    mean_name = f"{args.layer}.batch_mean.bin"
    inv_std_name = f"{args.layer}.batch_invar.bin"

    shape = tuple(args.input_shape)
    channel_shape = (shape[3],)
    tensors = {}
    for label, directory in (("PyTorch", args.pt_dir), ("TunX", args.tunx_dir)):
        tensors[label] = {
            "input": load_tensor(os.path.join(directory, input_name), shape),
            "upstream": load_tensor(os.path.join(directory, upstream_name), shape),
            "gamma": load_tensor(os.path.join(args.pt_dir, gamma_name), channel_shape),
            "mean": load_tensor(os.path.join(directory, mean_name), channel_shape),
            "inv_std": load_tensor(os.path.join(directory, inv_std_name), channel_shape),
            "expected": load_tensor(os.path.join(directory, expected_name), shape),
        }

    references = {}
    for label, values in tensors.items():
        references[label] = batchnorm_dx(
            values["input"], values["upstream"], values["gamma"], values["mean"],
            values["inv_std"], accumulation_dtype)
        report(references[label], values["expected"],
               f"{label} reference vs dumped gradient")

    report(references["PyTorch"], references["TunX"],
           "PyTorch reference vs TunX reference")
    report(tensors["PyTorch"]["expected"], tensors["TunX"]["expected"],
           "PyTorch dumped gradient vs TunX dumped gradient")


if __name__ == "__main__":
    main()
