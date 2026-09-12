#!/usr/bin/env python3
"""Diagnose a convolution weight-gradient mismatch from raw NHWC dumps."""
import argparse
import os

import numpy as np


def load_tensor(path, shape):
    values = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(shape))
    if values.size != expected:
        raise ValueError(f"{path}: expected {expected} values for {shape}, got {values.size}")
    return values.reshape(shape)


def convolution_wgrad_simple(input_tensor, grad_output, kernel, stride, padding, accumulation_dtype):
    batch, input_height, input_width, input_channels = input_tensor.shape
    _, output_height, output_width, output_channels = grad_output.shape
    kernel_height, kernel_width = kernel
    stride_height, stride_width = stride
    pad_height, pad_width = padding
    gradient = np.zeros(
        (output_channels, kernel_height, kernel_width, input_channels),
        dtype=accumulation_dtype,
    )
    for output_y in range(output_height):
        input_y = output_y * stride_height - pad_height
        for output_x in range(output_width):
            input_x = output_x * stride_width - pad_width
            for kernel_y in range(kernel_height):
                source_y = input_y + kernel_y
                if source_y < 0 or source_y >= input_height:
                    continue
                for kernel_x in range(kernel_width):
                    source_x = input_x + kernel_x
                    if source_x < 0 or source_x >= input_width:
                        continue
                    gradient[:, kernel_y, kernel_x, :] += np.einsum(
                        "no,ni->oi",
                        grad_output[:, output_y, output_x, :].astype(accumulation_dtype),
                        input_tensor[:, source_y, source_x, :].astype(accumulation_dtype),
                        optimize=True,
                    )
    return gradient


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt-dir", required=True)
    parser.add_argument("--tunx-dir", required=True)
    parser.add_argument("--input-file", required=True,
                        help="Input activation filename relative to each dump directory")
    parser.add_argument("--grad-output-file", required=True,
                        help="Upstream activation-gradient filename relative to each dump directory")
    parser.add_argument("--weight-grad-name", required=True,
                        help="Parameter base name, for example conv1.weight")
    parser.add_argument("--input-shape", nargs=4, type=int, required=True,
                        metavar=("N", "H", "W", "C"))
    parser.add_argument("--grad-output-shape", nargs=4, type=int, required=True,
                        metavar=("N", "H", "W", "C_out"))
    parser.add_argument("--kernel", nargs=2, type=int, default=(3, 3), metavar=("KH", "KW"))
    parser.add_argument("--stride", nargs=2, type=int, default=(1, 1), metavar=("SH", "SW"))
    parser.add_argument("--padding", nargs=2, type=int, default=(0, 0), metavar=("PH", "PW"))
    parser.add_argument("--float64", action="store_true",
                        help="Accumulate the reference gradient in float64")
    args = parser.parse_args()

    input_path = os.path.join(args.pt_dir, args.input_file)
    pt_grad_output_path = os.path.join(args.pt_dir, args.grad_output_file)
    tunx_grad_output_path = os.path.join(args.tunx_dir, args.grad_output_file)
    pt_weight_path = os.path.join(args.pt_dir, f"{args.weight_grad_name}.grad.bin")
    tunx_weight_path = os.path.join(args.tunx_dir, f"{args.weight_grad_name}.grad.bin")

    accumulation_dtype = np.float64 if args.float64 else np.float32
    input_tensor = load_tensor(input_path, args.input_shape)
    pt_grad_output = load_tensor(pt_grad_output_path, args.grad_output_shape)
    tunx_grad_output = load_tensor(tunx_grad_output_path, args.grad_output_shape)
    pt_weight_gradient = load_tensor(pt_weight_path, (
        args.grad_output_shape[3], args.kernel[0], args.kernel[1], args.input_shape[3]))
    tunx_weight_gradient = load_tensor(tunx_weight_path, pt_weight_gradient.shape)

    pt_reference = convolution_wgrad_simple(
        input_tensor, pt_grad_output, args.kernel, args.stride, args.padding, accumulation_dtype)
    tunx_reference = convolution_wgrad_simple(
        input_tensor, tunx_grad_output, args.kernel, args.stride, args.padding, accumulation_dtype)

    print(f"Reference input: {args.input_file} {tuple(args.input_shape)}")
    print(f"Reference grad: {args.grad_output_file} {tuple(args.grad_output_shape)}")
    print(f"Reference accumulation dtype: {accumulation_dtype.__name__}")
    report(pt_reference, tunx_reference, "PyTorch reference vs TunX reference")
    report(pt_reference, pt_weight_gradient, "PyTorch reference vs dumped PyTorch weight grad")
    report(pt_reference, tunx_weight_gradient, "PyTorch reference vs dumped TunX weight grad")
    report(tunx_reference, tunx_weight_gradient, "TunX reference vs dumped TunX weight grad")


if __name__ == "__main__":
    main()
