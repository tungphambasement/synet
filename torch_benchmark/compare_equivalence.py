#!/usr/bin/env python3
import os
import sys
import argparse
import json
import re
import numpy as np
from pathlib import Path
from tabulate import tabulate


def topological_key(name):
    """Return the execution-order key for TunX/PyTorch operator names."""
    base_name = name
    for suffix in (".act.grad.bin", ".act.bin", ".batch_mean.bin", ".batch_invar.bin",
                   ".running_mean.bin", ".running_var.bin", ".weight.grad",
                   ".bias.grad", ".weight.updated", ".bias.updated",
                   ".weight", ".bias", ".grad", ".updated"):
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
            break

    if base_name == "conv1":
        return (0, 0, 0, name)
    if base_name == "bn1":
        return (0, 1, 0, name)
    if base_name == "maxpool":
        return (0, 2, 0, name)

    match = re.fullmatch(r"layer(\d+)_block(\d+)_(conv|bn)(\d+)", base_name)
    if match:
        stage, block, operator, index = match.groups()
        operator_order = {"conv1": 0, "bn0": 1, "conv2": 2, "bn1": 3,
                          "conv3": 4, "bn2": 5, "conv0": 6, "bn3": 7}
        return (1, int(stage), int(block), operator_order[f"{operator}{index}"], name)

    tail_order = {"avgpool": 0, "flatten": 1, "fc": 2}
    if base_name in tail_order:
        return (2, tail_order[base_name], 0, 0, name)

    numbers = tuple(int(value) for value in re.findall(r"\d+", base_name))
    return (3, base_name, numbers, name)


def sort_results(results):
    results.sort(key=lambda result: topological_key(result["name"]))
    return results

def load_tensor_bin(path, dtype=np.float32):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=dtype)
    return data

def compare_tensors(t1, t2, name, rtol=1e-3, atol=1e-3):
    if t1 is None or t2 is None:
        return None
        
    if t1.shape != t2.shape:
        print(f"Shape mismatch for {name}: {t1.shape} vs {t2.shape}")
        # Try to flatten
        t1 = t1.flatten()
        t2 = t2.flatten()
        
        if t1.shape != t2.shape:
            return None
            
    abs_diff = np.abs(t1 - t2)
    max_abs = np.max(abs_diff)
    p99_abs = np.percentile(abs_diff, 99)

    # Element-wise max relative error is dominated by outliers where t2 is
    # near zero (division blows up even for a tiny absolute difference), so
    # report it alongside the normalized-L2 relative error, which is a much
    # more stable aggregate measure of relative discrepancy.
    t2_safe = np.where(np.abs(t2) < 1e-12, 1e-12, t2)
    rel_diff = np.abs(t1 - t2) / np.abs(t2_safe)
    max_rel = np.max(rel_diff)

    norm1 = np.linalg.norm(t1)
    norm2 = np.linalg.norm(t2)
    normalized_l2 = np.linalg.norm(abs_diff) / norm2 if norm2 > 1e-12 else float("nan")

    # Cosine similarity
    if norm1 < 1e-12 or norm2 < 1e-12:
        cos_sim = 1.0 if max_abs < 1e-6 else 0.0
    else:
        cos_sim = np.dot(t1.flatten(), t2.flatten()) / (norm1 * norm2)
        
    is_allclose_mask = np.isclose(t1, t2, rtol=rtol, atol=atol)
    is_allclose = np.all(is_allclose_mask)
    elements_passed = np.sum(is_allclose_mask)
    total_elements = t1.size
    
    return {
        'name': name,
        'max_abs': max_abs,
        'p99_abs': p99_abs,
        'max_rel': max_rel,
        'normalized_l2': normalized_l2,
        'cos_sim': cos_sim,
        'allclose': is_allclose,
        'elements_passed': int(elements_passed),
        'total_elements': int(total_elements)
    }

def print_table(title, results):
    if not results:
        return
    sort_results(results)
    print(f"\n--- {title} ---")
    table_data = [[r['name'], f"{r['max_abs']:.2e}", f"{r['p99_abs']:.2e}", f"{r['max_rel']:.2e}",
                   f"{r['normalized_l2']:.2e}", f"{r['cos_sim']:.6f}", r['allclose'],
                   f"{r['elements_passed']}/{r['total_elements']} ({(r['elements_passed']/r['total_elements'])*100:.2f}%)"]
                  for r in results]
    print(tabulate(table_data, headers=["Tensor", "Max Abs Err", "P99 Abs Err", "Max Rel Err",
                                        "Normalized L2 Rel Err", "Cosine Sim", "Allclose (1e-3)",
                                        "Elements Passed"]))

def summarize_section(results):
    finite_l2 = [r['normalized_l2'] for r in results if np.isfinite(r['normalized_l2'])]
    return {
        'tensor_count': len(results),
        'max_normalized_l2': float(max(finite_l2)) if finite_l2 else float('nan'),
        'mean_normalized_l2': float(np.mean(finite_l2)) if finite_l2 else float('nan'),
        'min_cosine': float(min(r['cos_sim'] for r in results)),
        'tensor_allclose_pct': float(100 * sum(r['allclose'] for r in results) / len(results)),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dir", type=str, required=True)
    parser.add_argument("--tunx_dir", type=str, required=True)
    parser.add_argument("--summary-json", type=str, default=None)
    args = parser.parse_args()

    results_out = []
    results_loss_grad = []
    results_act = []
    results_act_grad = []
    results_bn_stats = []
    results_grad = []
    results_upd = []
    
    # 1. Compare outputs
    pt_out = load_tensor_bin(os.path.join(args.pt_dir, "outputs.bin"))
    tunx_out = load_tensor_bin(os.path.join(args.tunx_dir, "outputs.bin"))
    
    if pt_out is None or tunx_out is None:
        print("Error: Missing outputs.bin in one of the directories.")
        sys.exit(1)
        
    res = compare_tensors(pt_out, tunx_out, "Forward Outputs")
    if res:
        results_out.append(res)

    # Compare the loss gradient entering model backward before any layer-specific
    # backward implementation can amplify a discrepancy.
    pt_loss_grad = load_tensor_bin(os.path.join(args.pt_dir, "grad_output.bin"))
    tunx_loss_grad = load_tensor_bin(os.path.join(args.tunx_dir, "grad_output.bin"))
    if pt_loss_grad is not None and tunx_loss_grad is not None:
        res = compare_tensors(pt_loss_grad, tunx_loss_grad, "Loss Gradient")
        if res:
            results_loss_grad.append(res)
        
    # 2. Compare gradients and updated parameters
    pt_files = os.listdir(args.pt_dir)
    param_names = set()
    
    for filename in pt_files:
        if filename.endswith((".batch_mean.bin", ".batch_invar.bin", ".running_mean.bin", ".running_var.bin")):
            pt_path = os.path.join(args.pt_dir, filename)
            tunx_path = os.path.join(args.tunx_dir, filename)
            if os.path.exists(tunx_path):
                metrics = compare_tensors(load_tensor_bin(pt_path), load_tensor_bin(tunx_path), filename)
                if metrics: results_bn_stats.append(metrics)
            continue

        # Check for intermediate activations
        if filename.endswith(".act.bin"):
            pt_path = os.path.join(args.pt_dir, filename)
            tunx_path = os.path.join(args.tunx_dir, filename)
            if os.path.exists(tunx_path):
                metrics = compare_tensors(load_tensor_bin(pt_path), load_tensor_bin(tunx_path), filename)
                if metrics: results_act.append(metrics)
            continue
            
        # Check for intermediate gradients
        if filename.endswith(".act.grad.bin"):
            pt_path = os.path.join(args.pt_dir, filename)
            tunx_path = os.path.join(args.tunx_dir, filename)
            if os.path.exists(tunx_path):
                metrics = compare_tensors(load_tensor_bin(pt_path), load_tensor_bin(tunx_path), filename)
                if metrics: results_act_grad.append(metrics)
            continue

        # Extract base parameter names
        if filename.endswith(".bin") and not (filename.endswith(".act.bin") or filename.endswith(".grad.bin") or filename.endswith(".updated.bin") or filename.endswith((".batch_mean.bin", ".batch_invar.bin", ".running_mean.bin", ".running_var.bin")) or filename in ["inputs.bin", "labels.bin", "outputs.bin", "grad_output.bin"]):
            param_names.add(filename[:-4])
            
    if not param_names:
        print("Error: No parameters found in PyTorch dump directory.")
        sys.exit(1)
        
    missing_files = False
    
    for name in sorted(param_names, key=topological_key):
        # Compare gradients
        pt_grad = load_tensor_bin(os.path.join(args.pt_dir, f"{name}.grad.bin"))
        tunx_grad = load_tensor_bin(os.path.join(args.tunx_dir, f"{name}.grad.bin"))
        if pt_grad is None or tunx_grad is None:
            print(f"Error: Missing gradient dump for {name}")
            missing_files = True
        else:
            res = compare_tensors(pt_grad, tunx_grad, f"{name}.grad")
            if res: results_grad.append(res)
            
        # Compare updated params
        pt_upd = load_tensor_bin(os.path.join(args.pt_dir, f"{name}.updated.bin"))
        tunx_upd = load_tensor_bin(os.path.join(args.tunx_dir, f"{name}.updated.bin"))
        if pt_upd is None or tunx_upd is None:
            print(f"Error: Missing updated param dump for {name}")
            missing_files = True
        else:
            res = compare_tensors(pt_upd, tunx_upd, f"{name}.updated")
            if res: results_upd.append(res)

    if missing_files:
        print("Error: Failed due to missing expected dumps.")
        sys.exit(1)
        
    print_table("Outputs", results_out)
    if results_loss_grad: print_table("Loss Gradient", results_loss_grad)
    if results_act: print_table("Intermediate Activations", results_act)
    if results_act_grad: print_table("Intermediate Gradients", results_act_grad)
    if results_bn_stats: print_table("BatchNorm Statistics", results_bn_stats)
    print_table("Gradients", results_grad)
    print_table("Updated Parameters", results_upd)

    sections = {
        "forward_outputs": results_out,
        "loss_gradient": results_loss_grad,
        "activations": results_act,
        "activation_gradients": results_act_grad,
        "batchnorm_statistics": results_bn_stats,
        "parameter_gradients": results_grad,
        "updated_parameters": results_upd,
    }
    section_summary = {name: summarize_section(items) for name, items in sections.items() if items}
    print("\n--- Section Aggregates ---")
    print(tabulate(
        [[name, values['tensor_count'], f"{values['max_normalized_l2']:.2e}",
          f"{values['mean_normalized_l2']:.2e}", f"{values['min_cosine']:.6f}",
          f"{values['tensor_allclose_pct']:.2f}%"]
         for name, values in section_summary.items()],
        headers=["Section", "Tensors", "Max Norm L2", "Mean Norm L2", "Min Cosine", "Allclose"]
    ))

    all_results = results_out + results_loss_grad + results_act + results_act_grad + results_bn_stats + results_grad + results_upd
    if not all_results:
        print("No comparison results found.")
        return

    # Calculate aggregates across all tensors
    overall_max_abs = max(r['max_abs'] for r in all_results)
    overall_p99_abs = max(r['p99_abs'] for r in all_results)
    overall_max_rel = max(r['max_rel'] for r in all_results)
    finite_l2 = [r['normalized_l2'] for r in all_results if np.isfinite(r['normalized_l2'])]
    overall_max_l2 = max(finite_l2) if finite_l2 else float("nan")
    overall_mean_l2 = float(np.mean(finite_l2)) if finite_l2 else float("nan")
    overall_min_cos = min(r['cos_sim'] for r in all_results)
    
    total_tensors = len(all_results)
    allclose_passes = sum(1 for r in all_results if r['allclose'])
    allclose_pct = (allclose_passes / total_tensors) * 100
    
    total_elements = sum(r['total_elements'] for r in all_results)
    total_passed_elements = sum(r['elements_passed'] for r in all_results)
    element_pass_pct = (total_passed_elements / total_elements) * 100 if total_elements > 0 else 0
    
    print("\n--- Aggregate Metrics (RQ4) ---")
    agg_data = [
        ["Max absolute error", f"{overall_max_abs:.2e}"],
        ["P99 absolute error", f"{overall_p99_abs:.2e}"],
        ["Max element-wise relative error", f"{overall_max_rel:.2e}"],
        ["Max normalized L2 relative error", f"{overall_max_l2:.2e}"],
        ["Mean normalized L2 relative error", f"{overall_mean_l2:.2e}"],
        ["Min cosine similarity", f"{overall_min_cos:.6f}"],
        ["Tensor-level allclose passes (%)", f"{allclose_pct:.2f}% ({allclose_passes}/{total_tensors})"],
        ["Element-level allclose passes (%)", f"{element_pass_pct:.2f}% ({total_passed_elements}/{total_elements})"]
    ]
    print(tabulate(agg_data, headers=["Metric", "Value"]))

    if args.summary_json:
        summary = {
            "outputs": results_out,
            "loss_gradient": results_loss_grad,
            "activations": results_act,
            "activation_gradients": results_act_grad,
            "batchnorm_statistics": results_bn_stats,
            "gradients": results_grad,
            "updated_parameters": results_upd,
            "sections": section_summary,
            "aggregate": {
                "max_abs": float(overall_max_abs),
                "p99_abs": float(overall_p99_abs),
                "max_rel": float(overall_max_rel),
                "max_normalized_l2": float(overall_max_l2),
                "mean_normalized_l2": float(overall_mean_l2),
                "min_cosine": float(overall_min_cos),
                "tensor_allclose_pct": float(allclose_pct),
                "element_allclose_pct": float(element_pass_pct),
                "tensor_count": total_tensors,
            },
        }
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, allow_nan=True,
                      default=lambda value: value.item() if hasattr(value, "item") else str(value))

if __name__ == "__main__":
    main()

