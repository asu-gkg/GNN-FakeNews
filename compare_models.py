#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to compare the performance of better_gnn and gnn models
Run both models with the same dataset and parameter settings and compare their performance metrics
"""
import argparse
import time
import os
import numpy as np
import torch
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path

# Set command line arguments
parser = argparse.ArgumentParser(description='Compare the performance of gnn and better_gnn')
parser.add_argument('--dataset', type=str, default='politifact', help='Dataset [politifact, gossipcop]')
parser.add_argument('--feature', type=str, default='bert', help='Feature type [profile, spacy, bert, content]')
parser.add_argument('--device', type=str, default='cuda:0', help='GPU device')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
parser.add_argument('--runs', type=int, default=3, help='Number of runs for each model')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--output_dir', type=str, default='comparison_results', help='Output directory')
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Set parameters for the two models
gnn_params = [
    'python', '-m', 'gnn_model.gnn',
    '--dataset', args.dataset,
    '--feature', args.feature,
    '--device', args.device,
    '--batch_size', str(args.batch_size),
    '--epochs', str(args.epochs),
    '--lr', '0.01',
    '--weight_decay', '0.01',
    '--model', 'sage',
    '--seed'
]

better_gnn_params = [
    'python', '-m', 'gnn_model.better_gnn',
    '--dataset', args.dataset,
    '--feature', args.feature,
    '--device', args.device,
    '--batch_size', str(args.batch_size),
    '--epochs', str(args.epochs),
    '--lr', '1e-3',
    '--weight_decay', '5e-4',
    '--seed'
]

# Results storage
results = {
    'model': [],
    'run': [],
    'time': [],
    'accuracy': [],
    'f1_macro': [],
    'auc': [],
    'ap': []
}

# Run models and collect results
def run_model(model_name, params, run_id, seed):
    print(f"\n{'='*80}")
    print(f"Running {model_name} (Run {run_id+1}/{args.runs})")
    print(f"{'='*80}")
    
    # Set random seed
    current_seed = seed + run_id
    current_params = params + [str(current_seed)]
    
    # Record start time
    start_time = time.time()
    
    # Set environment variables to ensure modules from project root directory can be found
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__)) + ":" + my_env.get('PYTHONPATH', '')
    
    # Run model and capture output
    process = subprocess.Popen(
        current_params,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        env=my_env
    )
    
    stdout, stderr = process.communicate()
    
    # Record end time
    elapsed_time = time.time() - start_time
    
    # Save output to file
    output_file = os.path.join(args.output_dir, f"{model_name}_run{run_id+1}.log")
    with open(output_file, 'w') as f:
        f.write(stdout)
        if stderr:
            f.write("\nERRORS:\n")
            f.write(stderr)
    
    # Extract test results from output
    results['model'].append(model_name)
    results['run'].append(run_id+1)
    results['time'].append(elapsed_time)
    
    # Try to parse results from output
    try:
        # For GNN model output format
        if model_name == 'gnn':
            test_result_lines = [line for line in stdout.split('\n') if 'Test set results:' in line]
            if not test_result_lines:
                raise ValueError(f"Could not find 'Test set results:' line in output. Please check model output format.")
            test_result_line = test_result_lines[0]
            metrics = test_result_line.split('Test set results:')[1].strip()
            accuracy = float(metrics.split('acc:')[1].split(',')[0].strip())
            f1_macro = float(metrics.split('f1_macro:')[1].split(',')[0].strip())
            auc = float(metrics.split('auc:')[1].split(',')[0].strip())
            ap = float(metrics.split('ap:')[1].strip())
        # For Better GNN model output format
        elif model_name == 'better_gnn':
            test_result_lines = [line for line in stdout.split('\n') if '>>> Test' in line]
            if not test_result_lines:
                raise ValueError(f"Could not find '>>> Test' line in output. Please check model output format.")
            test_result_line = test_result_lines[0]
            accuracy = float(test_result_line.split('Acc')[1].split('F1')[0].strip())
            f1_macro = float(test_result_line.split('F1')[1].split('AUC')[0].strip())
            auc = float(test_result_line.split('AUC')[1].strip())
            ap = 0.0  # Better GNN may not report AP value, set to 0
            
        results['accuracy'].append(accuracy)
        results['f1_macro'].append(f1_macro)
        results['auc'].append(auc)
        results['ap'].append(ap)
        
        print(f"Completed {model_name} run {run_id+1}:")
        print(f"  Accuracy: {accuracy:.4f}, F1: {f1_macro:.4f}, AUC: {auc:.4f}, Time: {elapsed_time:.2f}s")
        
    except Exception as e:
        print(f"Error parsing results for {model_name} run {run_id+1}: {str(e)}")
        print(f"Please check output file: {output_file}")
        # If it's a module import error or other serious error, show more information
        if "ModuleNotFoundError" in stderr or "ImportError" in stderr:
            print(f"Error details: Module import error, you may need to set the correct Python path.")
            error_lines = stderr.split('\n')[:5]  # Show first 5 lines of error
            for line in error_lines:
                print(f"  {line}")
        results['accuracy'].append(0.0)
        results['f1_macro'].append(0.0)
        results['auc'].append(0.0)
        results['ap'].append(0.0)

# Run both models multiple times
for run in range(args.runs):
    run_model('gnn', gnn_params, run, args.seed)
    run_model('better_gnn', better_gnn_params, run, args.seed)

# Create DataFrame
df = pd.DataFrame(results)

# Save raw results
df.to_csv(os.path.join(args.output_dir, 'all_results.csv'), index=False)

# Calculate average performance for each model
summary = df.groupby('model').agg({
    'time': ['mean', 'std'],
    'accuracy': ['mean', 'std'],
    'f1_macro': ['mean', 'std'],
    'auc': ['mean', 'std'],
    'ap': ['mean', 'std']
})

# Save summary results
summary.to_csv(os.path.join(args.output_dir, 'summary.csv'))

# Print a nice table in the console
print("\n\nPerformance Comparison Summary:")
table_data = []
for model in ['gnn', 'better_gnn']:
    if model in summary.index:
        row = [
            model,
            f"{summary.loc[model, ('accuracy', 'mean')]:.4f} ± {summary.loc[model, ('accuracy', 'std')]:.4f}",
            f"{summary.loc[model, ('f1_macro', 'mean')]:.4f} ± {summary.loc[model, ('f1_macro', 'std')]:.4f}",
            f"{summary.loc[model, ('auc', 'mean')]:.4f} ± {summary.loc[model, ('auc', 'std')]:.4f}",
            f"{summary.loc[model, ('time', 'mean')]:.2f}s ± {summary.loc[model, ('time', 'std')]:.2f}"
        ]
        table_data.append(row)

print(tabulate(table_data, headers=['Model', 'Accuracy', 'F1 Score', 'AUC', 'Runtime'], tablefmt='pretty'))

# Create performance comparison charts
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Accuracy comparison
axs[0].bar(['GNN', 'Better GNN'], 
          [summary.loc['gnn', ('accuracy', 'mean')], summary.loc['better_gnn', ('accuracy', 'mean')]],
          yerr=[summary.loc['gnn', ('accuracy', 'std')], summary.loc['better_gnn', ('accuracy', 'std')]])
axs[0].set_title('Accuracy Comparison')
axs[0].set_ylim(0.5, 1.0)  # Set y-axis range from 0.5 to 1.0 to make differences more visible

# F1 score comparison
axs[1].bar(['GNN', 'Better GNN'], 
          [summary.loc['gnn', ('f1_macro', 'mean')], summary.loc['better_gnn', ('f1_macro', 'mean')]],
          yerr=[summary.loc['gnn', ('f1_macro', 'std')], summary.loc['better_gnn', ('f1_macro', 'std')]])
axs[1].set_title('F1 Score Comparison')
axs[1].set_ylim(0.5, 1.0)

# AUC comparison
axs[2].bar(['GNN', 'Better GNN'], 
          [summary.loc['gnn', ('auc', 'mean')], summary.loc['better_gnn', ('auc', 'mean')]],
          yerr=[summary.loc['gnn', ('auc', 'std')], summary.loc['better_gnn', ('auc', 'std')]])
axs[2].set_title('AUC Comparison')
axs[2].set_ylim(0.5, 1.0)

plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'performance_comparison.png'), dpi=300)

print(f"\nResults saved to {args.output_dir} directory")
print(f"- CSV files: all_results.csv, summary.csv")
print(f"- Performance comparison chart: performance_comparison.png")
print(f"- Run logs: gnn_run*.log, better_gnn_run*.log")