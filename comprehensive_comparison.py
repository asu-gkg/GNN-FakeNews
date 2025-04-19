#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive Comparison Script
Compare the performance of GNN models under different datasets and feature methods, and generate detailed charts and reports
"""
import argparse
import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import time
from pathlib import Path

# Set command line arguments
parser = argparse.ArgumentParser(description='Comprehensive comparison of GNN model performance across different datasets and features')
parser.add_argument('--datasets', type=str, nargs='+', default=['politifact', 'gossipcop'], 
                    help='List of datasets to compare')
parser.add_argument('--features', type=str, nargs='+', default=['profile', 'spacy', 'bert', 'content'], 
                    help='List of feature types to compare')
parser.add_argument('--device', type=str, default='cuda:0', help='GPU device')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
parser.add_argument('--runs', type=int, default=3, help='Number of runs for each configuration')
parser.add_argument('--seed', type=int, default=42, help='Base random seed')
parser.add_argument('--output_dir', type=str, default='comprehensive_results', help='Output directory')
parser.add_argument('--skip_existing', action='store_true', help='Skip configurations that have already been run')
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Main dataframe for all results
all_results = pd.DataFrame()

# Timer
start_time = time.time()

# Run all configurations
total_configs = len(args.datasets) * len(args.features)
completed = 0

print(f"Will run {total_configs} different configurations, each with {args.runs} runs")

for dataset in args.datasets:
    for feature in args.features:
        config_dir = os.path.join(args.output_dir, f"{dataset}_{feature}")
        os.makedirs(config_dir, exist_ok=True)
        
        # Check if this configuration has already been completed
        summary_file = os.path.join(config_dir, 'summary.csv')
        if args.skip_existing and os.path.exists(summary_file):
            print(f"Configuration {dataset}_{feature} already exists, skipping...")
            # Read existing results
            df = pd.read_csv(summary_file)
            df['dataset'] = dataset
            df['feature'] = feature
            all_results = pd.concat([all_results, df])
            completed += 1
            continue
        
        print(f"\n{'#'*80}")
        print(f"Running configuration {completed+1}/{total_configs}: dataset={dataset}, feature={feature}")
        print(f"{'#'*80}")
        
        # Run compare_models.py
        cmd = [
            'python', 'compare_models.py',
            '--dataset', dataset,
            '--feature', feature,
            '--device', args.device,
            '--batch_size', str(args.batch_size),
            '--epochs', str(args.epochs),
            '--runs', str(args.runs),
            '--seed', str(args.seed),
            '--output_dir', config_dir
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        stdout, stderr = process.communicate()
        
        # Save run log
        with open(os.path.join(config_dir, 'run_log.txt'), 'w') as f:
            f.write(stdout)
            if stderr:
                f.write("\nERRORS:\n")
                f.write(stderr)
        
        # Read comparison results
        try:
            df = pd.read_csv(summary_file)
            df['dataset'] = dataset
            df['feature'] = feature
            all_results = pd.concat([all_results, df])
        except Exception as e:
            print(f"Error reading results for {dataset}_{feature}: {str(e)}")
        
        completed += 1
        
        # Show progress
        elapsed = time.time() - start_time
        estimated_total = elapsed / completed * total_configs
        remaining = estimated_total - elapsed
        
        print(f"Completed: {completed}/{total_configs} configurations")
        print(f"Time elapsed: {elapsed/60:.1f} minutes")
        print(f"Estimated remaining: {remaining/60:.1f} minutes")

# Save all results
all_results.to_csv(os.path.join(args.output_dir, 'all_configurations_results.csv'), index=False)

# Debug: Check the structure of the results dataframe
print("\nChecking results dataframe structure:")
print(f"Columns: {all_results.columns.tolist()}")
print(f"Sample data:\n{all_results.head()}")

# Reshape data for plotting - Handle potential MultiIndex columns
plot_data = all_results.copy()

# Check if we have a MultiIndex in columns (from groupby operations)
if isinstance(plot_data.columns, pd.MultiIndex):
    # Get the correct column names for accuracy, f1, auc
    accuracy_col = [col for col in plot_data.columns if 'accuracy' in str(col) and 'mean' in str(col)]
    f1_col = [col for col in plot_data.columns if 'f1_macro' in str(col) and 'mean' in str(col)]
    auc_col = [col for col in plot_data.columns if 'auc' in str(col) and 'mean' in str(col)]
    time_col = [col for col in plot_data.columns if 'time' in str(col) and 'mean' in str(col)]
    
    # Create simpler column names
    new_columns = {}
    if accuracy_col: new_columns[accuracy_col[0]] = 'accuracy_mean'
    if f1_col: new_columns[f1_col[0]] = 'f1_macro_mean'
    if auc_col: new_columns[auc_col[0]] = 'auc_mean'
    if time_col: new_columns[time_col[0]] = 'time_mean'
    
    # Rename columns
    plot_data = plot_data.rename(columns=new_columns)
else:
    # For non-MultiIndex, ensure we have the right columns
    if 'accuracy' in plot_data.columns and 'accuracy_mean' not in plot_data.columns:
        plot_data = plot_data.rename(columns={
            'accuracy': 'accuracy_mean',
            'f1_macro': 'f1_macro_mean',
            'auc': 'auc_mean',
            'time': 'time_mean'
        })

# Verify we have the necessary columns for plotting
required_cols = ['model', 'dataset', 'feature', 'accuracy_mean', 'f1_macro_mean', 'auc_mean']
missing_cols = [col for col in required_cols if col not in plot_data.columns]

if missing_cols:
    print(f"Warning: Missing required columns: {missing_cols}")
    print("Available columns:", plot_data.columns.tolist())
    print("Will attempt to create or rename columns as needed")
    
    # Try to fix missing columns
    for col in missing_cols:
        if col == 'accuracy_mean' and 'accuracy' in plot_data.columns:
            plot_data['accuracy_mean'] = plot_data['accuracy']
        elif col == 'f1_macro_mean' and 'f1_macro' in plot_data.columns:
            plot_data['f1_macro_mean'] = plot_data['f1_macro']
        elif col == 'auc_mean' and 'auc' in plot_data.columns:
            plot_data['auc_mean'] = plot_data['auc']
            
# Save the processed data for debugging
plot_data.to_csv(os.path.join(args.output_dir, 'processed_data_for_plotting.csv'), index=False)
print(f"Processed data saved to {os.path.join(args.output_dir, 'processed_data_for_plotting.csv')}")

# ====== Generate visualization charts ======
try:
    # 1. Accuracy heatmap - different datasets and features
    plt.figure(figsize=(12, 8))
    
    # Make sure we have the right columns
    if 'accuracy_mean' in plot_data.columns:
        pivot_acc = plot_data.pivot_table(
            index='dataset', 
            columns='feature', 
            values='accuracy_mean', 
            aggfunc='mean'
        )
        sns.heatmap(pivot_acc, annot=True, fmt='.4f', cmap='YlGnBu', cbar_kws={'label': 'Accuracy'})
        plt.title('Model Accuracy Comparison Across Datasets and Features')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'accuracy_heatmap.png'), dpi=300)
        plt.close()  # 确保图像关闭
    else:
        print("Cannot create accuracy heatmap: 'accuracy_mean' column not found")
    
    # 2. Model performance comparison - grouped bar chart
    metrics = []
    if 'accuracy_mean' in plot_data.columns: metrics.append('accuracy_mean')
    if 'f1_macro_mean' in plot_data.columns: metrics.append('f1_macro_mean')
    if 'auc_mean' in plot_data.columns: metrics.append('auc_mean')
    
    if not metrics:
        print("Cannot create performance comparison charts: No metric columns found")
    else:
        metric_names = [m.replace('_mean', '').title() for m in metrics]
        
        # Create charts for each dataset and feature combination
        for dataset in args.datasets:
            dataset_data = plot_data[plot_data['dataset'] == dataset]
            
            plt.figure(figsize=(15, 8))
            
            # Set group positions
            x = np.arange(len(metrics))
            width = 0.35 / len(args.features)  # Adjust width based on number of features
            
            for i, feature in enumerate(args.features):
                feature_data = dataset_data[dataset_data['feature'] == feature]
                
                if len(feature_data) < 2:
                    continue
                    
                gnn_data = feature_data[feature_data['model'] == 'gnn']
                better_gnn_data = feature_data[feature_data['model'] == 'better_gnn']
                
                if len(gnn_data) == 0 or len(better_gnn_data) == 0:
                    continue
                    
                # Extract performance metrics
                gnn_metrics = [gnn_data[m].values[0] if m in gnn_data else 0 for m in metrics]
                better_gnn_metrics = [better_gnn_data[m].values[0] if m in better_gnn_data else 0 for m in metrics]
                
                # Add bar charts
                offset = i * width * 2 - (len(args.features) * width) / 2
                plt.bar(x + offset, gnn_metrics, width, label=f'GNN ({feature})')
                plt.bar(x + offset + width, better_gnn_metrics, width, label=f'Better GNN ({feature})')
            
            plt.xlabel('Performance Metrics')
            plt.ylabel('Score')
            plt.title(f'Dataset: {dataset} - Performance Comparison Across Features and Models')
            plt.xticks(x, metric_names)
            plt.ylim(0.5, 1.0)  # Set y-axis range to make differences more visible
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(args.features)*2)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f'{dataset}_performance_comparison.png'), dpi=300)
            plt.close()  # 确保图像关闭
    
    # 3. Training time comparison chart
    if 'time_mean' in plot_data.columns:
        plt.figure(figsize=(12, 8))
        try:
            time_data = plot_data.pivot_table(
                index='dataset', 
                columns=['model', 'feature'], 
                values='time_mean'
            )
            time_data.plot(kind='bar', figsize=(15, 8))
            plt.title('Training Time Comparison Across Configurations')
            plt.ylabel('Training Time (seconds)')
            plt.xlabel('Dataset')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend(title='Model and Feature', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'training_time_comparison.png'), dpi=300)
            plt.close()  # 确保图像关闭
        except Exception as e:
            print(f"Error creating training time chart: {str(e)}")
    
    # 4. Model performance improvement percentage
    if all(col in plot_data.columns for col in ['model', 'dataset', 'feature', 'accuracy_mean']):
        improvement_data = []
        
        for dataset in args.datasets:
            for feature in args.features:
                subset = plot_data[(plot_data['dataset'] == dataset) & (plot_data['feature'] == feature)]
                
                if len(subset) < 2:
                    continue
                    
                gnn_data = subset[subset['model'] == 'gnn']
                better_gnn_data = subset[subset['model'] == 'better_gnn']
                
                if len(gnn_data) == 0 or len(better_gnn_data) == 0:
                    continue
                    
                try:
                    accuracy_improvement = (better_gnn_data['accuracy_mean'].values[0] - gnn_data['accuracy_mean'].values[0]) / gnn_data['accuracy_mean'].values[0] * 100
                    
                    f1_improvement = 0
                    if 'f1_macro_mean' in better_gnn_data.columns and 'f1_macro_mean' in gnn_data.columns:
                        f1_improvement = (better_gnn_data['f1_macro_mean'].values[0] - gnn_data['f1_macro_mean'].values[0]) / gnn_data['f1_macro_mean'].values[0] * 100
                    
                    auc_improvement = 0
                    if 'auc_mean' in better_gnn_data.columns and 'auc_mean' in gnn_data.columns:
                        auc_improvement = (better_gnn_data['auc_mean'].values[0] - gnn_data['auc_mean'].values[0]) / gnn_data['auc_mean'].values[0] * 100
                    
                    improvement_data.append({
                        'dataset': dataset,
                        'feature': feature,
                        'accuracy_improvement': accuracy_improvement,
                        'f1_improvement': f1_improvement,
                        'auc_improvement': auc_improvement
                    })
                except Exception as e:
                    print(f"Error calculating improvements for {dataset}_{feature}: {str(e)}")
        
        if improvement_data:
            improvement_df = pd.DataFrame(improvement_data)
            improvement_df.to_csv(os.path.join(args.output_dir, 'improvement_percentage.csv'), index=False)
            
            try:
                # Plot improvement percentage
                plt.figure(figsize=(15, 10))
                
                # Reshape data for plotting
                improvement_plot = pd.melt(
                    improvement_df, 
                    id_vars=['dataset', 'feature'], 
                    value_vars=['accuracy_improvement', 'f1_improvement', 'auc_improvement'],
                    var_name='metric', 
                    value_name='improvement_percentage'
                )
                
                # Replace metric names to make them more readable
                improvement_plot['metric'] = improvement_plot['metric'].replace({
                    'accuracy_improvement': 'Accuracy Improvement',
                    'f1_improvement': 'F1 Score Improvement',
                    'auc_improvement': 'AUC Improvement'
                })
                
                # Draw bar chart
                sns.barplot(x='dataset', y='improvement_percentage', hue='feature', data=improvement_plot)
                plt.title('Performance Improvement Percentage of Better GNN over GNN')
                plt.xlabel('Dataset')
                plt.ylabel('Improvement Percentage (%)')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.xticks(rotation=0)
                plt.legend(title='Feature')
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, 'improvement_percentage.png'), dpi=300)
                plt.close()  # 确保图像关闭
            except Exception as e:
                print(f"Error creating improvement percentage chart: {str(e)}")
except Exception as e:
    print(f"Error during chart generation: {str(e)}")

# Generate report
try:
    report_file = os.path.join(args.output_dir, 'comprehensive_report.md')
    with open(report_file, 'w') as f:
        f.write('# GNN Model Performance Comprehensive Comparison Report\n\n')
        
        f.write('## Experimental Setup\n\n')
        f.write(f'- Datasets: {", ".join(args.datasets)}\n')
        f.write(f'- Features: {", ".join(args.features)}\n')
        f.write(f'- Runs per configuration: {args.runs}\n')
        f.write(f'- Training epochs: {args.epochs}\n')
        f.write(f'- Batch size: {args.batch_size}\n\n')
        
        f.write('## Performance Summary\n\n')
        
        if 'model' in plot_data.columns and 'accuracy_mean' in plot_data.columns:
            # Add overall performance table
            f.write('### Average Performance Across All Configurations\n\n')
            
            # Create performance table
            performance_table = []
            for dataset in args.datasets:
                for feature in args.features:
                    subset = plot_data[(plot_data['dataset'] == dataset) & (plot_data['feature'] == feature)]
                    
                    if len(subset) < 2:
                        continue
                        
                    gnn_data = subset[subset['model'] == 'gnn']
                    better_gnn_data = subset[subset['model'] == 'better_gnn']
                    
                    if len(gnn_data) == 0 or len(better_gnn_data) == 0:
                        continue
                        
                    # Get metrics or use placeholders
                    gnn_acc = f"{gnn_data['accuracy_mean'].values[0]:.4f}" if 'accuracy_mean' in gnn_data.columns else "N/A"
                    bg_acc = f"{better_gnn_data['accuracy_mean'].values[0]:.4f}" if 'accuracy_mean' in better_gnn_data.columns else "N/A"
                    
                    gnn_f1 = f"{gnn_data['f1_macro_mean'].values[0]:.4f}" if 'f1_macro_mean' in gnn_data.columns else "N/A"
                    bg_f1 = f"{better_gnn_data['f1_macro_mean'].values[0]:.4f}" if 'f1_macro_mean' in better_gnn_data.columns else "N/A"
                    
                    gnn_auc = f"{gnn_data['auc_mean'].values[0]:.4f}" if 'auc_mean' in gnn_data.columns else "N/A"
                    bg_auc = f"{better_gnn_data['auc_mean'].values[0]:.4f}" if 'auc_mean' in better_gnn_data.columns else "N/A"
                    
                    row = [
                        dataset,
                        feature,
                        gnn_acc,
                        bg_acc,
                        gnn_f1,
                        bg_f1,
                        gnn_auc,
                        bg_auc
                    ]
                    performance_table.append(row)
            
            if performance_table:
                headers = ['Dataset', 'Feature', 'GNN Accuracy', 'Better GNN Accuracy', 'GNN F1', 'Better GNN F1', 'GNN AUC', 'Better GNN AUC']
                f.write(tabulate(performance_table, headers=headers, tablefmt='pipe'))
                f.write('\n\n')
            
            # Add improvement percentage table
            if improvement_data:
                f.write('### Performance Improvement of Better GNN over GNN\n\n')
                
                improvement_table = []
                for item in improvement_data:
                    row = [
                        item['dataset'],
                        item['feature'],
                        f"{item['accuracy_improvement']:.2f}%",
                        f"{item['f1_improvement']:.2f}%",
                        f"{item['auc_improvement']:.2f}%"
                    ]
                    improvement_table.append(row)
                
                headers = ['Dataset', 'Feature', 'Accuracy Improvement', 'F1 Score Improvement', 'AUC Improvement']
                f.write(tabulate(improvement_table, headers=headers, tablefmt='pipe'))
                f.write('\n\n')
            
            f.write('## Conclusions\n\n')
            
            # Automatically generate some conclusions
            f.write('Based on the experimental results, we can draw the following conclusions:\n\n')
            
            # Find the best feature
            best_feature = ''
            best_acc = 0
            
            for feature in args.features:
                feature_data = plot_data[plot_data['feature'] == feature]
                better_gnn_data = feature_data[feature_data['model'] == 'better_gnn']
                
                if len(better_gnn_data) > 0 and 'accuracy_mean' in better_gnn_data.columns:
                    avg_acc = better_gnn_data['accuracy_mean'].mean()
                    if avg_acc > best_acc:
                        best_acc = avg_acc
                        best_feature = feature
            
            if best_feature:
                f.write(f"1. Among the tested features, {best_feature} feature performed best overall, achieving an average accuracy of {best_acc:.4f}.\n")
            
            # Calculate average improvements
            if improvement_data:
                avg_acc_improvement = np.mean([item['accuracy_improvement'] for item in improvement_data])
                avg_f1_improvement = np.mean([item['f1_improvement'] for item in improvement_data])
                avg_auc_improvement = np.mean([item['auc_improvement'] for item in improvement_data])
                
                f.write(f"2. The Better GNN model shows significant improvements over the base GNN model:\n")
                f.write(f"   - Accuracy improved by an average of {avg_acc_improvement:.2f}%\n")
                f.write(f"   - F1 score improved by an average of {avg_f1_improvement:.2f}%\n")
                f.write(f"   - AUC improved by an average of {avg_auc_improvement:.2f}%\n")
        
        f.write('\n## Chart Descriptions\n\n')
        f.write('1. **accuracy_heatmap.png**: Accuracy heatmap across different datasets and features\n')
        f.write('2. **dataset_performance_comparison.png**: Performance comparison across different features and models for each dataset\n')
        f.write('3. **training_time_comparison.png**: Training time comparison across different configurations\n')
        f.write('4. **improvement_percentage.png**: Performance improvement percentage of Better GNN over GNN\n')
except Exception as e:
    print(f"Error generating report: {str(e)}")

print(f"\nComprehensive comparison completed!")
print(f"- All results saved to: {args.output_dir}/")
print(f"- Detailed report: {report_file}")
print(f"- Total runtime: {(time.time() - start_time)/60:.1f} minutes") 


#  python comprehensive_comparison.py --datasets politifact gossipcop --features bert spacy --runs 2 --epochs 50