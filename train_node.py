#!/usr/bin/env python

# Script to train on a specific portion of the data
# for distributed training across multiple nodes

import argparse
import os
import sys
import numpy as np
from helper_code import *
from team_code import train_model
import random

# Parse arguments.
def get_parser():
    description = 'Train the Challenge model on a specific data partition.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    parser.add_argument('-m', '--model_folder', type=str, required=True)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-p', '--start_percent', type=float, default=0.0, 
                        help='Starting percentage of data to process (0-100)')
    parser.add_argument('-t', '--percent_to_process', type=float, default=100.0,
                        help='Percentage of data to process (0-100)')
    return parser

# Run the code.
def run(args):
    # Find the data files.
    if args.verbose:
        print(f'Finding the Challenge data for range {args.start_percent}% to {args.start_percent + args.percent_to_process}%...')

    # Get all records
    all_records = find_records(args.data_folder)
    random.seed(42)  # Use a fixed seed for reproducibility
    random.shuffle(all_records)  # Shuffle for better distribution
    
    # Calculate record indices for this node
    total_records = len(all_records)
    start_idx = int(args.start_percent / 100 * total_records)
    end_idx = int((args.start_percent + args.percent_to_process) / 100 * total_records)
    end_idx = min(end_idx, total_records)
    
    # Create a temporary subdirectory with only this node's records
    node_data_folder = f"{args.data_folder}_node{os.environ.get('SLURM_PROCID', '0')}"
    os.makedirs(node_data_folder, exist_ok=True)
    
    # Create symbolic links to the records for this node
    if args.verbose:
        print(f'Processing {end_idx - start_idx} records out of {total_records} total records')
        
    for record in all_records[start_idx:end_idx]:
        os.symlink(os.path.join(args.data_folder, record), os.path.join(node_data_folder, record))
    
    # Train the model on this node's data partition
    if args.verbose:
        print(f'Training model on node {os.environ.get("SLURM_PROCID", "0")}...')
    
    train_model(node_data_folder, args.model_folder, args.verbose)
    
    # Clean up the temporary directory
    for record in all_records[start_idx:end_idx]:
        if os.path.exists(os.path.join(node_data_folder, record)):
            os.unlink(os.path.join(node_data_folder, record))
    
    os.rmdir(node_data_folder)

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
