#!/usr/bin/env python

# Script to split datasets into training and holdout sets with 80/20 split

import argparse
import os
import shutil
import random
import sys
from collections import defaultdict
import glob

def get_parser():
    description = 'Split datasets into training and holdout sets with 80/20 split'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_directory', type=str, required=True, 
                        help='Directory containing all_datasets, training_data, and holdout_data subdirectories')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Force overwrite of existing data in training_data and holdout_data')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    return parser

def add_chagas_label_to_file(file_path):
    """Add Chagas label to PTBXL .hea files"""
    with open(file_path, 'r') as f:
        content = f.read()
        
    # Check if the file already has a Chagas label
    if "# Chagas label:" not in content:
        # Add label at the end of the file
        if content.endswith('\n'):
            content += "# Chagas label: False\n# Source: PTBXL\n"
        else:
            content += "\n# Chagas label: False\n# Source: PTBXL\n"
        
        with open(file_path, 'w') as f:
            f.write(content)

def check_and_create_dirs(base_dir):
    """Check if required directories exist and create them if needed"""
    all_datasets_dir = os.path.join(base_dir, 'all_datasets')
    training_dir = os.path.join(base_dir, 'training_data')
    holdout_dir = os.path.join(base_dir, 'holdout_data')
    
    # Check if all_datasets exists
    if not os.path.isdir(all_datasets_dir):
        print(f"Error: all_datasets directory not found at {all_datasets_dir}")
        return False
    
    # Create training and holdout directories if they don't exist
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)
    
    if not os.path.exists(holdout_dir):
        os.makedirs(holdout_dir)
        
    return all_datasets_dir, training_dir, holdout_dir

def process_code15_output(src_dir, train_dir, holdout_dir, force=False):
    """Process code15_output directory with exams_partX subdirectories"""
    print(f"Processing CODE-15% dataset...")
    
    # Check for exam_part directories
    exam_parts = [d for d in os.listdir(src_dir) if d.startswith('exams_part')]
    if not exam_parts:
        print(f"Warning: No exams_part directories found in {src_dir}")
        
    # Copy CSV file to both directories
    csv_file = os.path.join(src_dir, 'code15_chagas_labels.csv')
    if os.path.exists(csv_file):
        # Copy (not move) the CSV file since it needs to be in both directories
        shutil.copy2(csv_file, train_dir)
        shutil.copy2(csv_file, holdout_dir)
        print(f"Copied code15_chagas_labels.csv to both directories")
    else:
        print(f"Warning: code15_chagas_labels.csv not found in {src_dir}")
    
    # Process each exams_part directory
    all_records = []
    for part_dir in exam_parts:
        part_path = os.path.join(src_dir, part_dir)
        if os.path.isdir(part_path):
            # Get all unique record IDs (without extensions)
            records = set()
            for file in os.listdir(part_path):
                base_name = os.path.splitext(file)[0]
                if os.path.isfile(os.path.join(part_path, file)) and (file.endswith('.dat') or file.endswith('.hea')):
                    records.add(base_name)
            
            # Check each record for both .hea and .dat files
            for record in records:
                hea_file = os.path.join(part_path, f"{record}.hea")
                dat_file = os.path.join(part_path, f"{record}.dat")
                
                if os.path.exists(hea_file) and os.path.exists(dat_file):
                    all_records.append((record, part_dir, hea_file, dat_file))
                else:
                    print(f"Warning: Record {record} in {part_dir} is missing a file:")
                    if not os.path.exists(hea_file):
                        print(f"  - Missing {hea_file}")
                    if not os.path.exists(dat_file):
                        print(f"  - Missing {dat_file}")
    
    # Split records between training and holdout
    random.shuffle(all_records)
    split_idx = int(len(all_records) * 0.8)
    train_records = all_records[:split_idx]
    holdout_records = all_records[split_idx:]
    
    # Move files to training directory
    for record, part_dir, hea_file, dat_file in train_records:
        # Create part directory in training if it doesn't exist
        train_part_dir = os.path.join(train_dir, part_dir)
        os.makedirs(train_part_dir, exist_ok=True)
        
        # Move files instead of copying
        shutil.move(hea_file, train_part_dir)
        shutil.move(dat_file, train_part_dir)
    
    # Move files to holdout directory
    for record, part_dir, hea_file, dat_file in holdout_records:
        # Create part directory in holdout if it doesn't exist
        holdout_part_dir = os.path.join(holdout_dir, part_dir)
        os.makedirs(holdout_part_dir, exist_ok=True)
        
        # Move files instead of copying
        shutil.move(hea_file, holdout_part_dir)
        shutil.move(dat_file, holdout_part_dir)
    
    print(f"CODE-15% dataset: Moved {len(train_records)} records to training and {len(holdout_records)} records to holdout")
    
def process_ptbxl_output(src_dir, train_dir, holdout_dir, force=False):
    """Process ptbxl_output directory with numbered subdirectories"""
    print(f"Processing PTB-XL dataset...")
    
    # Find all numbered directories (00000, 01000, etc.)
    subdirs = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d)) and d.isdigit()]
    
    if not subdirs:
        print(f"Warning: No numbered directories found in {src_dir}")
        return
    
    # Process each subdirectory
    all_records = []
    for subdir in subdirs:
        subdir_path = os.path.join(src_dir, subdir)
        
        # Get all unique record IDs (without extensions)
        files = os.listdir(subdir_path)
        record_bases = set()
        
        for file in files:
            if file.endswith('.hea') or file.endswith('.dat'):
                record_bases.add(os.path.splitext(file)[0])
        
        # Check each record for both .hea and .dat files
        for record_base in record_bases:
            hea_file = os.path.join(subdir_path, f"{record_base}.hea")
            dat_file = os.path.join(subdir_path, f"{record_base}.dat")
            
            if os.path.exists(hea_file) and os.path.exists(dat_file):
                all_records.append((record_base, subdir, hea_file, dat_file))
            else:
                print(f"Warning: Record {record_base} in {subdir} is missing a file:")
                if not os.path.exists(hea_file):
                    print(f"  - Missing {hea_file}")
                if not os.path.exists(dat_file):
                    print(f"  - Missing {dat_file}")
    
    # Split records between training and holdout
    random.shuffle(all_records)
    split_idx = int(len(all_records) * 0.8)
    train_records = all_records[:split_idx]
    holdout_records = all_records[split_idx:]
    
    # Move files to training directory
    for record, subdir, hea_file, dat_file in train_records:
        # Create subdir in training if it doesn't exist
        train_subdir = os.path.join(train_dir, subdir)
        os.makedirs(train_subdir, exist_ok=True)
        
        # Move files instead of copying
        train_hea_path = os.path.join(train_subdir, f"{record}.hea")
        train_dat_path = os.path.join(train_subdir, f"{record}.dat")
        
        shutil.move(hea_file, train_hea_path)
        shutil.move(dat_file, train_dat_path)
        
        # Add Chagas label to header file
        add_chagas_label_to_file(train_hea_path)
    
    # Move files to holdout directory
    for record, subdir, hea_file, dat_file in holdout_records:
        # Create subdir in holdout if it doesn't exist
        holdout_subdir = os.path.join(holdout_dir, subdir)
        os.makedirs(holdout_subdir, exist_ok=True)
        
        # Move files instead of copying
        holdout_hea_path = os.path.join(holdout_subdir, f"{record}.hea")
        holdout_dat_path = os.path.join(holdout_subdir, f"{record}.dat")
        
        shutil.move(hea_file, holdout_hea_path)
        shutil.move(dat_file, holdout_dat_path)
        
        # Add Chagas label to header file
        add_chagas_label_to_file(holdout_hea_path)
    
    print(f"PTB-XL dataset: Moved {len(train_records)} records to training and {len(holdout_records)} records to holdout")

def process_samitrop_output(src_dir, train_dir, holdout_dir, force=False):
    """Process samitrop_output directory with files directly in the directory"""
    print(f"Processing SaMi-Trop dataset...")
    
    # Copy CSV file to both directories
    csv_file = os.path.join(src_dir, 'samitrop_chagas_labels.csv')
    if os.path.exists(csv_file):
        shutil.copy2(csv_file, train_dir)
        shutil.copy2(csv_file, holdout_dir)
        print(f"Copied samitrop_chagas_labels.csv to both directories")
    else:
        print(f"Warning: samitrop_chagas_labels.csv not found in {src_dir}")
    
    # Get all files and group by base name
    files = os.listdir(src_dir)
    record_files = defaultdict(list)
    
    for file in files:
        if file.endswith('.hea') or file.endswith('.dat'):
            base_name = os.path.splitext(file)[0]
            record_files[base_name].append(file)
    
    # Check each record for both .hea and .dat files
    all_records = []
    for record, files in record_files.items():
        if len(files) == 2:  # Should have both .hea and .dat
            hea_file = os.path.join(src_dir, f"{record}.hea")
            dat_file = os.path.join(src_dir, f"{record}.dat")
            
            if os.path.exists(hea_file) and os.path.exists(dat_file):
                all_records.append((record, hea_file, dat_file))
            else:
                print(f"Warning: Inconsistency with record {record} files")
        else:
            print(f"Warning: Record {record} does not have both .hea and .dat files:")
            if f"{record}.hea" not in files:
                print(f"  - Missing {record}.hea")
            if f"{record}.dat" not in files:
                print(f"  - Missing {record}.dat")
    
    # Split records between training and holdout
    random.shuffle(all_records)
    split_idx = int(len(all_records) * 0.8)
    train_records = all_records[:split_idx]
    holdout_records = all_records[split_idx:]
    
    # Move files to training directory
    for record, hea_file, dat_file in train_records:
        shutil.move(hea_file, train_dir)
        shutil.move(dat_file, train_dir)
    
    # Move files to holdout directory
    for record, hea_file, dat_file in holdout_records:
        shutil.move(hea_file, holdout_dir)
        shutil.move(dat_file, holdout_dir)
    
    print(f"SaMi-Trop dataset: Moved {len(train_records)} records to training and {len(holdout_records)} records to holdout")

def run(args):
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Check and create directories
    dir_result = check_and_create_dirs(args.data_directory)
    if not dir_result:
        return
    
    all_datasets_dir, training_dir, holdout_dir = dir_result
    
    # Check if target directories are empty or if force flag is used
    if (os.listdir(training_dir) or os.listdir(holdout_dir)) and not args.force:
        print("Error: Training or holdout directories are not empty. Use --force to overwrite.")
        return
    
    # Process each dataset directory if it exists
    code15_dir = os.path.join(all_datasets_dir, 'code15_output')
    ptbxl_dir = os.path.join(all_datasets_dir, 'ptbxl_output')
    samitrop_dir = os.path.join(all_datasets_dir, 'samitrop_output')
    
    if os.path.isdir(code15_dir):
        process_code15_output(code15_dir, training_dir, holdout_dir, args.force)
    else:
        print(f"Warning: code15_output directory not found at {code15_dir}")
        
    if os.path.isdir(ptbxl_dir):
        process_ptbxl_output(ptbxl_dir, training_dir, holdout_dir, args.force)
    else:
        print(f"Warning: ptbxl_output directory not found at {ptbxl_dir}")
        
    if os.path.isdir(samitrop_dir):
        process_samitrop_output(samitrop_dir, training_dir, holdout_dir, args.force)
    else:
        print(f"Warning: samitrop_output directory not found at {samitrop_dir}")
    
    print("\nDataset splitting completed successfully!")

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
