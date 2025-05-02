#!/usr/bin/env python

import argparse
import os
import sys
from helper_code import load_text, save_text

def get_parser():
    description = 'Add labels to header files for testing purposes'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str, required=True, help='Folder containing header files')
    parser.add_argument('-l', '--label_value', type=int, default=0, choices=[0, 1], help='Label value to add (0 or 1)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    return parser

def add_labels_to_headers(data_folder, label_value, verbose=False):
    """Add Chagas labels to header files"""
    
    # Find all .hea files in the folder and subfolders
    header_files = []
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.hea'):
                header_files.append(os.path.join(root, file))
    
    if verbose:
        print(f"Found {len(header_files)} header files")
    
    modified_count = 0
    for header_file in header_files:
        # Read the header file
        header_content = load_text(header_file)
        
        # Check if the file already has a Chagas label
        if "# Chagas label:" in header_content:
            if verbose:
                print(f"Skipping {header_file} - already has a label")
            continue
        
        # Add the label at the end of the file
        if header_content.endswith('\n'):
            header_content += f"# Chagas label: {label_value}\n"
        else:
            header_content += f"\n# Chagas label: {label_value}\n"
        
        # Save the updated header file
        save_text(header_file, header_content)
        modified_count += 1
        
        if verbose:
            print(f"Added label {label_value} to {header_file}")
    
    return modified_count

def run(args):
    modified_count = add_labels_to_headers(args.data_folder, args.label_value, args.verbose)
    print(f"Added labels to {modified_count} header files")

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
