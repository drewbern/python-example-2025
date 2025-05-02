#!/usr/bin/env python

import argparse
import os

def get_parser():
    description = 'Inspect a header file'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-f', '--header_file', type=str, required=True, help='Path to the header file')
    return parser

def run(args):
    if not os.path.exists(args.header_file):
        print(f"Error: File not found: {args.header_file}")
        return
    
    with open(args.header_file, 'r') as f:
        content = f.read()
    
    print("=== Header File Content ===")
    print(content)
    print("=========================")
    
    if "# Chagas label:" in content:
        for line in content.split('\n'):
            if line.startswith("# Chagas label:"):
                print(f"Found label: {line}")
                break
    else:
        print("No Chagas label found in this file")

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
