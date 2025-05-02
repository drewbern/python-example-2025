import os
import shutil
import random
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Move half of WFDB file pairs (.hea and .dat) from source to destination directory.')
    parser.add_argument('source_dir', help='Source directory containing .hea and .dat files')
    parser.add_argument('dest_dir', help='Destination directory for moved files')
    args = parser.parse_args()
    
    source_dir = args.source_dir
    dest_dir = args.dest_dir
    
    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    
    # Find all .hea files
    hea_files = [f for f in os.listdir(source_dir) if f.endswith('.hea')]
    
    # Check for matching .dat files and create pairs
    valid_pairs = []
    for hea_file in hea_files:
        base_name = os.path.splitext(hea_file)[0]
        dat_file = f"{base_name}.dat"
        if os.path.exists(os.path.join(source_dir, dat_file)):
            valid_pairs.append((hea_file, dat_file))
    
    if not valid_pairs:
        print("No valid .hea/.dat file pairs found in the source directory.")
        return
    
    # Select half of the pairs randomly
    num_to_move = max(1, len(valid_pairs) // 2)
    pairs_to_move = random.sample(valid_pairs, num_to_move)
    
    # Move the selected pairs
    moved_count = 0
    for hea_file, dat_file in pairs_to_move:
        try:
            shutil.move(os.path.join(source_dir, hea_file), os.path.join(dest_dir, hea_file))
            shutil.move(os.path.join(source_dir, dat_file), os.path.join(dest_dir, dat_file))
            moved_count += 1
            print(f"Moved: {hea_file} and {dat_file}")
        except Exception as e:
            print(f"Error moving {hea_file}/{dat_file}: {e}")
    
    print(f"\nMoved {moved_count} file pairs out of {len(valid_pairs)} total pairs.")
    print(f"Source directory: {source_dir}")
    print(f"Destination directory: {dest_dir}")

if __name__ == "__main__":
    main()
