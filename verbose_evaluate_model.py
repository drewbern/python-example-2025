#!/usr/bin/env python

import argparse
import numpy as np
import os
import os.path
import sys

from helper_code import *

# Parse arguments.
def get_parser():
    description = 'Evaluate the Challenge model with verbose output.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-s', '--score_file', type=str, required=False)
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode - show which files are causing issues')
    parser.add_argument('-i', '--ignore_missing', action='store_true', help='Ignore missing labels and continue evaluation')
    return parser

# Evaluate the models.
def evaluate_model(data_folder, output_folder, verbose=False, ignore_missing=False):
    # Find the records.
    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No records found.')

    labels = []
    binary_outputs = []
    probability_outputs = []
    skipped_records = []

    # Load the labels and model outputs.
    for i, record in enumerate(records):
        label_filename = os.path.join(data_folder, record)
        
        # Try to load the label, but allow it to be missing if specified
        try:
            header = load_header(label_filename)
            label_value, has_label = get_variable(header, label_string)
            
            if not has_label:
                if verbose:
                    print(f"Missing label in record {record}")
                if ignore_missing:
                    skipped_records.append(record)
                    continue
                else:
                    raise Exception(f'No label is available in {record}: are you trying to load the labels from the held-out data?')
                    
            label = sanitize_boolean_value(label_value)
            
            output_filename = os.path.join(output_folder, record + '.txt')
            if not os.path.exists(output_filename):
                if verbose:
                    print(f"Missing output file for record {record}")
                if ignore_missing:
                    skipped_records.append(record)
                    continue
                else:
                    raise FileNotFoundError(f"Missing output file: {output_filename}")
                
            output = load_text(output_filename)
            binary_output = get_label(output, allow_missing=True)
            probability_output = get_probability(output, allow_missing=True)

            # Missing model outputs are interpreted as zero
            labels.append(label)
            binary_outputs.append(0 if is_nan(binary_output) else binary_output)
            probability_outputs.append(0 if is_nan(probability_output) else probability_output)
            
            # Print successful evaluation
            if verbose:
                print(f"Successfully evaluated record {i+1}/{num_records}: {record}")
            
        except Exception as e:
            if verbose:
                print(f"Error processing record {record}: {str(e)}")
            if not ignore_missing:
                raise

    if verbose:
        successful_count = len(labels)
        print(f"Successfully evaluated {successful_count} out of {num_records} records")
        if skipped_records:
            print(f"Skipped {len(skipped_records)} records due to missing labels or outputs")
    
    if len(labels) == 0:
        raise ValueError("No valid records found with labels for evaluation")

    # Convert to numpy arrays for evaluation
    labels = np.array(labels)
    binary_outputs = np.array(binary_outputs)
    probability_outputs = np.array(probability_outputs)

    # Evaluate the model outputs.
    challenge_score = compute_challenge_score(labels, probability_outputs)
    auroc, auprc = compute_auc(labels, probability_outputs)
    accuracy = compute_accuracy(labels, binary_outputs)
    f_measure = compute_f_measure(labels, binary_outputs)

    # Compute and print the confusion matrix
    confusion_matrix = compute_confusion_matrix(labels, binary_outputs)
    if verbose:
        print("\nConfusion Matrix:")
        print("[TP  FP]")
        print("[FN  TN]")
        print(confusion_matrix)
        print(f"True Positives: {confusion_matrix[0,0]}")
        print(f"False Positives: {confusion_matrix[0,1]}")
        print(f"False Negatives: {confusion_matrix[1,0]}")
        print(f"True Negatives: {confusion_matrix[1,1]}")

    return challenge_score, auroc, auprc, accuracy, f_measure, len(skipped_records), confusion_matrix

# Run the code.
def run(args):
    # Compute the scores for the model outputs.
    try:
        challenge_score, auroc, auprc, accuracy, f_measure, skipped, confusion_matrix = evaluate_model(
            args.data_folder, args.output_folder, args.verbose, args.ignore_missing)

        output_string = \
            f'Challenge score: {challenge_score:.3f}\n' + \
            f'AUROC: {auroc:.3f}\n' \
            f'AUPRC: {auprc:.3f}\n' + \
            f'Accuracy: {accuracy:.3f}\n' \
            f'F-measure: {f_measure:.3f}\n'
        
# Add confusion matrix to output
        output_string += f'\nConfusion Matrix:\n'
        output_string += f'[TP={confusion_matrix[0,0]}  FP={confusion_matrix[0,1]}]\n'
        output_string += f'[FN={confusion_matrix[1,0]}  TN={confusion_matrix[1,1]}]\n'
        
        if skipped > 0:
            output_string += f'Skipped records: {skipped}\n'

        # Output the scores to screen and/or a file.
        if args.score_file:
            save_text(args.score_file, output_string)
        else:
            print(output_string)
    
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
