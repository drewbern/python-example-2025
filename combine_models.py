#!/usr/bin/env python

# Script to combine models trained on different data partitions

import argparse
import os
import sys
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from team_code import save_model

def get_parser():
    description = 'Combine models trained on different data partitions.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-m', '--model_folders', type=str, required=True,
                        help='Comma-separated list of model folders')
    parser.add_argument('-o', '--output_folder', type=str, required=True,
                        help='Output folder for the combined model')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser

def run(args):
    model_folders = args.model_folders.split(',')
    
    if args.verbose:
        print(f'Combining models from {len(model_folders)} folders...')
    
    # Load all models
    models = []
    for folder in model_folders:
        if os.path.exists(os.path.join(folder, 'model.sav')):
            model_data = joblib.load(os.path.join(folder, 'model.sav'))
            if 'model' in model_data:
                models.append(model_data['model'])
                if args.verbose:
                    print(f'Loaded model from {folder}')
            else:
                print(f'Warning: Missing model key in {folder}/model.sav')
        else:
            print(f'Warning: Missing model file in {folder}')
    
    if not models:
        print('Error: No models found to combine')
        return
    
    # Create a combined model by averaging predictions
    # For RandomForestClassifier, we can combine the estimators
    combined_model = models[0]
    
    # In case of RandomForestClassifier, combine all the trees
    if isinstance(combined_model, RandomForestClassifier):
        all_estimators = []
        for model in models:
            all_estimators.extend(model.estimators_)
        
        # Create a new forest with all trees
        combined_model = RandomForestClassifier()
        combined_model.estimators_ = all_estimators
        combined_model.n_estimators = len(all_estimators)
        
        # Copy other parameters from the first model
        for attr in ['classes_', 'n_classes_', 'n_features_in_', 'feature_names_in_']:
            if hasattr(models[0], attr):
                setattr(combined_model, attr, getattr(models[0], attr))
    
    # Make sure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Save the combined model
    save_model(args.output_folder, combined_model)
    
    if args.verbose:
        print(f'Combined model saved to {args.output_folder}')

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
