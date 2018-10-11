#!/usr/bin/env python3
import os
import sys
import json
import argparse

from surprise import SVD
from surprise import Dataset

from surprise.model_selection import cross_validate

from surprise.prediction_algorithms.knns import KNNWithMeans

def process_args(args=None):
    parser = argparse.ArgumentParser(description="Run and output statistics on SVD recommendations with MovieLens")
    parser.add_argument('-l', '--location', help="Output directory for results")
    args = parser.parse_args()

    return args.location

def write_results_to_file(rmse, mae, title):
    avg_rmse = sum(rmse) / len(rmse)
    avg_mae = sum(mae) / len(mae)
    json.dump({'rmse': avg_rmse, 'mae': avg_mae}, open(title, 'w'))

def main(args=None):
    location = process_args(args)

    out_path = os.path.expanduser(location)
    print('Checking output directory...')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    else:
        ans = input("Overwrite output directory?: ").upper()
        if ans == 'N' or ans == 'NO':
            print('Exiting...')
            exit()
    print("Loading dataset...")
    data = Dataset.load_builtin('ml-1m')
    algo = SVD()
    print("Running SVD...")
    result = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10, verbose=True)
    write_results_to_file(result['test_rmse'], result['test_mae'], 'svd_out.json')
    print("Running KNN...")
    algo = KNNWithMeans()
    result = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10, verbose=True)
    write_results_to_file(result['test_rmse'], result['test_mae'], 'knn_out.json')
    print("Done.")


if __name__ == '__main__':
    main(args)
