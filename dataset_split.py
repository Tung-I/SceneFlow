import logging
import argparse
import csv
import numpy as np
from pathlib import Path


def main(args):
    data_dir = args.data_dir

    paths = sorted(list(data_dir.glob('*')))

    num_paths = len(paths)

    random_seed = 0
    np.random.seed(int(random_seed))
    np.random.shuffle(paths)

    csv_path = str(args.output_path)
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for idx, p in enumerate(paths):
            # if idx < int(num_paths * 0.9):
            #     writer.writerow([str(p.parts[-1]), 'Training'])
            # else:
            #     writer.writerow([str(p.parts[-1]), 'Validation'])
            if 'TRAIN' in p.parts[-1]:
                writer.writerow([str(p.parts[-1]), 'Training'])
            else:
                writer.writerow([str(p.parts[-1]), 'Validation'])


def _parse_args():
    parser = argparse.ArgumentParser(description="To create the split.csv.")
    parser.add_argument('data_dir', type=Path,
                        help='The directory of the data.')
    parser.add_argument('output_path', type=Path,
                        help='The output path of the csv file.')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)