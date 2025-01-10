import argparse
from lift.gated_memory.utils import preprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--source_dir')
    parser.add_argument('-D', '--dest_dir')
    args = parser.parse_args()
    preprocess(args.source_dir, args.dest_dir)
