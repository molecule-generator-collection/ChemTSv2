import argparse
import ast
import os

import pandas as pd


def get_parser():
    parser = argparse.ArgumentParser(
        description="",
        usage="chemtsv2-column-splitter -i INPUT_FILE -t TARGET_COL_NAME -n NEW_COL_NAMES01 NEW_COL_NAMES02 ...",
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help="path to a result CSV file",
    )
    parser.add_argument(
        "-t",
        "--target_column",
        type=str,
        required=True,
        help="target column name to be split",
    )
    parser.add_argument(
        "-n",
        "--new_column_names",
        type=str,
        nargs="*",
        required=True,
        help="new column names to be created from the target column",
    )
    return parser.parse_args()


def main():
    args = get_parser()
    print(
        f"[INFO] Input file: {args.input_file}\n"
        f"[INFO] Target column name: {args.target_column}\n"
        f"[INFO] New column names: {args.new_column_names}\n"
    )

    df = pd.read_csv(args.input_file)
    df[args.target_column] = df[args.target_column].apply(ast.literal_eval)
    df[args.new_column_names] = pd.DataFrame(df[args.target_column].tolist(), index=df.index)
    df.drop(args.target_column, axis=1, inplace=True)

    stem, ext = os.path.splitext(args.input_file)
    output_fname = f"{stem}_column_{args.target_column}_split{ext}"
    df.to_csv(output_fname, mode="w", index=False)
    print(f"[INFO] Save to {output_fname}\n[INFO] Done!")


if __name__ == "__main__":
    main()
