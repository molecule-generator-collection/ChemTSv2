import argparse
import sys

import pandas as pd


def get_parser():
    parser = argparse.ArgumentParser(
        description="", usage="chemtsv2-debug-check"
    )
    parser.add_argument(
        "--long_generation",
        action="store_true",
        help="Use this flag if you want to compare the result of a long generation process",
    )
    return parser.parse_args()


def compare_result_with_reference(input_file, reference_file):
    df_ref = pd.read_csv(reference_file)
    df = pd.read_csv(input_file)

    df_ref.drop("elapsed_time", axis=1, inplace=True)
    df.drop("elapsed_time", axis=1, inplace=True)

    df_ref["reward"] = df_ref["reward"].round(decimals=10)
    df["reward"] = df["reward"].round(decimals=10)
    if df.equals(df_ref):
        print("[INFO] Output validation successful")
        sys.exit(0)
    else:
        print("[ERROR] Output validation failed. Please review your changes.")
        pd.set_option("display.float_format", lambda x: f"{x:.20f}")
        print(df.compare(df_ref))
        sys.exit(1)


def main():
    args = get_parser()
    if args.long_generation:
        # generation_num: 1500
        # Check the process of handling duplications in chemtsv2.utils.evaluate_node()
        compare_result_with_reference(
            input_file="result/example01/result_C1.0.csv",
            reference_file="data/result_for_debug_long.csv",
        )
    else:
        # generation_num: 300
        # Default setting
        compare_result_with_reference(
            input_file="result/example01/result_C1.0.csv",
            reference_file="data/result_for_debug.csv",
        )


if __name__ == "__main__":
    main()
