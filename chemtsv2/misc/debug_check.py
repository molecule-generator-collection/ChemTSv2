import sys

import pandas as pd


def main():
    df_ref = pd.read_csv('data/result_for_debug.csv')
    df = pd.read_csv('result/example01/result_C1.0.csv')

    df_ref.drop('elapsed_time', axis=1, inplace=True)
    df.drop('elapsed_time', axis=1, inplace=True)

    df_ref['reward'] = df_ref['reward'].round(decimals=10)
    df['reward'] = df['reward'].round(decimals=10)
    if df.equals(df_ref):
        print('[INFO] Output validation successful')
        sys.exit(0)
    else:
        print('[ERROR] Output validation failed. Please review your changes.')
        pd.set_option('display.float_format', lambda x: f'{x:.20f}')
        print(df.compare(df_ref))
        sys.exit(1)


if __name__ == "__main__":
    main()

