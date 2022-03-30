import pandas as pd


def main():
    df_ref = pd.read_csv('data/result_for_debug.csv')
    df = pd.read_csv('result/example01/result_C1.0.csv')

    df_ref.drop('elapsed_time', axis=1, inplace=True)
    df.drop('elapsed_time', axis=1, inplace=True)

    if df.equals(df_ref):
        print('[INFO] OK!')
    else:
        print('[INFO] False, check your modifications.')


if __name__ == "__main__":
    main()

