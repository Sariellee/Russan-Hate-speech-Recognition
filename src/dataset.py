from pandas import DataFrame, read_csv


def print_proportions(df):
    print()
    print(f"label 0: {len(df.loc[df['hate_speech'] == 0])} samples")
    print(f"label 1: {len(df.loc[df['hate_speech'] == 1])} samples")

    proportion_0 = round(len(df.loc[df['hate_speech'] == 0]) / (
            len(df.loc[df['hate_speech'] == 0]) + len(df.loc[df['hate_speech'] == 1])), 2)
    propotion_1 = round(1 - proportion_0, 2)

    print(f"data proportion is: {proportion_0}/{propotion_1}")
    print()


def balance_dataset(df: DataFrame) -> DataFrame:
    if len(df[df['hate_speech'] == 0]) > len(df[df['hate_speech'] == 1]):
        indexes = list(df[df['hate_speech'] == 0][:len(df[df['hate_speech'] == 1])].index)
    else:
        indexes = list(df[df['hate_speech'] == 1][:len(df[df['hate_speech'] == 0])].index)

    df = df.drop(indexes)

    print("AFTER REBALANCING")
    print_proportions(df)

    return df


def read_dataset(datapath: str = 'data/final_data.csv') -> DataFrame:
    df = read_csv(datapath, delimiter=';')
    print_proportions(df)

    return df
