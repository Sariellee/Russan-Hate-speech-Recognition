import os

from pandas import DataFrame, read_csv, Series

INPUT_FILE = 'south_park.csv'
OUTPUT_FILE = '../processed_separate/out_south_park.csv'
SAVE_EVERY = 10


def clear():
    os.system('clear')


if __name__ == "__main__":
    df = read_csv(INPUT_FILE, delimiter=";")

    if OUTPUT_FILE in os.listdir('../..'):
        with open(OUTPUT_FILE, 'r') as f:
            count = sum(1 for _ in f) - 1
        new_df = read_csv(OUTPUT_FILE, header=None, delimiter=';')
    else:
        new_df = DataFrame(columns=['text', 'sentiment', 'hate_speech'])
        count = 0
    i = count
    while i < len(df):
        row = df.iloc[i]
        clear()
        print(f'Total samples: {i}')
        if i % SAVE_EVERY == 0 and i != count:
            new_df.to_csv(OUTPUT_FILE, index=False, sep=';')
            print('saved!')
        else:
            print(f'{SAVE_EVERY - i % SAVE_EVERY} till save')

        print('\n')
        print(row['content'])
        print('\n')

        inp = input('hate speech? (0-3)    ')
        if inp == "prev":
            if i >= 2:
                i = i - 1
                continue

        hate_speech = None
        while hate_speech is None:
            try:
                hate_speech = int(inp)
            except ValueError:
                inp = input("Wrong input, try again      ")

        inp = input('sentiment? (-1/0/1)    ')
        if inp == "prev":
            if i >= 2:
                i = i - 1
                continue

        sentiment = None
        while sentiment is None:
            try:
                sentiment = int(inp)
            except ValueError:
                inp = input("Wrong input, try again      ")

        new_df.loc[i] = [row['content'].replace('\n', ' '), sentiment, hate_speech]

        i += 1
