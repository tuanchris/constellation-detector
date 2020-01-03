import pandas as pd

def process_constl_names():
    df = pd.read_csv('./data/raw/constellationship.fab', header=None)
    df['constellation'] = df[0].str.split().str.get(0)
    df['num_pairs'] = df[0].str.split().str.get(1)
    df['stars'] = df[0].str.split().str[2:]
    df.drop(0, axis=1, inplace=True)
    display(df.head())

    df_names = pd.read_csv('./data/raw/constellation_names.eng.fab', header=None)
    df_names = df_names[0].str.replace('\t', '').str.split('"', expand=True)
    df_names.drop([2, 3, 4], axis=1, inplace=True)
    df_names.columns = ['constellation', 'name']
    display(df_names.head())

    assert len(df) == len(df_names)
    df = pd.merge(df, df_names, on="constellation")
    return df
