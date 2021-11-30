from typing import List, Tuple, Dict, Union, Any
import pandas as pd


def load_data(path_real: str,
              path_fake: str,
              real_sep: str = ',',
              fake_sep: str = ',',
              drop_columns: List = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from a real and synthetic data csv. This function makes sure that the loaded data has the same columns
    with the same data types.

    :param path_real: string path to csv with real data
    :param path_fake: string path to csv with real data
    :param real_sep: separator of the real csv
    :param fake_sep: separator of the fake csv
    :param drop_columns: names of columns to drop.
    :return: Tuple with DataFrame containing the real data and DataFrame containing the synthetic data.
    """
    real = pd.read_csv(path_real, sep=real_sep, low_memory=False)
    fake = pd.read_csv(path_fake, sep=fake_sep, low_memory=False)
    if set(fake.columns.tolist()).issubset(set(real.columns.tolist())):
        real = real[fake.columns]
    elif drop_columns is not None:
        real = real.drop(drop_columns, axis=1)
        try:
            fake = fake.drop(drop_columns, axis=1)
        except:
            print(f'Some of {drop_columns} were not found on fake.index.')
        assert len(fake.columns.tolist()) == len(real.columns.tolist()), \
            f'Real and fake do not have same nr of columns: {len(fake.columns)} and {len(real.columns)}'
        fake.columns = real.columns
    else:
        fake.columns = real.columns

    for col in fake.columns:
        fake[col] = fake[col].astype(real[col].dtype)
    return real, fake


def dict_to_df(data: Dict[str, Any]):
    return pd.DataFrame(
        {'result': list(data.values())},
        index=list(data.keys())
    )
