from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def load_data(
    path_real: str,
    path_fake: str,
    real_sep: str = ",",
    fake_sep: str = ",",
    drop_columns: Optional[List] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        real = real.drop(column=drop_columns)
        try:
            fake = fake.drop(columns=drop_columns)
        except KeyError:
            ValueError(f"Some of {drop_columns} were not found on fake.index.")
        if len(fake.columns.tolist()) != len(real.columns.tolist()):
            raise ValueError(
                f"Real and fake do not have same nr of columns: {len(fake.columns)} and {len(real.columns)}"
            )
        fake.columns = real.columns
    else:
        fake.columns = real.columns

    for col in fake.columns:
        fake[col] = fake[col].astype(real[col].dtype)
    return real, fake


def dict_to_df(data: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame({"result": list(data.values())}, index=list(data.keys()))


def _preprocess_data(
    real: pd.DataFrame,
    fake: pd.DataFrame,
    cat_cols=None,
    unique_thresh=0,
    n_samples=None,
    seed=1337,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    # Make sure columns and their order are the same.
    if set(real.columns.tolist()) != set(fake.columns.tolist()):
        raise ValueError("Columns in real and fake dataframe are not the same")
    if len(real.columns) == len(fake.columns):  # Apply identical ordering.
        fake = fake[real.columns.tolist()]

    if cat_cols is None:
        real = real.infer_objects()
        fake = fake.infer_objects()
        numerical_columns = [
            column
            for column in real.select_dtypes(include="number").columns
            if len(real[column].unique()) > unique_thresh
        ]
        categorical_columns = [
            column for column in real.columns if column not in numerical_columns
        ]
    else:
        categorical_columns = cat_cols
        numerical_columns = [
            column for column in real.columns if column not in cat_cols
        ]

    # Make sure the number of samples is equal in both datasets.
    if n_samples is None:
        n_samples = min(len(real), len(fake))
    elif len(fake) >= n_samples and len(real) >= n_samples:
        n_samples = n_samples
    else:
        raise Exception(
            f"Make sure n_samples < len(fake/real). len(real): {len(real)}, len(fake): {len(fake)}"
        )

    real = real.sample(n_samples, random_state=seed).reset_index(drop=True)
    fake = fake.sample(n_samples, random_state=seed).reset_index(drop=True)
    assert len(real) == len(fake), "len(real) != len(fake)"

    real.loc[:, categorical_columns] = real.loc[:, categorical_columns].fillna("[NAN]")
    fake.loc[:, categorical_columns] = fake.loc[:, categorical_columns].fillna("[NAN]")

    real.loc[:, numerical_columns] = real.loc[:, numerical_columns].fillna(
        real[numerical_columns].mean()
    )
    fake.loc[:, numerical_columns] = fake.loc[:, numerical_columns].fillna(
        fake[numerical_columns].mean()
    )
    return real, fake, numerical_columns, categorical_columns
