import pandas as pd

from table_evaluator.utils import dict_to_df, load_data


def test_load_data(mocker):
    # Test loading real and fake data with matching columns
    real_data = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    fake_data = pd.DataFrame({"A": [5, 6], "B": [7, 8]})

    # Mocking pd.read_csv
    mocker.patch(
        "pandas.read_csv",
        side_effect=lambda path, sep, low_memory: real_data
        if "real" in path
        else fake_data,
    )

    real, fake = load_data("path/to/real.csv", "path/to/fake.csv")

    assert real.equals(real_data)
    assert fake.equals(fake_data)


def test_dict_to_df():
    # Test converting a dictionary to a DataFrame
    data = {"key1": 1, "key2": 2}
    expected_df = pd.DataFrame({"result": [1, 2]}, index=["key1", "key2"])

    result_df = dict_to_df(data)

    pd.testing.assert_frame_equal(result_df, expected_df)
