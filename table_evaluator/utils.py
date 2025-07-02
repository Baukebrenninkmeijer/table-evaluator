from typing import Any, Dict, List, Optional, Tuple, Union
from os import PathLike

import pandas as pd

from .backends import (
    BackendManager,
    BackendType,
    DataFrameWrapper,
    is_backend_available,
    is_feature_enabled,
)


def load_data(
    path_real: str,
    path_fake: str,
    real_sep: str = ",",
    fake_sep: str = ",",
    drop_columns: Optional[List] = None,
    backend: Optional[Union[str, BackendType]] = None,
    auto_detect_backend: bool = True,
    lazy: bool = False,
    **kwargs,
) -> Tuple[
    Union[pd.DataFrame, DataFrameWrapper], Union[pd.DataFrame, DataFrameWrapper]
]:
    """
    Load data from real and synthetic data files with automatic backend detection and optimization.

    This function supports both pandas and Polars backends, automatically choosing the optimal
    backend based on file format, size, and availability. It ensures that the loaded data has
    the same columns with the same data types.

    Args:
        path_real: Path to file with real data
        path_fake: Path to file with synthetic data
        real_sep: Separator for the real data file
        fake_sep: Separator for the synthetic data file
        drop_columns: Names of columns to drop
        backend: Backend to use ("pandas", "polars", "auto", or BackendType)
        auto_detect_backend: Whether to auto-detect optimal backend
        lazy: Whether to use lazy evaluation (Polars LazyFrame)
        **kwargs: Additional arguments for backend loaders

    Returns:
        Tuple with DataFrames containing real and synthetic data

    Examples:
        # Auto-detect optimal backend
        real, fake = load_data("real.csv", "fake.csv")

        # Force specific backend
        real, fake = load_data("real.parquet", "fake.parquet", backend="polars")

        # Use lazy evaluation for large files
        real, fake = load_data("large_real.csv", "large_fake.csv", lazy=True)
    """
    # Initialize backend manager
    if is_feature_enabled("automatic_backend_detection") and auto_detect_backend:
        manager = BackendManager()
    else:
        manager = BackendManager(default_backend=BackendType.PANDAS)

    # Convert string backend to BackendType
    if isinstance(backend, str):
        if backend.lower() == "pandas":
            backend = BackendType.PANDAS
        elif backend.lower() == "polars":
            backend = BackendType.POLARS
        elif backend.lower() == "auto":
            backend = BackendType.AUTO
        else:
            raise ValueError(f"Unknown backend: {backend}")

    # Load real data
    real_wrapper = manager.load_data(
        path_real,
        backend=backend,
        auto_detect=auto_detect_backend,
        sep=real_sep,
        lazy=lazy,
        **kwargs,
    )

    # Load fake data
    fake_wrapper = manager.load_data(
        path_fake,
        backend=backend,
        auto_detect=auto_detect_backend,
        sep=fake_sep,
        lazy=lazy,
        **kwargs,
    )

    # Extract underlying DataFrames for compatibility processing
    if isinstance(real_wrapper, DataFrameWrapper):
        real = real_wrapper.to_pandas()  # Convert to pandas for processing
        fake = fake_wrapper.to_pandas()
    else:
        real = real_wrapper
        fake = fake_wrapper
    # Ensure column compatibility (existing logic)
    if set(fake.columns.tolist()).issubset(set(real.columns.tolist())):
        real = real[fake.columns]
    elif drop_columns is not None:
        real = real.drop(columns=drop_columns)
        try:
            fake = fake.drop(columns=drop_columns)
        except KeyError:
            raise ValueError(
                f"Some of {drop_columns} were not found in fake data columns."
            )
        if len(fake.columns.tolist()) != len(real.columns.tolist()):
            raise ValueError(
                f"Real and fake do not have same number of columns: {len(fake.columns)} and {len(real.columns)}"
            )
        fake.columns = real.columns
    else:
        fake.columns = real.columns

    # Ensure compatible data types
    for col in fake.columns:
        fake[col] = fake[col].astype(real[col].dtype)

    # Return format based on backend configuration
    if backend in [BackendType.POLARS] and is_backend_available("polars"):
        # Convert back to original backend format if Polars was used
        from .backends import convert_to_polars

        real_result = convert_to_polars(real, lazy=lazy)
        fake_result = convert_to_polars(fake, lazy=lazy)

        return DataFrameWrapper(real_result), DataFrameWrapper(fake_result)
    else:
        # Return pandas DataFrames for backward compatibility
        return real, fake


def load_data_with_backend(
    path: Union[str, PathLike],
    backend: Optional[Union[str, BackendType]] = None,
    auto_detect: bool = True,
    lazy: bool = False,
    **kwargs,
) -> DataFrameWrapper:
    """
    Load single data file with backend detection and optimization.

    Args:
        path: Path to data file
        backend: Backend to use ("pandas", "polars", "auto", or BackendType)
        auto_detect: Whether to auto-detect optimal backend
        lazy: Whether to use lazy evaluation (Polars LazyFrame)
        **kwargs: Additional arguments for backend loaders

    Returns:
        DataFrameWrapper containing the loaded data

    Examples:
        # Auto-detect optimal backend
        data = load_data_with_backend("data.csv")

        # Force Polars with lazy evaluation
        data = load_data_with_backend("large_data.parquet", backend="polars", lazy=True)
    """
    # Initialize backend manager
    if is_feature_enabled("automatic_backend_detection") and auto_detect:
        manager = BackendManager()
    else:
        manager = BackendManager(default_backend=BackendType.PANDAS)

    # Convert string backend to BackendType
    if isinstance(backend, str):
        if backend.lower() == "pandas":
            backend = BackendType.PANDAS
        elif backend.lower() == "polars":
            backend = BackendType.POLARS
        elif backend.lower() == "auto":
            backend = BackendType.AUTO
        else:
            raise ValueError(f"Unknown backend: {backend}")

    # Load data
    return manager.load_data(path, backend=backend, auto_detect=auto_detect, **kwargs)


def dict_to_df(data: Dict[str, Any]):
    return pd.DataFrame({"result": list(data.values())}, index=list(data.keys()))


def _preprocess_data(
    real: Union[pd.DataFrame, DataFrameWrapper],
    fake: Union[pd.DataFrame, DataFrameWrapper],
    cat_cols=None,
    unique_thresh=0,
    n_samples=None,
    seed=1337,
    backend: Optional[Union[str, BackendType]] = None,
    optimize_for_backend: bool = True,
) -> Tuple[
    Union[pd.DataFrame, DataFrameWrapper],
    Union[pd.DataFrame, DataFrameWrapper],
    List[str],
    List[str],
]:
    """
    Preprocess real and fake data with backend-agnostic operations.

    This function supports both pandas DataFrames and DataFrameWrappers with
    backend-specific optimizations for efficient preprocessing.

    Args:
        real: Real dataset (pandas DataFrame or DataFrameWrapper)
        fake: Synthetic dataset (pandas DataFrame or DataFrameWrapper)
        cat_cols: Explicit categorical columns (if None, auto-inferred)
        unique_thresh: Threshold for numerical vs categorical classification
        n_samples: Number of samples to take (if None, uses minimum length)
        seed: Random seed for sampling
        backend: Preferred backend for processing
        optimize_for_backend: Whether to apply backend-specific optimizations

    Returns:
        Tuple of (real_data, fake_data, numerical_columns, categorical_columns)
    """
    # Initialize backend manager if optimizations are enabled
    if optimize_for_backend and is_feature_enabled("automatic_backend_detection"):
        from .backends import BackendManager, BackendContext

        manager = BackendManager()
    else:
        manager = None

    # Handle DataFrameWrapper inputs
    if isinstance(real, DataFrameWrapper):
        real_backend_type = real.backend_type
        real_df = real.underlying_df
        real_is_wrapper = True
    else:
        real_backend_type = BackendType.PANDAS
        real_df = real
        real_is_wrapper = False

    if isinstance(fake, DataFrameWrapper):
        fake_backend_type = fake.backend_type
        fake_df = fake.underlying_df
        fake_is_wrapper = True
    else:
        fake_backend_type = BackendType.PANDAS
        fake_df = fake
        fake_is_wrapper = False

    # Determine target backend for processing
    if backend is not None:
        if isinstance(backend, str):
            target_backend = BackendType[backend.upper()]
        else:
            target_backend = backend
    else:
        # Use the backend of the first wrapper, or pandas as fallback
        target_backend = real_backend_type if real_is_wrapper else fake_backend_type

    # Convert to pandas for processing if needed (for compatibility)
    if real_is_wrapper:
        real_pandas = real.to_pandas()
    else:
        real_pandas = real_df

    if fake_is_wrapper:
        fake_pandas = fake.to_pandas()
    else:
        fake_pandas = fake_df

    # Make sure columns and their order are the same
    if set(real_pandas.columns.tolist()) != set(fake_pandas.columns.tolist()):
        raise ValueError("Columns in real and fake dataframe are not the same")

    if len(real_pandas.columns) == len(fake_pandas.columns):
        # Apply identical ordering
        fake_pandas = fake_pandas[real_pandas.columns.tolist()]

    # Infer data types and column classification
    if cat_cols is None:
        real_pandas = real_pandas.infer_objects()
        fake_pandas = fake_pandas.infer_objects()

        numerical_columns = [
            column
            for column in real_pandas.select_dtypes(include="number").columns
            if len(real_pandas[column].unique()) > unique_thresh
        ]
        categorical_columns = [
            column for column in real_pandas.columns if column not in numerical_columns
        ]
    else:
        categorical_columns = cat_cols
        numerical_columns = [
            column for column in real_pandas.columns if column not in cat_cols
        ]

    # Determine sample size
    if n_samples is None:
        n_samples = min(len(real_pandas), len(fake_pandas))
    elif len(fake_pandas) >= n_samples and len(real_pandas) >= n_samples:
        n_samples = n_samples
    else:
        raise Exception(
            f"Make sure n_samples < len(fake/real). len(real): {len(real_pandas)}, len(fake): {len(fake_pandas)}"
        )

    # Apply backend-specific optimizations for sampling
    if (
        optimize_for_backend
        and manager
        and target_backend == BackendType.POLARS
        and is_backend_available("polars")
    ):
        # Use Polars for efficient sampling if available
        with BackendContext.use_backend(BackendType.POLARS) as polars_backend:
            # Convert to Polars for optimized operations
            from .backends import convert_to_polars

            real_polars = convert_to_polars(real_pandas, lazy=False)
            fake_polars = convert_to_polars(fake_pandas, lazy=False)

            # Perform sampling with Polars
            real_sampled = polars_backend.sample(real_polars, n_samples, seed)
            fake_sampled = polars_backend.sample(fake_polars, n_samples, seed)

            # Convert back to pandas for fillna operations (which are more mature in pandas)
            real_pandas = polars_backend.to_pandas(real_sampled)
            fake_pandas = polars_backend.to_pandas(fake_sampled)
    else:
        # Use pandas sampling
        real_pandas = real_pandas.sample(n_samples, random_state=seed).reset_index(
            drop=True
        )
        fake_pandas = fake_pandas.sample(n_samples, random_state=seed).reset_index(
            drop=True
        )

    assert len(real_pandas) == len(fake_pandas), "len(real) != len(fake)"

    # Fill missing values
    # Categorical columns: fill with "[NAN]"
    if categorical_columns:
        # Convert categorical columns back to object type to allow arbitrary fillna values
        for col in categorical_columns:
            if col in real_pandas.columns and real_pandas[col].dtype.name == "category":
                real_pandas[col] = real_pandas[col].astype("object")
            if col in fake_pandas.columns and fake_pandas[col].dtype.name == "category":
                fake_pandas[col] = fake_pandas[col].astype("object")

        real_pandas.loc[:, categorical_columns] = real_pandas.loc[
            :, categorical_columns
        ].fillna("[NAN]")
        fake_pandas.loc[:, categorical_columns] = fake_pandas.loc[
            :, categorical_columns
        ].fillna("[NAN]")

    # Numerical columns: fill with mean
    if numerical_columns:
        real_means = real_pandas[numerical_columns].mean()
        fake_means = fake_pandas[numerical_columns].mean()

        real_pandas.loc[:, numerical_columns] = real_pandas.loc[
            :, numerical_columns
        ].fillna(real_means)
        fake_pandas.loc[:, numerical_columns] = fake_pandas.loc[
            :, numerical_columns
        ].fillna(fake_means)

    # Return in appropriate format
    if optimize_for_backend and (real_is_wrapper or fake_is_wrapper):
        # Convert back to target backend if wrappers were used
        if target_backend == BackendType.POLARS and is_backend_available("polars"):
            from .backends import convert_to_polars

            real_result = DataFrameWrapper(convert_to_polars(real_pandas, lazy=False))
            fake_result = DataFrameWrapper(convert_to_polars(fake_pandas, lazy=False))
        else:
            real_result = DataFrameWrapper(real_pandas)
            fake_result = DataFrameWrapper(fake_pandas)

        return real_result, fake_result, numerical_columns, categorical_columns
    else:
        # Return pandas DataFrames for backward compatibility
        return real_pandas, fake_pandas, numerical_columns, categorical_columns
