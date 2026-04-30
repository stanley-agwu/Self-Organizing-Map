from os.path import join
import numpy as np
import pandas as pd
import glob as gb
import os


# read all the csv files
def read_data(dataroot, file_pattern="*.csv"):
    """
    Read and concatenate all CSV files in a folder.

    Returns:
        pandas.DataFrame
    """

    if file_pattern is None:
        raise ValueError("Please specify file pattern for reading files.")

    search_path = join(dataroot, file_pattern)
    print(f"Reading files from: {search_path}")

    filenames = gb.glob(search_path)

    if len(filenames) == 0:
        raise FileNotFoundError(f"No files found in {dataroot} matching {file_pattern}")

    print(f"Found {len(filenames)} files")

    dataframes = [pd.read_csv(filename, dtype=object) for filename in filenames]

    combined_data = pd.concat(dataframes, ignore_index=True, sort=False)

    print(f"Combined shape: {combined_data.shape}")
    return combined_data


def transform_data(dataframe):
    """
    Transform absolute joint coordinates into relative coordinates.

    Uses joint 1, columns [3, 4, 5], as the reference point.
    Assumes data is arranged as:
        x0, y0, z0, x1, y1, z1, ...
    """
    data_array = dataframe.values

    # Remove rows that still contain header-like strings such as 'x0'
    header_row_indices = [
        row_index for row_index, row in enumerate(data_array) if "x0" in row
    ]

    data_array = np.delete(data_array, header_row_indices, axis=0)

    # Convert to float and scale coordinates
    data_array = 1000 * data_array.astype(float)
    transformed_rows = []

    for posture_vector in data_array:
        reference_point = posture_vector[3:6]
        relative_posture = []

        for coordinate_index in range(0, len(posture_vector), 3):
            joint_point = posture_vector[coordinate_index : coordinate_index + 3]

            relative_joint_point = joint_point - reference_point
            relative_posture.extend(relative_joint_point)

        transformed_rows.append(relative_posture)
    transformed_data = pd.DataFrame(transformed_rows)

    return transformed_data


def load_training_data(datafile):
    """
    Load and preprocess training posture data from multiple CSV sources.

    This function reads raw posture data from a specified directory, typically
    from multiple camera views (e.g., camera 1 and camera 2), and performs a
    complete preprocessing pipeline to prepare the data for SOM training.

    Processing steps include:
        - Reading and concatenating CSV files from different sources
        - Stripping whitespace from column names
        - Dropping irrelevant joints and metadata columns
        - Converting all values to numeric format
        - Removing rows with invalid or missing values
        - Applying coordinate transformation to relative joint positions
        - Handling missing values (NaNs) using mean, forward-fill, and backward-fill
        - Replacing missing feature markers (-1) with 0
        - Normalizing features using zero-centered scaling
        - Merging data from multiple sources into a single dataset

    Args:
        datafile (str):
            Path to the directory containing training CSV files.

    Returns:
        np.ndarray:
            Preprocessed training data of shape (num_samples, num_features),
            where each row represents a normalized posture sample suitable
            for SOM training.

    Notes:
        - Assumes posture data consists of 3D joint coordinates arranged as (x, y, z).
        - Supports multiple input sources (e.g., multi-camera setups).
        - Applies dataset-specific adjustments (e.g., axis flipping when required).
        - Ensures consistency between training and test preprocessing pipelines.
    """
    data_train = []
    eps = 1e-8

    data_paths = [read_data(datafile, "*c1-w.csv"), read_data(datafile, "*c2-w.csv")]

    for data_path in data_paths:
        num_records, num_features = data_path.shape
        print(f"There are {num_records} records with {num_features} features")

        data = data_path.rename(columns=lambda x: x.strip())
        print("Stripped column names")

        if data.shape[1] == 138:
            data = data.drop(
                columns=[
                    "Frame number",
                    "person count",
                    "x8",
                    "y8",
                    "z8",
                    "x9",
                    "y9",
                    "z9",
                    "x10",
                    "y10",
                    "z10",
                    "x15",
                    "y15",
                    "z15",
                    "x16",
                    "y16",
                    "z16",
                    "x17",
                    "y17",
                    "z17",
                    "x21",
                    "y21",
                    "z21",
                    "x25",
                    "y25",
                    "z25",
                    "x26",
                    "y26",
                    "z26",
                    "x28",
                    "y28",
                    "z28",
                    "x29",
                    "y29",
                    "z29",
                    "x30",
                    "y30",
                    "z30",
                    "x31",
                    "y31",
                    "z31",
                    "x32",
                    "y32",
                    "z32",
                    "x33",
                    "y33",
                    "z33",
                ],
                errors="ignore",
            )

            data = data.drop(columns=[f"p{i}" for i in range(34)], errors="ignore")

        elif data.shape[1] == 128:
            data = data.drop(
                columns=[
                    "x8",
                    "y8",
                    "z8",
                    "x9",
                    "y9",
                    "z9",
                    "x10",
                    "y10",
                    "z10",
                    "x15",
                    "y15",
                    "z15",
                    "x16",
                    "y16",
                    "z16",
                    "x17",
                    "y17",
                    "z17",
                    "x21",
                    "y21",
                    "z21",
                    "x25",
                    "y25",
                    "z25",
                    "x26",
                    "y26",
                    "z26",
                    "x28",
                    "y28",
                    "z28",
                    "x29",
                    "y29",
                    "z29",
                    "x30",
                    "y30",
                    "z30",
                    "x31",
                    "y31",
                    "z31",
                ],
                errors="ignore",
            )

            data = data.drop(columns=[f"p{i}" for i in range(32)], errors="ignore")

            # Flip across y-axis for Waseda-format data
            data.iloc[:, 1::3] = -data.iloc[:, 1::3]

        else:
            raise ValueError(f"Unexpected number of columns: {data.shape[1]}")

        data = data.apply(pd.to_numeric, errors="coerce")
        data = data.dropna()

        print("Dropped bad columns")
        print(f"Cleaned shape before transform: {data.shape}")

        data = transform_data(data)

        if data.isnull().values.any():
            data.fillna(data.mean(), inplace=True)
            print("Filled NaNs with column mean")

        if data.isnull().values.any():
            data.ffill(inplace=True)
            print("Forward-filled remaining NaNs")

        if data.isnull().values.any():
            data.bfill(inplace=True)
            print("Backward-filled remaining NaNs")

        data = data.astype(np.float32)

        # Replace missing marker -1 with 0
        data = data.replace(-1, 0)

        mean_values = data.mean()
        min_values = data.min()
        max_values = data.max()

        value_range = (max_values - min_values) + eps

        data = (data - mean_values) / value_range

        data = data.astype(np.float32)

        print("Final processed shape:", data.shape)

        data_train.extend(data.values.tolist())

    return np.asarray(data_train, dtype=np.float32)


def load_test_data(datafile):
    """
    Load and preprocess test posture data from CSV files.

    This function reads raw posture data from a specified directory,
    cleans and filters irrelevant columns, converts all values to numeric,
    handles missing values, applies coordinate transformations, and
    normalizes the data for use in SOM-based clustering and classification.

    Processing steps include:
        - Reading and concatenating CSV files
        - Stripping whitespace from column names
        - Dropping irrelevant joint and metadata columns
        - Converting all values to numeric format
        - Removing or imputing missing values (NaNs)
        - Transforming coordinates to relative joint positions
        - Replacing missing feature markers (-1) with 0
        - Normalizing features using zero-centered scaling

    Args:
        datafile (str):
            Path to the directory containing test CSV files.

    Returns:
        np.ndarray:
            Preprocessed test data of shape (num_samples, num_features),
            where each row represents a posture sample in normalized feature space.

    Notes:
        - Assumes input data consists of 3D joint coordinates arranged as (x, y, z).
        - Applies dataset-specific adjustments (e.g., axis flipping for Waseda data).
        - Output is suitable for SOM classification and distance computations.
    """
    data_paths = read_data(datafile, "*.csv")

    num_records, num_features = data_paths.shape
    print(f"{num_records} records, {num_features} features")

    data = data_paths.rename(columns=lambda x: x.strip())

    # -----------------------------
    # Column filtering
    # -----------------------------
    if data.shape[1] == 138:
        data = data.drop(
            columns=[
                "x8",
                "y8",
                "z8",
                "x9",
                "y9",
                "z9",
                "x10",
                "y10",
                "z10",
                "x15",
                "y15",
                "z15",
                "x16",
                "y16",
                "z16",
                "x17",
                "y17",
                "z17",
                "x21",
                "y21",
                "z21",
                "x25",
                "y25",
                "z25",
                "x26",
                "y26",
                "z26",
                "x28",
                "y28",
                "z28",
                "x29",
                "y29",
                "z29",
                "x30",
                "y30",
                "z30",
                "x31",
                "y31",
                "z31",
                "x32",
                "y32",
                "z32",
                "x33",
                "y33",
                "z33",
            ],
            errors="ignore",
        )

        data = data.drop(
            columns=[
                "Frame number",
                "person count",
                "p0",
                "p1",
                "p2",
                "p3",
                "p4",
                "p5",
                "p6",
                "p7",
                "p8",
                "p9",
                "p10",
                "p11",
                "p12",
                "p13",
                "p14",
                "p15",
                "p16",
                "p17",
                "p18",
                "p19",
                "p20",
                "p21",
                "p22",
                "p23",
                "p24",
                "p25",
                "p26",
                "p27",
                "p28",
                "p29",
                "p30",
                "p31",
                "p32",
                "p33",
            ],
            errors="ignore",
        )

    elif data.shape[1] == 128:
        data = data.drop(
            columns=[
                "x8",
                "y8",
                "z8",
                "x9",
                "y9",
                "z9",
                "x10",
                "y10",
                "z10",
                "x15",
                "y15",
                "z15",
                "x16",
                "y16",
                "z16",
                "x17",
                "y17",
                "z17",
                "x21",
                "y21",
                "z21",
                "x25",
                "y25",
                "z25",
                "x26",
                "y26",
                "z26",
                "x28",
                "y28",
                "z28",
                "x29",
                "y29",
                "z29",
                "x30",
                "y30",
                "z30",
                "x31",
                "y31",
                "z31",
                "x32",
                "y32",
                "z32",
                "x33",
                "y33",
                "z33",
            ],
            errors="ignore",
        )

        data = data.drop(
            columns=[
                "Frame number",
                "person count",
                "p0",
                "p1",
                "p2",
                "p3",
                "p4",
                "p5",
                "p6",
                "p7",
                "p8",
                "p9",
                "p10",
                "p11",
                "p12",
                "p13",
                "p14",
                "p15",
                "p16",
                "p17",
                "p18",
                "p19",
                "p20",
                "p21",
                "p22",
                "p23",
                "p24",
                "p25",
                "p26",
                "p27",
                "p28",
                "p29",
                "p30",
                "p31",
                "p32",
                "p33",
            ],
            errors="ignore",
        )
        data.iloc[:, 1::3] = -data.iloc[:, 1::3]

    # -----------------------------
    # Convert to numeric
    # -----------------------------
    data = data.apply(pd.to_numeric, errors="coerce")

    # -----------------------------
    # Handle NaNs
    # -----------------------------
    if data.isnull().values.any():
        data.fillna(data.mean(), inplace=True)

    if data.isnull().values.any():
        data.fillna(method="ffill", inplace=True)

    if data.isnull().values.any():
        data.fillna(method="bfill", inplace=True)

    # -----------------------------
    # Transform
    # -----------------------------
    data = transform_data(data)

    # -----------------------------
    # Replace missing (-1)
    # -----------------------------
    data = data.replace(-1, 0)

    # -----------------------------
    # Normalize
    # -----------------------------
    eps = 1e-8

    mean_i = data.mean()
    min_i = data.min()
    max_i = data.max()

    r = (max_i - min_i) + eps
    data = (data - mean_i) / r

    # -----------------------------
    # Final conversion
    # -----------------------------
    data = data.values.astype(np.float32)

    print("Final shape:", data.shape)

    return data
