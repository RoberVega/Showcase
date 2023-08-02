import os
import click
import joblib
import pickle
import numpy as np
import pandas as pd
from preprocess_utils.preprocess import prepare_features
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from prefect import task,flow

def dump_pickle(obj, filename: str)-> None:
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


@task(name='Read the dataframe and prepare features',retries=3, retry_delay_seconds=2)
def read_dataframe(filename: str):
    df = pd.read_csv(filename)

    df = prepare_features(df)
    
    return df

@task(name='Prepocess the data')
def preprocess(X: pd.DataFrame, 
               ct: ColumnTransformer, 
               fit_ct: bool = False):

    transformer_columns = ["General_Health","Checkup","Age_Category","Height_(cm)",
                        "Weight_(kg)","BMI","Alcohol_Consumption","Fruit_Consumption",
                        "Green_Vegetables_Consumption","FriedPotato_Consumption"]
    remainder_columns = []

    for col in X.columns:
        if col not in transformer_columns:
            remainder_columns.append(col)

    column_transformer_order = np.concatenate((transformer_columns,remainder_columns))

    if fit_ct:
        X = pd.DataFrame(data = ct.fit_transform(X), index = X.index, columns = column_transformer_order)
    else:
        X = pd.DataFrame(data = ct.transform(X), index = X.index, columns = column_transformer_order)

    return X, ct


@click.command()
@click.option(
    "--raw_data_path",
    help="Location where the raw NYC taxi trip data was saved"
)
@click.option(
    "--dest_path",
    help="Location where the resulting files will be saved"
)
@flow(name='Run data preparation')
def run_data_prep(raw_data_path: str, dest_path: str):
    # Load parquet files
    df_train = read_dataframe(
        os.path.join(raw_data_path, f"data_jan.csv")
    )
    df_val = read_dataframe(
        os.path.join(raw_data_path, f"data_feb.csv")
    )
    df_test = read_dataframe(
        os.path.join(raw_data_path, f"data_mar.csv")
    )

    # Extract the target
    target = 'Heart_Disease'
    y_train = df_train[target]
    y_val = df_val[target]
    y_test = df_test[target]
    X_train = df_train.drop(columns=[target])
    X_val = df_val.drop(columns=[target])
    X_test = df_test.drop(columns=[target])


    # Fit the ColumnTransformer and preprocess data
    ct = ColumnTransformer(
        [
            ("gen_health_preprocess", 
            OrdinalEncoder(categories=[["Poor","Fair","Good","Very Good","Excellent"]]),
            ["General_Health"]),
            
            
            ("checkup_preprocess",
            OrdinalEncoder(categories=[['Never','5 or more years ago','Within the past 5 years',
                                        'Within the past 2 years', 'Within the past year']]),
            ["Checkup"]),
            
            
            ("age_preprocess",
            OrdinalEncoder(categories=[['18-24','25-29','30-34','35-39','40-44','45-49','50-54',
                                        '55-59','60-64','65-69','70-74','75-79','80+']],
                        handle_unknown='use_encoded_value', unknown_value=-1),
            ["Age_Category"]),
            
            
            ("num_preprocess", 
            StandardScaler(),
            ["Height_(cm)","Weight_(kg)","BMI","Alcohol_Consumption","Fruit_Consumption",
            "Green_Vegetables_Consumption","FriedPotato_Consumption"])
        ],
        remainder = 'passthrough'
    )
    X_train, ct = preprocess(X_train, ct, fit_ct=True)
    X_val, _ = preprocess(X_val, ct, fit_ct=False)
    X_test, _ = preprocess(X_test, ct, fit_ct=False)

    rus = RandomUnderSampler(sampling_strategy = 0.5)
    X_train, y_train = rus.fit_resample(X_train, y_train)
    
    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # Save DictVectorizer and datasets
    joblib.dump(ct, os.path.join(dest_path, "ct.pkl"))
    joblib.dump(rus, os.path.join(dest_path, "rus.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))


if __name__ == '__main__':
    run_data_prep()