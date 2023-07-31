import pandas as pd

def prepare_features(data: pd.DataFrame):
    df = data.copy()
           
    df = transform_binaries(df)

    df = transform_diabetes(df) if "Diabetes" in df.columns else df
    
    return df

def transform_binaries(data: pd.DataFrame, 
                       columns: list[str] = ["Exercise", "Skin_Cancer", "Other_Cancer",
                                             "Depression", "Arthritis", "Smoking_History"])-> pd.DataFrame:
    
    df = data.copy()
    for column in columns:
        df[column] = df[column].str.lower().map({'yes': 1, 'no':0})
    
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"Female": 1, "Male": 0})
    
    if "Heart_Disease" in df.columns:
        df["Heart_Disease"] = df["Heart_Disease"].str.lower().map({'yes': 1, 'no':0})
        
    return df

def transform_diabetes(data: pd.DataFrame):
    df = data.copy()
    df['Diabetes'] = df['Diabetes'].str.lower().map({"yes": 1,
                                         "yes, but female told only during pregnancy": 0, 
                                         "no": 0, 
                                         "no, pre-diabetes or borderline diabetes": 0.5})
    return df


