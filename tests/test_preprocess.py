import scripts.preprocess_utils.preprocess as preprocess
import pandas as pd
from pandas.testing import assert_frame_equal

def test_transform_binaries():
    data = pd.DataFrame({"Exercise": ["Yes", "No", "Yes", "No", "Yes"],
        "Heart_Disease": ["No", "No", "Yes", "No", "Yes"],
        "Skin_Cancer": ["No", "Yes", "No", "No", "Yes"],
        "Other_Cancer": ["Yes", "No", "Yes", "Yes", "No"],
        "Depression": ["Yes", "No", "Yes", "No", "No"],
        "Arthritis": ["No", "Yes", "No", "No", "Yes"],
        "Smoking_History": ["Yes", "No", "No", "No", "Yes"],
        "Sex": ["Female", "Female", "Male", "Male", "Female"],
        "Extra_feature": ["A", "B", "C", "D", "E"]
    })

    function_features = preprocess.transform_binaries(data)

    expected_features = pd.DataFrame({"Exercise": [1, 0, 1, 0, 1],
        "Heart_Disease": [0, 0, 1, 0, 1],
        "Skin_Cancer": [0, 1, 0, 0, 1],
        "Other_Cancer": [1, 0, 1, 1, 0],
        "Depression": [1, 0, 1, 0, 0],
        "Arthritis": [0, 1, 0, 0, 1],
        "Smoking_History": [1, 0, 0, 0, 1],
        "Sex": [1, 1, 0, 0, 1],
        "Extra_feature": ["A", "B", "C", "D", "E"]
    })
        
    assert_frame_equal(function_features, expected_features) 


def test_transform_diabetes():
    
    data = {
        "Diabetes": [
            "Yes",
            "Yes, but female told only during pregnancy",
            "No",
            "No, pre-diabetes or borderline diabetes",
            "Yes",
            "No",
            "Yes, but female told only during pregnancy",
            "No, pre-diabetes or borderline diabetes",
            "No",
            "Yes",
            "Yes, but female told only during pregnancy",
            "Yes",
            "No",
            "No, pre-diabetes or borderline diabetes",
        ]
    }
    data = pd.DataFrame(data)
    function_features = preprocess.transform_diabetes(data)
    expected_features = pd.DataFrame({        
        "Diabetes": [
            1,
            0,
            0,
            0.5,
            1,
            0,
            0,
            0.5,
            0,
            1,
            0,
            1,
            0,
            0.5,
        ]
    })
    assert_frame_equal(function_features, expected_features) 
