import requests


def test_flask():
    patient = {'General_Health': 'Poor',
               'Checkup': 'Within the past 2 years',
               'Exercise': 'No',
               'Skin_Cancer': 'No',
               'Other_Cancer': 'No',
               'Depression': 'No',
               'Diabetes': 'No',
               'Arthritis': 'Yes',
               'Sex': 'Female',
               'Age_Category': '70-74',
               'Height_(cm)': 150.0,
               'Weight_(kg)': 32.66,
               'BMI': 14.54,
               'Smoking_History': 'Yes',
               'Alcohol_Consumption': 0.0,
               'Fruit_Consumption': 30,
               'Green_Vegetables_Consumption': 16.0,
               'FriedPotato_Consumption': 12.0
               }
    url = 'http://localhost:9696/predict'
    response = requests.post(url, json=patient)

    print(response.json())  # Print the JSON content of the response

    prediction = float(response.json()['Probability of diabetes'])  # Access the prediction value

    assert prediction > 0