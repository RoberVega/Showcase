This is how the workflow went:

1. Created a virtual enviroment with pipenv in order to generate a Pipfile and Pipfile.lock that I could then use to generate a virtual environment within docker. I installed pandas, scikit-learn, notebooks and mlflow for general usage and pytest and pylint as development dependencies.

2. Created the folder structure: notebooks for EDA, data for the data, integration_tests and tests for testing.

3. Artificially divide the dataset into two in order to represent different months.

4. Analyse the data by creating reusable functions that we will be able to apply to newer datasets.

5. Create a ColumnTransformer to apply to the functions.

Directory structure:
- data: contains data (a folder where the processed data is stored) and raw_data (folder for original data)
- integration_tests: self-explannatory
- models: to store the models in pickle
- notebooks: original EDA
- scripts: contains the preprocess_utils folder (a folder made a package to store the preprocessing funtions); preprocess_data.py imports the raw data from a source (in this case the raw_data folder) and saves the prepocess it saving it into a folder (in this case the data/data folder); hpo.py runs a hyperparameter optimization using optuna for the dataset and saves all tries on a database without saving the models themselves to save space (only saves the parameters and metrics); train.py selects the bests models in the MLflow database, trains them and saves them for future usage; register.py is a simple script to register the model; predict.py uses flask to deploy the model for requests.
- tests: script tests 

- We created a Dockerfile in which we only needed to copy the folder, activate the pipenv environment and then open a couple of ports for jupyter notebook and MLflow

Docker bash:

docker build -t showcase_image .

MLFlow bash (the default-artifact-root has to be added since we do not log artifacts in our hpo.py script):

mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

Tip: In order to activate python extension with the selected pipenv environment, press in VSCode ctrl + shift + p and select the environment. This will allow VSCode to perform autocomplete and other cool things.

Whole model-building from the terminal:

python scripts/preprocess_data.py --raw_data_path=data/raw_data --dest_path=data/data

python scripts/hpo.py

python scripts/train.py

python scripts/register_model.py
