# Stanford University Students Success Prediction

This machine learning project uses Random Forest Classifier to predict whether a student will successfully graduate or drop out.

## Project Structure

The project consists of the following folders:

1. **data**: This folder contains the following splitted training/validation and prediction datasets.

2. **models**: This folder is used to store the trained Random Forest Classifier model.

3. **processors**: This folder contains the imputers and scalers that were trained during feature engineering and will be used for prediction dataset processing.

3. **src**: This folder contains the source code files:
   - `feature_engineering.py`: renames columns to snake_case, imputes missing values, encodes categorical columns, replaces outliers, scales numerical features and saves processed dataset. All the processors (imputers and scaler) are saved to `processors` folder.
   - `model_report.py`: a class representing model training report. Contains the model name and its main metrics.
   - `predicting.py`: loads a trained model and predicts the target value.
   - `splitter.py`: splits the given dataset into  2 parts: the first for model training and validation and the second for prediction.
   - `training.py`: splits the given dataset into training and validation sets, oversamples the training set using SMOTE to equalize the target class distribution, trains the Random Forest Classifier model, evaluates it and returns a corresponding model report. Trained model is saved to `models` folder.
   - `constants.py`: contains constants related to data folder paths and dataset columns.
   - `main.py`: contains endpoints for splitting dataset, traing the model and predicting the target.

## Requirements

The project requires the following dependencies specified in the `requirements.txt` file:

```
fastapi
uvicorn
pandas
scikit-learn
imbalanced-learn
joblib
```

Please install the dependencies using the following command:

```shell
pip install -r requirements.txt
```

## Usage

To use this project, follow these steps:

1. Ensure that the required dependencies are installed by running the above command.

2. Run the API:
   ```shell
   uvicorn src.main:app --reload
   ```

3. Go to http://127.0.0.1:8000/docs#

4. Split the dataset training/validation and prediction sets using `/split` endpoint.

5. Train the model using `/train` endpoint.

6. Predict the results using the `/predict` endpoint.

The prediction results will be saved in the `files/data/prediction_results.csv` file.

## Conclusion

This project provides a comprehensive approach to predict students success using Random Fores Classifiers. Feel free to explore the code and customize it to fit your specific needs.