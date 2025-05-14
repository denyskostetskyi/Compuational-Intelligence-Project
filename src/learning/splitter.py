import pandas as pd
from sklearn.model_selection import train_test_split

import constants


def split_dataset(df: pd.DataFrame, predict_size: float):
    df_train, df_predict = train_test_split(df, test_size=predict_size, random_state=constants.RANDOM_STATE)
    # df_predict.drop(labels=[constants.COLUMN_TARGET], inplace=True)
    df_train.to_csv(constants.PATH_TRAINING_DATASET, index=False)
    df_predict.to_csv(constants.PATH_PREDICTION_DATASET, index=False)

    return len(df_train), len(df_predict)
