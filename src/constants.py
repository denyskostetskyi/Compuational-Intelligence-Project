import os

RANDOM_STATE = 42

TEST_DATASET_SIZE = 0.2

PATH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATH_FILES = os.path.join(PATH_ROOT, "files")

PATH_DATA = os.path.join(PATH_FILES, "data")

#  models paths constants
PATH_MODELS = os.path.join(PATH_FILES, "models")
PATH_MODEL_RANDOM_FOREST_CLASSIFIER = os.path.join(PATH_MODELS, "random_forest_classifier.pkl")

# processors paths constants
PATH_PROCESSORS = os.path.join(PATH_FILES, "processors")
PATH_SCALER_NUMERICAL = os.path.join(PATH_PROCESSORS, "numerical_values_standard_scaler.pkl")
PATH_IMPUTER_NUMERICAL = os.path.join(PATH_PROCESSORS, "numerical_missing_values_imputer.pkl")
PATH_IMPUTER_CATEGORICAL = os.path.join(PATH_PROCESSORS, "categorical_missing_values_imputer.pkl")

# dataset paths constants
PATH_PROCESSED_DATASET = os.path.join(PATH_DATA, "processed_dataset.csv")
PATH_SPLITTED_DATA = os.path.join(PATH_DATA, "splitted")
PATH_TRAINING_DATASET = os.path.join(PATH_DATA, "training_dataset.csv")
PATH_PREDICTION_DATASET = os.path.join(PATH_DATA, "prediction_dataset.csv")
PATH_X_TRAIN = os.path.join(PATH_SPLITTED_DATA, "X_train.csv")
PATH_Y_TRAIN = os.path.join(PATH_SPLITTED_DATA, "y_train.csv")
PATH_X_TEST = os.path.join(PATH_SPLITTED_DATA, "X_test.csv")
PATH_Y_TEST = os.path.join(PATH_SPLITTED_DATA, "y_test.csv")
PATH_PREDICTION_RESULTS = os.path.join(PATH_DATA, "prediction_results.csv")

# all columns from the raw dataset
COLUMNS_RAW = [
    "Unnamed: 0", "Marital status", "Application mode", "Application order",
    "Course", "Daytime/evening attendance\t", "Previous qualification",
    "Previous qualification (grade)", "Nacionality", "Mother's qualification",
    "Father's qualification", "Mother's occupation", "Father's occupation",
    "Admission grade", "Displaced", "Educational special needs", "Debtor",
    "Tuition fees up to date", "Gender", "Scholarship holder",
    "Age at enrollment", "International", "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)", "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)", "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)", "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)", "Unemployment rate",
    "Inflation rate", "GDP", "Target", "Citizenship", "Family Position",
    "Attendance", "Field of Study", "Special Needs"
]

# columns selected for model training and prediction
COLUMNS_SELECTED = [
    'marital_status',
    'application_mode',
    'application_order',
    'course',
    'daytime_attendance',
    'previous_qualification',
    'previous_qualification_grade',
    'mothers_qualification',
    'fathers_qualification',
    'mothers_occupation',
    'admission_grade',
    'displaced',
    'educational_special_needs',
    'debtor',
    'tuition_fees_up_to_date',
    'male',
    'scholarship_holder',
    'age',
    'curricular_units_1st_sem_enrolled',
    'curricular_units_1st_sem_evaluations',
    'curricular_units_1st_sem_approved',
    'curricular_units_1st_sem_grade',
    'curricular_units_1st_sem_without_evaluations',
    'curricular_units_2nd_sem_credited',
    'curricular_units_2nd_sem_enrolled',
    'curricular_units_2nd_sem_evaluations',
    'curricular_units_2nd_sem_approved',
    'curricular_units_2nd_sem_grade',
    'curricular_units_2nd_sem_without_evaluations',
    'inflation_rate',
    'target'
]

# selected columns with numerical values
COLUMNS_NUMERICAL = [
    "application_order",
    "previous_qualification_grade",
    "admission_grade",
    "age",
    "curricular_units_1st_sem_enrolled",
    "curricular_units_1st_sem_evaluations",
    "curricular_units_1st_sem_approved",
    "curricular_units_1st_sem_grade",
    "curricular_units_1st_sem_without_evaluations",
    "curricular_units_2nd_sem_credited",
    "curricular_units_2nd_sem_enrolled",
    "curricular_units_2nd_sem_evaluations",
    "curricular_units_2nd_sem_approved",
    "curricular_units_2nd_sem_grade",
    "curricular_units_2nd_sem_without_evaluations",
    "inflation_rate"
]

# contains selected numerical columns with their possible value ranges
COLUMNS_NUMERICAL_WITH_VALUES = [
    ("previous_qualification_grade", 0, 200),
    ("admission_grade", 0, 200),
    ("age", 0, 100),
    ("curricular_units_1st_sem_enrolled", 0, 23),
    # ("curricular_units_1st_sem_evaluations", 0, 46),
    ("curricular_units_1st_sem_approved", 0, 23),
    ("curricular_units_1st_sem_grade", 0, 20),
    ("curricular_units_1st_sem_without_evaluations", 0, 23),
    ("curricular_units_2nd_sem_credited", 0, 23),
    ("curricular_units_2nd_sem_enrolled", 0, 23),
    # ("curricular_units_2nd_sem_evaluations", 0, 46),
    ("curricular_units_2nd_sem_approved", 0, 23),
    ("curricular_units_2nd_sem_grade", 0, 20),
    ("curricular_units_2nd_sem_without_evaluations", 0, 23)
    # 'inflation_rate' can be less than 0
]

# selected columns with categorical values
COLUMNS_CATEGORICAL = [
    "marital_status",
    "application_mode",
    "course",
    "daytime_attendance",
    "previous_qualification",
    "mothers_qualification",
    "fathers_qualification",
    "mothers_occupation",
    "displaced",
    "educational_special_needs",
    "debtor",
    "tuition_fees_up_to_date",
    "male",
    "scholarship_holder",
    "target"
]

# contains selected categorical columns with their possible values
COLUMNS_CATEGORICAL_WITH_VALUES = [
    ("marital_status", [1, 2, 3, 4, 5, 6]),
    ("application_mode", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
    ("application_order", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ("course", [33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238, 9254, 9500, 9556, 9670, 9773, 9853, 9991]),
    ("previous_qualification", [1, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 19, 38, 39, 40, 42, 43]),
    ("mothers_qualification", [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 18, 19, 22, 26, 27, 29, 30, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]),
    ("fathers_qualification", [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 18, 19, 20, 22, 25, 26, 27, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]),
    ("mothers_occupation", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90, 99, 122, 123, 125, 131, 132, 134, 141, 143, 144, 151, 152, 153, 171, 173, 175, 191, 192, 193, 194])
]

# selected columns that should have integer data type
# contains both numerical and encoded categorical columns
COLUMNS_INTEGER = [
    'daytime_attendance',
    'displaced',
    'educational_special_needs',
    'debtor',
    'tuition_fees_up_to_date',
    'male',
    'scholarship_holder',
    'marital_status',
    'application_mode',
    'application_order',
    'course',
    'previous_qualification',
    'mothers_qualification',
    'fathers_qualification',
    'mothers_occupation',
    'age',
    'curricular_units_1st_sem_enrolled',
    'curricular_units_1st_sem_evaluations',
    'curricular_units_1st_sem_approved',
    'curricular_units_1st_sem_without_evaluations',
    'curricular_units_2nd_sem_credited',
    'curricular_units_2nd_sem_enrolled',
    'curricular_units_2nd_sem_evaluations',
    'curricular_units_2nd_sem_approved',
    'curricular_units_2nd_sem_without_evaluations',
    'target'
]

COLUMN_TARGET = "target"

TARGET_MAPPING = { "Dropout": 0, "Enrolled": 1, "Graduate": 2}

# mapping for prediction results
TARGET_MAPPING_INV = {v: k for k, v in TARGET_MAPPING.items()}