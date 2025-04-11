import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the local dataset
file_paths = [
    "C:/VS code projects/data_files/Monday-WorkingHours.pcap_ISCX.csv",
    "C:/VS code projects/data_files/Tuesday-WorkingHours.pcap_ISCX.csv",
    "C:/VS code projects/data_files/Wednesday-workingHours.pcap_ISCX.csv",
    "C:/VS code projects/data_files/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "C:/VS code projects/data_files/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "C:/VS code projects/data_files/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "C:/VS code projects/data_files/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "C:/VS code projects/data_files/Friday-WorkingHours-Morning.pcap_ISCX.csv"
]



# Read and clean datasets
dataframes = []
for file_path in file_paths:
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Remove whitespace from column names
    dataframes.append(df)

# Combine all datasets into a single DataFrame
df = pd.concat(dataframes, ignore_index=True)df['Label'] = df['Label'].apply(lambda x: 1 if x == 'BENIGN' else 0)

# Find the minimum class count
min_count = df['Label'].value_counts().min()


# Perform undersampling to balance the dataset
df = df.groupby('Label', group_keys=False).apply(lambda x: x.sample(n=min_count, random_state=42, ignore_index=True))

# Preprocessing dataset
print("Preprocessing dataset...")
df.dropna(inplace=True)  # Remove missing values
df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinite values with NaN
df.columns = df.columns.str.strip()  # Strip whitespaces from column names

# Separate features and labels
X = df.drop('Label', axis=1)  # Features
Y = df['Label']              # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# Define the objective function for Optuna
def objective(trial):
    # Define the hyperparameter search space
    param_grid = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
    }

    # Train LightGBM with the current set of hyperparameters
    model = lgb.LGBMClassifier(**param_grid, random_state=42)  # Unpack param_grid using **
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_test, y_test)], 
        eval_metric='logloss', 
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]  # Use callbacks for verbosity and early stopping
    )

    # Predict on the validation set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Return the accuracy as the objective to maximize
    return accuracy

# Create a study and optimize the objective function
study = optuna.create_study(direction='maximize')  # Maximize accuracy
study.optimize(objective, n_trials=50, timeout=3600)  # Run 50 trials or stop after 1 hour

# Print the best hyperparameters
print("Best Hyperparameters:")
print(study.best_params)