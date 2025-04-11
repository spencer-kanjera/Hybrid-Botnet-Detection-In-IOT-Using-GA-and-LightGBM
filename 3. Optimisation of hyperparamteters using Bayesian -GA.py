import optuna
import pandas as pd
from deap import base, creator, tools, algorithms
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
import numpy as np

# Load the dataset
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
df = pd.concat(dataframes, ignore_index=True)

print("Preprocessing dataset...")
df.dropna(inplace=True)  # Remove missing values
df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinite values with NaN
df.columns = df.columns.str.strip()  # Strip whitespaces from column names
X = df.drop(columns=['Label'])  # Replace 'Label' with the actual target column name if different
y = df['Label']

# Define the DEAP toolbox
num_features = X.shape[1]  # Number of features in the dataset

# Create the fitness function (maximize accuracy)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.randint, 0, 2)  # Binary representation (0 or 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register genetic operators
toolbox.register("mate", tools.cxTwoPoint)  # Two-point crossover
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  # Bit-flip mutation
toolbox.register("select", tools.selTournament, tournsize=3)  # Tournament selection

# Define the evaluation function
def evaluate_model(selected_features):
    # Use only the selected features
    X_selected = X.iloc[:, selected_features]

    # Define a LightGBM classifier
    clf = lgb.LGBMClassifier(random_state=42)

    # Perform cross-validation and return the mean score
    scores = cross_val_score(clf, X_selected, y, cv=5, scoring='accuracy')
    return scores.mean()

# Register the evaluation function to the toolbox
def evaluate_individual(individual):
    # Convert the individual (binary list) into selected feature indices
    selected_features = [i for i, bit in enumerate(individual) if bit == 1]
    if len(selected_features) == 0:  # Handle cases where no features are selected
        return 0.0,
    score = evaluate_model(selected_features)
    return score,

toolbox.register("evaluate", evaluate_individual)

# Define the objective function for Optuna
def objective(trial):
    # Suggest values for GA parameters
    population_size = trial.suggest_int("population_size", 20, 100, step=10)
    ngen = trial.suggest_int("ngen", 10, 50, step=10)
    cxpb = trial.suggest_float("cxpb", 0.5, 0.9, step=0.1)
    mutpb = trial.suggest_float("mutpb", 0.1, 0.3, step=0.05)

    # Set up the GA with suggested parameters
    population = toolbox.population(n=population_size)
    result_population = algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False)
    best_individual = tools.selBest(result_population[0], k=1)[0]

    # Evaluate the selected features
    selected_features = [i for i, bit in enumerate(best_individual) if bit == 1]
    if len(selected_features) == 0:  
        return 0.0
    score = evaluate_model(selected_features)

    return score  e

# Run the optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, timeout=3600)

# Print the best parameters
print("Best Parameters:", study.best_params)
print("Best Score:", study.best_value)