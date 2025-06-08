# Hybrid Botnet Detection in IOT using GA and LightGBM

This repository implements a comprehensive pipeline for detecting botnet attacks in network traffic by combining Genetic Algorithms (GA) for feature selection with LightGBM for classification. The pipeline includes robust data cleaning, pre-processing, evolutionary feature selection, and model training/evaluation.

## Project Overview

The project consists of the following main components:

1. **Data Import & Cleaning:**  
   - Loads multiple CSV files containing network traffic data.  
   - Concatenates the data into a single DataFrame, removes duplicates, cleans column names, replaces infinity values, and imputes missing data (using mean for numeric features and mode for categorical ones).  
   - Drops constant columns that do not contribute useful information.

2. **Data Splitting & Preprocessing:**  
   - Splits the cleaned data into stratified training and test sets.
   - Encodes categorical variables using techniques such as Label Encoding and One-Hot Encoding.
   - Applies SMOTE to mitigate class imbalance and scales features with MinMaxScaler.

3. **Feature Selection via Genetic Algorithm (GA):**  
   - Uses the DEAP framework to represent feature subsets as binary vectors.
   - Evaluates each individual (feature subset) by training a LightGBM model on a random subsample of the training data.
   - Applies GA operators (crossover, mutation, selection) over multiple generations.
   - Records fitness (measured by model accuracy) and identifies the best subset of features.

4. **Final Model Training & Evaluation:**  
   - Trains a final LightGBM model with early stopping using only the GA-selected features.
   - Uses 5-fold cross-validation to compute metrics such as Accuracy, Precision, Recall, F1 Score, AUC, Log Loss, and Matthews Correlation Coefficient (MCC).
   - Evaluates the final model on an unseen test set and generates visualizations (ROC curve, precision-recall curve, confusion matrix heatmap, and bar charts of metrics).

## Repository Structure

```
Hybrid-Botnet-Detection-Using-GA-and-LightGBM/
├── .gitignore
├── README.md               # This file
├── requirements.txt        # List of required Python packages
├── data_files/
│   ├── bot_iot/            # CSV files for Bot-IoT dataset
│   └── ton_iot/            # CSV files for ToN_IoT dataset
├── 2.IDS_GA_LightGBM_SingleModal copy.ipynb   # Primary notebook with pipeline implementation
├── 2.IDS_GA_LightGBM_SingleModal.ipynb         # Alternative version or additional experiments
└── model_export_cell7/
    ├── ids_lightgbm_final_model.pkl
    ├── scaler_final_model.pkl
    ├── selected_feature_names_final.json
    └── selected_features_indices_final.npy
```

## Prerequisites

- Python 3.7 or later
- The following Python packages (install via `pip install -r requirements.txt`):
  - numpy
  - pandas
  - scikit-learn
  - lightgbm
  - imbalanced-learn
  - deap
  - matplotlib
  - seaborn
  - joblib

## How to Run

### 1. Clone the Repository

Clone the repository to your local machine:

```sh
git clone https://github.com/yourusername/Hybrid-Botnet-Detection-Using-GA-and-LightGBM.git
cd Hybrid-Botnet-Detection-Using-GA-and-LightGBM
```

### 2. Install Dependencies

Install the required packages using pip:

```sh
pip install -r requirements.txt
```

### 3. Prepare the Data

Place the CSV data files in the proper subdirectories:
- Bot-IoT dataset CSV files go in `data_files/bot_iot/`
- ToN_IoT dataset CSV files go in `data_files/ton_iot/`

### 4. Run the Notebooks

You can open the provided Jupyter Notebooks in VS Code or Jupyter Lab:
- Open `2.IDS_GA_LightGBM_SingleModal copy.ipynb` to run the main pipeline.

The notebook is organized into cells that:
- **Import and clean data:** Load and preprocess the Bot-IoT and/or ToN_IoT datasets.
- **Split and scale data:** Perform stratified splitting, apply SMOTE, and scale features.
- **Perform GA-based Feature Selection:** Evaluate feature subsets using LightGBM, and track fitness evolution.
- **Train and Evaluate Final Model:** Use GA-selected features for final training, cross-validation, and test set evaluation with visualizations.

Run cells sequentially (or use “Run All”) to execute the full pipeline.

### 5. Model Export

After training, the trained model, scaler, and GA-selected feature indices/names are saved in the `model_export_cell7` folder. You can load these artifacts later for inference.

## Additional Information

- **ToN_IoT Processing:**  
  The notebook also contains cells dedicated to processing the ToN_IoT dataset. These cells follow the same pipeline structure as the Bot-IoT processing steps, including data loading, cleaning, feature engineering, and cross-validation.

- **Customization:**  
  Adjust parameters (e.g., GA hyperparameters, LightGBM parameters) within the notebook cells as needed to experiment with different configurations.

## License

This project is provided "as is", without warranty of any kind. See [LICENSE](./LICENSE) for details.

## Acknowledgments

- [LightGBM](https://github.com/Microsoft/LightGBM) for the gradient boosting framework.
- [DEAP](https://github.com/DEAP/deap) for the genetic algorithm library.
- Contributions from various research papers and the open-source community related to botnet detection.

---

Feel free to explore the notebooks, experiment with the parameters, and modify the code to suit your needs.
