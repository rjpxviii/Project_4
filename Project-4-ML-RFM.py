import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import logging
import sys

# Set up logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """
    Loads the dataset from the given file path and drops the 'Unnamed' column if present.
    """
    try:
        logging.info(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)

        # Drop the unnamed index column if it exists
        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns=['Unnamed: 0'])
            logging.info("'Unnamed: 0' column dropped from the data")

        logging.info("Data successfully loaded")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)

def preprocess_data(data, target_column):
    """
    Prepares the dataset by removing 'Percent_Owner_Occ', separating features and target,
    imputing missing values, scaling features, and ensuring only numeric columns are used.
    The 'City' column is also retained for future use.

    Parameters:
        data (pd.DataFrame): The dataset.
        target_column (str): The name of the target column.

    Returns:
        tuple: Scaled training and testing sets, target variable sets, and the 'City' column for reporting.
    """
    try:
        logging.info(f"Separating features from target column '{target_column}' and removing 'Percent_Owner_Occ'")
        X = data.drop(columns=[target_column, 'Percent_Owner_Occ'])
        y = data[target_column]
        
        # Retain 'City' column for future reference
        city_column = X['City'] if 'City' in X.columns else None
        if 'City' in X.columns:
            X = X.drop(columns=['City'])

        # Ensure only numeric columns are processed for X
        X_numeric = X.select_dtypes(include=[np.number])

        # Handle missing or infinite values for y
        y = y.replace([np.inf, -np.inf], np.nan).dropna()

        # Log-transform the target variable to handle skewness
        y_log = np.log1p(y)

        # Ensure that the shapes of X and y match after removing invalid y values
        X_numeric = X_numeric.loc[y.index]

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_numeric, y_log, test_size=0.2, random_state=42)

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        logging.info("Data preprocessing complete")
        return X_train_scaled, X_test_scaled, y_train, y_test, city_column
    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        sys.exit(1)

def tune_random_forest(X_train, y_train):
    """
    Performs hyperparameter tuning on RandomForestRegressor using GridSearchCV.
    """
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],          # Number of trees
        'max_depth': [10, 20, 30, None],          # Maximum depth of trees
        'min_samples_split': [2, 5, 10],          # Minimum number of samples to split
        'min_samples_leaf': [1, 2, 4],            # Minimum number of samples at a leaf
        'max_features': ['sqrt', 'log2', None],   # Valid values for max_features
        'bootstrap': [True, False]                # Whether bootstrap samples are used
    }

    # Instantiate the RandomForestRegressor
    rf = RandomForestRegressor(random_state=42)

    # Instantiate GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2, scoring='r2')

    # Fit the model to find the best parameters
    grid_search.fit(X_train, y_train)

    logging.info(f"Best parameters found: {grid_search.best_params_}")
    return grid_search

def evaluate_model(model, X_test, y_test, city_column):
    """
    Evaluates the model using the test data and prints the results. 
    It also saves the predictions along with the city information to a CSV file.

    Parameters:
        model: The trained model to evaluate.
        X_test: Scaled test features.
        y_test: Actual test target values.
        city_column: Original 'City' column for test data.

    Returns:
        None
    """
    try:
        logging.info("Making predictions and evaluating the model")
        y_pred_log = model.predict(X_test)

        # Reverse the log transformation to get original scale
        y_pred = np.expm1(y_pred_log)
        y_test_original = np.expm1(y_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)

        logging.info(f"Mean Squared Error (MSE): {mse}")
        logging.info(f"R-squared (R^2): {r2}")

        print(f"Mean Squared Error: {mse}")
        print(f"R^2 Score: {r2}")

        # Combine predictions with the City column (if available)
        results_df = pd.DataFrame({
            'Actual': y_test_original,
            'Predicted': y_pred
        })
        
        if city_column is not None:
            results_df['City'] = city_column.loc[y_test.index]

        # Save the results to a CSV file
        results_file_path = '/Users/ryanpope/Documents/Project-4/model_results.csv'
        results_df.to_csv(results_file_path, index=False)
        logging.info(f"Results saved to '{results_file_path}'")
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        sys.exit(1)

def analyze_feature_importance(model, feature_names):
    """
    Analyzes and displays feature importance for the best estimator from GridSearchCV.

    Parameters:
        model (GridSearchCV): The GridSearchCV object containing the best estimator.
        feature_names (list): The names of the features in X_train after preprocessing.
        
    Returns:
        None
    """
    try:
        logging.info("Analyzing feature importances")
        
        # Access the best estimator from GridSearchCV
        best_estimator = model.best_estimator_

        # Check if the model has feature_importances_ attribute
        if hasattr(best_estimator, 'feature_importances_'):
            importances = best_estimator.feature_importances_

            # Ensure feature_names matches the length of importances
            if len(importances) == len(feature_names):
                feature_importance_df = pd.DataFrame({
                    'Feature': feature_names, 
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False)
                print("\nFeature Importances:\n", feature_importance_df)
            else:
                logging.error("The number of features and importances do not match.")
        else:
            logging.error("The model does not have feature importances attribute.")
    except Exception as e:
        logging.error(f"Error during feature importance analysis: {e}")
        sys.exit(1)

def main():
    # Define file path and target column
    file_path = '/Users/ryanpope/Downloads/model_data.csv'
    target_column = 'Crime Rate'

    # Load the dataset
    data = load_data(file_path)

    # Preprocess the data (removes 'Percent_Owner_Occ' and log-transform target)
    X_train, X_test, y_train, y_test, city_column = preprocess_data(data, target_column)

    # Tune the model
    best_model = tune_random_forest(X_train, y_train)

    # Evaluate the model
    evaluate_model(best_model, X_test, y_test, city_column)

    # Analyze feature importance using the correct feature names from the preprocessed data
    analyze_feature_importance(best_model, X_train.columns)

if __name__ == "__main__":
    main()
