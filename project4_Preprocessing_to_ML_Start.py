import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import logging
import sys

# Set up logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """
    Loads the dataset from the given file path.
    
    Parameters:
        file_path (str): The path to the dataset file.
        
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    try:
        logging.info(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)
        logging.info("Data successfully loaded")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)

def preprocess_data(data, target_column):
    """
    Prepares the dataset by separating features and target, imputing missing values, and scaling features.
    
    Parameters:
        data (pd.DataFrame): The dataset.
        target_column (str): The name of the target column.
        
    Returns:
        tuple: Scaled training and testing sets, and the target variable sets.
    """
    try:
        # Separate features and target dlendent var (Crime Rate)
        logging.info(f"Preparing data by separating features from target column '{target_column}'")
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Identify numeric and categorical columns
        numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = X.select_dtypes(include=['object']).columns

        logging.info(f"Numeric columns: {numeric_columns}")
        logging.info(f"Categorical columns: {categorical_columns}")

        # Define transformers for numeric and categorical data
        numeric_transformer = SimpleImputer(strategy='mean')
        categorical_transformer = SimpleImputer(strategy='most_frequent')

        # Preprocessing for both numeric and categorical features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_columns),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
            ])

        # Split data into training and testing sets
        logging.info("Splitting data into training and testing sets")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit and transform the data
        logging.info("Imputing missing values and scaling feature data")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Scale the numeric features only
        scaler = StandardScaler(with_mean=False)  # Avoid centering the data after one-hot encoding
        X_train_scaled = scaler.fit_transform(X_train_processed)
        X_test_scaled = scaler.transform(X_test_processed)

        return X_train_scaled, X_test_scaled, y_train, y_test
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        sys.exit(1)

def train_model(X_train, y_train):
    """
    Trains a Random Forest model on the training data.
    
    Parameters:
        X_train (np.array): The scaled training features.
        y_train (pd.Series): The target training data.
        
    Returns:
        model (RandomForestRegressor): The trained model.
    """
    try:
        logging.info("Training Random Forest model")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logging.info("Model training complete")
        return model
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        sys.exit(1)

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test data.
    
    Parameters:
        model (RandomForestRegressor): The trained model.
        X_test (np.array): The scaled testing features.
        y_test (pd.Series): The target testing data.
        
    Returns:
        None
    """
    try:
        logging.info("Making predictions and evaluating the model")
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logging.info(f"Mean Squared Error (MSE): {mse}")
        logging.info(f"R-squared (R^2): {r2}")

        print(f"Mean Squared Error: {mse}")
        print(f"R^2 Score: {r2}")
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        sys.exit(1)

def analyze_feature_importance(model, feature_names):
    """
    Analyzes and displays feature importance for the trained model.
    
    Parameters:
        model (RandomForestRegressor): The trained model.
        feature_names (list): The names of the features.
        
    Returns:
        None
    """
    try:
        logging.info("Analyzing feature importances")
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names, 
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        print("\nFeature Importances:\n", feature_importance_df)
    except Exception as e:
        logging.error(f"Error during feature importance analysis: {e}")
        sys.exit(1)

def main():
    # Define file path and target column
    file_path = '/Users/ryanpope/Downloads/model_data.csv'
    target_column = 'Crime Rate'

    # Load the dataset
    data = load_data(file_path)

    # Preprocess the data
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(data, target_column)

    # Train modl
    model = train_model(X_train_scaled, y_train)

    # evaluate the model
    evaluate_model(model, X_test_scaled, y_test)

    # Analyze feature importances
    analyze_feature_importance(model, data.drop(columns=[target_column]).columns)

if __name__ == "__main__":
    main()
