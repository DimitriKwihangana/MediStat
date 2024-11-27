import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

def preprocessing(csv_path):
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # List of columns to drop
    columns_to_drop = [
        'histogram_mean', 
        'mean_value_of_long_term_variability', 
        'histogram_variance', 
        'severe_decelerations'
    ]
    
    # Drop the columns
    df = df.drop(columns=columns_to_drop)
    
    X = df.drop(columns=['fetal_health'])
    y = df['fetal_health']
    
    # Initialize the StandardScaler, fit and transform the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    y = y - 1  
    
    trainX, testX, trainY, testY = train_test_split(X_scaled_df, y, test_size=0.2, random_state=101)
    
    # Convert target variables to one-hot encoding
    num_classes = len(y.unique())  # Ensure num_classes matches the number of unique classes
    trainY = to_categorical(trainY, num_classes=num_classes)
    testY = to_categorical(testY, num_classes=num_classes)
    
    # Return processed data
    return trainX, testX, trainY, testY
