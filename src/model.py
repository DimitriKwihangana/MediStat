from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Define an alternative model
def create_model(input_shape):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_shape,)),  # Fewer units for simplicity
        BatchNormalization(),
        Dropout(0.2),  # Less dropout to retain more information

        Dense(64, activation="relu", kernel_regularizer=l2(0.01)),  # L2 regularization only
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation="relu"),  # Reduced depth for faster training
        Dropout(0.2),

        Dense(3, activation="softmax")  # Output layer for 3 classes
    ])
    
    # Callbacks
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Slightly higher learning rate for experimentation
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model, [early_stopping, reduce_lr]
