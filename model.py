
from config import *
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
import gc
import psutil

def build_model(params):
    model = Sequential([
        Conv2D(params['filters1'], (3, 3), activation="relu", input_shape=(256, 256, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(params['filters2'], (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(params['units1'], activation="relu"),
        Dropout(params['dropout']),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=params['lr']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def objective_function(params, X_train, y_train):
    model = build_model(params)
    history = model.fit(X_train, y_train, epochs=3, batch_size=params['batch_size'], verbose=0, validation_split=0.1)
    val_acc = max(history.history['val_accuracy'])

    # Logging
    print(f"Evaluated Params: {params}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print("Memory usage (MB):", psutil.Process().memory_info().rss / 1024 ** 2)

    # Cleanup
    del history
    del model
    K.clear_session()
    gc.collect()

    return -val_acc
