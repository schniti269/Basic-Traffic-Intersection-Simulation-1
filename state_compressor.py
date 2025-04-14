import numpy as np
import os
import pickle
from datetime import datetime

# Try to import TensorFlow, set flag accordingly
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Input, Dense

    # Configure TensorFlow to use GPU if available
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # Display GPU information
            print(f"GPU acceleration enabled. Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")

            # Use mixed precision for better performance on compatible GPUs
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision enabled (float16/float32)")
        except RuntimeError as e:
            print(f"Error configuring GPUs: {e}")
    else:
        print("No GPU found. Using CPU for computation.")

    TF_AVAILABLE = True
    print("TensorFlow detected. Will use autoencoder for state compression.")
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not detected. Will use numpy-based dimensionality reduction.")

# Constants
DATA_DIR = "state_data"
MODEL_DIR = "state_models"
DATA_FILE = os.path.join(DATA_DIR, "collected_states.pkl")
TF_MODEL_FILE = os.path.join(MODEL_DIR, "autoencoder.h5")
BOTTLENECK_SIZE = 8  # Size of compressed state representation


class StateCompressor:
    def __init__(self):
        self.collected_data = []
        self.autoencoder = None
        self.data_mean = None
        self.data_std = None
        self.encoder_type = None

        # Create necessary directories
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Try to load existing model
        self.load_model()

    def collect_state(self, state_data):
        """
        Collect a state vector for later training
        """
        self.collected_data.append(state_data)

        # Save data periodically
        if len(self.collected_data) % 1000 == 0:
            print(f"Collected {len(self.collected_data)} state samples")
            self.save_collected_data()

    def save_collected_data(self):
        """Save collected state data to file"""
        with open(DATA_FILE, "wb") as f:
            pickle.dump(self.collected_data, f)
        print(f"Saved {len(self.collected_data)} state samples to {DATA_FILE}")

    def load_collected_data(self):
        """Load collected state data from file"""
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "rb") as f:
                self.collected_data = pickle.load(f)
            print(f"Loaded {len(self.collected_data)} state samples from {DATA_FILE}")
            return True
        return False

    def train_model(self, force=False, min_samples=5000):
        """
        Train a state compression model if enough data is available

        Args:
            force: Force retraining even if a model already exists
            min_samples: Minimum number of samples required for training

        Returns:
            True if a model was trained, False otherwise
        """
        # Check if we need to load data
        if len(self.collected_data) == 0:
            self.load_collected_data()

        # Check if we have enough data
        if len(self.collected_data) < min_samples:
            print(
                f"Not enough data for training. Need at least {min_samples} samples, have {len(self.collected_data)}"
            )
            return False

        # Check if we already have a model and aren't forcing retraining
        if self.autoencoder is not None and not force:
            print("Model already exists. Use force=True to retrain.")
            return False

        print(f"Training state compressor on {len(self.collected_data)} samples...")

        # Convert data to numpy array
        data = np.array(self.collected_data, dtype=np.float32)

        # Normalize data
        self.data_mean = np.mean(data, axis=0)
        self.data_std = (
            np.std(data, axis=0) + 1e-6
        )  # Add small epsilon to avoid division by zero
        data_normalized = (data - self.data_mean) / self.data_std

        # Train appropriate model
        if TF_AVAILABLE:
            self._train_autoencoder(data_normalized)
            self.encoder_type = "autoencoder"
        else:
            self._train_pca(data_normalized)
            self.encoder_type = "pca"

        # Save the trained model
        self.save_model()
        return True

    def _train_autoencoder(self, data_normalized):
        """Train TensorFlow autoencoder model"""
        print("Training autoencoder model...")

        # Define autoencoder architecture
        input_dim = data_normalized.shape[1]

        # Input layer
        input_layer = Input(shape=(input_dim,))

        # Encoder
        encoded = Dense(32, activation="relu")(input_layer)
        encoded = Dense(16, activation="relu")(encoded)
        bottleneck = Dense(BOTTLENECK_SIZE, activation="relu", name="bottleneck")(
            encoded
        )

        # Decoder
        decoded = Dense(16, activation="relu")(bottleneck)
        decoded = Dense(32, activation="relu")(decoded)
        output_layer = Dense(input_dim, activation="linear")(decoded)

        # Create and compile full autoencoder model
        full_autoencoder = Model(input_layer, output_layer)

        # Use a better optimizer with learning rate scheduling
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.9
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        full_autoencoder.compile(optimizer=optimizer, loss="mse")

        # Add callbacks for early stopping and model checkpointing
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(MODEL_DIR, "autoencoder_checkpoint.h5"),
                monitor="val_loss",
                save_best_only=True,
            ),
        ]

        # Train the model
        full_autoencoder.fit(
            data_normalized,
            data_normalized,
            epochs=100,  # Increase epochs but use early stopping
            batch_size=64,  # Larger batch size for GPU efficiency
            shuffle=True,
            verbose=1,
            validation_split=0.2,
            callbacks=callbacks,
        )

        # Create encoder model (only the bottleneck part)
        self.autoencoder = Model(inputs=input_layer, outputs=bottleneck)

        # Save TensorFlow model
        # Use the modern .keras format instead of .h5
        keras_model_file = os.path.join(MODEL_DIR, "autoencoder.keras")
        self.autoencoder.save(keras_model_file)
        print(f"Autoencoder model saved to {keras_model_file}")

        # Also save in .h5 format for backward compatibility
        self.autoencoder.save(TF_MODEL_FILE)
        print(f"Autoencoder model also saved in legacy format to {TF_MODEL_FILE}")

    def _train_pca(self, data_normalized):
        """Train PCA compressor using numpy"""
        print("Training PCA state compressor...")

        # Calculate covariance matrix
        cov_matrix = np.cov(data_normalized.T)

        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        components = eigenvectors[:, idx[:BOTTLENECK_SIZE]]

        # Create a simple model that can apply the PCA transformation
        class PCAModel:
            def __init__(self, components):
                self.components = components

            def predict(self, x):
                return np.dot(x, self.components)

        self.autoencoder = PCAModel(components)

        # Save PCA model
        with open(PCA_MODEL_FILE, "wb") as f:
            pickle.dump(
                {
                    "components": components,
                },
                f,
            )
        print(f"PCA model saved to {PCA_MODEL_FILE}")

    def save_model(self):
        """Save model parameters"""
        # Save normalization parameters separately (used by both model types)
        with open(os.path.join(MODEL_DIR, "normalization_params.pkl"), "wb") as f:
            pickle.dump(
                {
                    "data_mean": self.data_mean,
                    "data_std": self.data_std,
                    "encoder_type": self.encoder_type,
                },
                f,
            )

    def load_model(self):
        """Load a previously trained model"""
        # First check if normalization parameters exist
        norm_file = os.path.join(MODEL_DIR, "normalization_params.pkl")
        if not os.path.exists(norm_file):
            print("No trained model found")
            return False

        # Load normalization parameters
        with open(norm_file, "rb") as f:
            params = pickle.load(f)
            self.data_mean = params["data_mean"]
            self.data_std = params["data_std"]
            self.encoder_type = params.get("encoder_type", "unknown")

        # Load the appropriate model
        if self.encoder_type == "autoencoder" and TF_AVAILABLE:
            # First try the modern format
            keras_model_file = os.path.join(MODEL_DIR, "autoencoder.keras")
            if os.path.exists(keras_model_file):
                try:
                    self.autoencoder = load_model(keras_model_file)
                    print(f"Loaded TensorFlow autoencoder from {keras_model_file}")
                    return True
                except Exception as e:
                    print(
                        f"Error loading TensorFlow model from {keras_model_file}: {e}"
                    )

            # Try the legacy .h5 format
            if os.path.exists(TF_MODEL_FILE):
                try:
                    self.autoencoder = load_model(TF_MODEL_FILE)
                    print(f"Loaded TensorFlow autoencoder from {TF_MODEL_FILE}")
                    return True
                except Exception as e:
                    print(f"Error loading TensorFlow model from {TF_MODEL_FILE}: {e}")

        # Try loading PCA model as fallback
        if os.path.exists(PCA_MODEL_FILE):
            try:
                with open(PCA_MODEL_FILE, "rb") as f:
                    pca_data = pickle.load(f)

                class PCAModel:
                    def __init__(self, components):
                        self.components = components

                    def predict(self, x):
                        return np.dot(x, self.components)

                self.autoencoder = PCAModel(pca_data["components"])
                print(f"Loaded PCA model from {PCA_MODEL_FILE}")
                return True
            except Exception as e:
                print(f"Error loading PCA model: {e}")

        print("Could not load any state compression model")
        return False

    def encode(self, state_data):
        """
        Encode a state vector using the trained model

        Args:
            state_data: Raw state vector to encode

        Returns:
            Compressed state representation or None if no model is available
        """
        if self.autoencoder is None:
            return None

        # Convert to numpy array
        x = np.array([state_data], dtype=np.float32)

        # Normalize
        x_normalized = (x - self.data_mean) / self.data_std

        # Get encoded representation
        # Use TensorFlow's efficient prediction if available
        if (
            TF_AVAILABLE
            and hasattr(self.autoencoder, "predict")
            and callable(getattr(self.autoencoder, "predict"))
        ):
            return self.autoencoder.predict(x_normalized, verbose=0)[0]
        elif hasattr(self.autoencoder, "predict") and callable(
            getattr(self.autoencoder, "predict")
        ):
            return self.autoencoder.predict(x_normalized)[0]
        else:
            print("Warning: Model doesn't have a predict method")
            return None

    def is_ready(self):
        """Check if the model is ready to use"""
        return self.autoencoder is not None


# Command-line training utility
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a state compression model")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument(
        "--force", action="store_true", help="Force retraining even if a model exists"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1000,
        help="Minimum number of samples required for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (larger values may work better on GPUs)",
    )
    args = parser.parse_args()

    compressor = StateCompressor()

    if args.train:
        compressor.train_model(force=args.force, min_samples=args.min_samples)
    elif compressor.is_ready():
        print("Model is ready for use")
        print(
            f"Compression: {compressor.collected_data[0].shape if len(compressor.collected_data) > 0 else 'unknown'} -> {BOTTLENECK_SIZE}"
        )
    else:
        print("No model is available. Collect data and train one.")
