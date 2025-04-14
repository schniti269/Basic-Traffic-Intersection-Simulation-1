import numpy as np
import os
import pickle
from datetime import datetime
import random
from base_model import Model

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )

    TF_AVAILABLE = True

    # Configure TensorFlow to use GPU if available
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(
                f"GPU acceleration enabled for neural model. Found {len(gpus)} GPU(s)"
            )

            # Use mixed precision for better performance on compatible GPUs
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision enabled for neural model")
        except RuntimeError as e:
            print(f"Error configuring GPUs for neural model: {e}")
    else:
        print("No GPU found. Using CPU for neural model computation.")
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not detected. Neural model won't be available.")


class NeuralModel(Model):
    """
    Neural network model for traffic signal control based on compressed state representation.
    This model inherits from the base Model class and implements a neural network
    to make traffic control decisions based on compressed state data.
    """

    def __init__(
        self,
        input_dim=8,
        learning_rate=0.001,
        exploration_rate=0.1,
        exploration_decay=0.995,
        min_exploration_rate=0.01,
        model_path=None,
    ):
        super().__init__()

        # Check if TensorFlow is available
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for NeuralModel")

        self.input_dim = input_dim  # Dimension of the compressed state vector
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

        # For experience replay
        self.experience_buffer = []
        self.max_buffer_size = 10000
        self.batch_size = 64
        self.min_experiences = 1000  # Min experiences before training

        # Create model directory
        os.makedirs("neural_models", exist_ok=True)

        # Create or load the model
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
            print(f"Loaded neural model from {model_path}")
        else:
            self.model = self._build_model()
            print("Created new neural model")

        # For action tracking
        self.last_state = None
        self.last_action = None
        self.tick_counter = 0
        self.min_action_duration = 200  # Minimum ticks to maintain an action

        # Stats for training
        self.train_count = 0
        self.rewards = []

    def _build_model(self):
        """Build the neural network model"""
        model = Sequential(
            [
                # Input layer with the compressed state dimension
                Dense(32, activation="relu", input_shape=(self.input_dim,)),
                BatchNormalization(),
                # Hidden layers
                Dense(64, activation="relu"),
                Dropout(0.2),
                BatchNormalization(),
                Dense(32, activation="relu"),
                Dropout(0.2),
                BatchNormalization(),
                # Output layer - single neuron with sigmoid activation for binary decision
                # (True for north-south green, False for east-west green)
                Dense(1, activation="sigmoid"),
            ]
        )

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def get_action(self, state, history):
        """
        Determine the traffic light control action based on the current state.

        Args:
            state: Either a dict with vehicle counts (will be ignored) or
                  a compressed state vector (will be used directly)
            history: List of previous actions (not used in this implementation)

        Returns:
            bool: True for north-south green, False for east-west green
        """
        # Increment tick counter
        self.tick_counter += 1

        # If we haven't reached the minimum action duration, stick with the previous action
        if (
            self.last_action is not None
            and self.tick_counter < self.min_action_duration
        ):
            return self.last_action

        # Reset tick counter
        self.tick_counter = 0

        # If we received a dict state, it's the old format - we need compressed state
        if isinstance(state, dict):
            print(
                "Warning: NeuralModel received dict state instead of compressed vector"
            )
            # Just make a simple decision based on traffic in each direction
            ns_traffic = sum(state["up"].values()) + sum(state["down"].values())
            ew_traffic = sum(state["left"].values()) + sum(state["right"].values())
            action = ns_traffic > ew_traffic
            self.last_action = action
            return action

        # Now we know state is a compressed vector
        state_vector = np.array(state).reshape(1, -1)

        # Log every 1000th state vector for debugging
        if random.random() < 0.001:
            print(f"State vector sample: {state}")

        # Choose between exploration and exploitation
        is_exploring = random.random() < self.exploration_rate
        if is_exploring:
            # Exploration: choose a random action
            action = random.choice([True, False])
            action_src = "exploration"
        else:
            # Exploitation: use the model to predict the action
            prediction = self.model.predict(state_vector, verbose=0)[0][0]
            action = prediction > 0.5  # Convert to boolean
            action_src = f"model (p={prediction:.2f})"

            # Periodically log model predictions
            if random.random() < 0.01:
                action_name = "NS" if action else "EW"
                print(f"Model predicted {action_name} ({prediction:.4f})")

        # Store the current state and action for later training
        self.last_state = state
        self.last_action = action

        return action

    def add_experience(self, state, action, reward, next_state):
        """
        Add an experience to the replay buffer

        Args:
            state: Compressed state vector
            action: Boolean (True/False) action taken
            reward: Reward received
            next_state: Next compressed state vector
        """
        # Store the experience
        self.experience_buffer.append((state, action, reward, next_state))

        # Print significant rewards for visibility
        if abs(reward) > 10:
            action_str = "NS" if action else "EW"
            print(f"High reward experience: {reward:.2f} for action {action_str}")

        # Limit buffer size
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)

        # Print buffer stats periodically
        buffer_size = len(self.experience_buffer)
        if buffer_size % 1000 == 0 and buffer_size > 0:
            actions = [1 if exp[1] else 0 for exp in self.experience_buffer]
            ns_percent = sum(actions) / len(actions) * 100
            avg_reward = np.mean([exp[2] for exp in self.experience_buffer])
            print(
                f"Experience buffer: {buffer_size} entries, {ns_percent:.1f}% NS actions, avg reward: {avg_reward:.2f}"
            )

    def train(self, force=False):
        """
        Train the model on a batch of experiences from the replay buffer

        Args:
            force: Force training even if we don't have enough experiences

        Returns:
            Dictionary with training metrics or None if no training occurred
        """
        # Check if we have enough experiences to train
        if len(self.experience_buffer) < self.min_experiences and not force:
            if force:
                print(
                    f"Forcing training with only {len(self.experience_buffer)} experiences"
                )
            else:
                return None

        # Sample a batch of experiences
        if len(self.experience_buffer) > self.batch_size:
            batch = random.sample(self.experience_buffer, self.batch_size)
            batch_size_str = (
                f"sample of {self.batch_size}/{len(self.experience_buffer)}"
            )
        else:
            batch = self.experience_buffer
            batch_size_str = f"all {len(self.experience_buffer)}"

        # Prepare data for training
        states = np.array([exp[0] for exp in batch])
        actions = np.array([1.0 if exp[1] else 0.0 for exp in batch])
        rewards = np.array([exp[2] for exp in batch])

        # Print distribution of actions in batch
        ns_actions = np.sum(actions)
        ew_actions = len(actions) - ns_actions
        print(
            f"Training on {batch_size_str} experiences - Actions: {ns_actions:.0f} NS, {ew_actions:.0f} EW - Avg reward: {np.mean(rewards):.2f}"
        )

        # Train the model
        history = self.model.fit(
            states,
            actions,
            batch_size=min(self.batch_size, len(batch)),
            epochs=1,
            verbose=0,
        )

        # Decay exploration rate
        old_rate = self.exploration_rate
        self.exploration_rate = max(
            self.min_exploration_rate, self.exploration_rate * self.exploration_decay
        )

        # Track training stats
        self.train_count += 1
        avg_reward = np.mean(rewards)
        self.rewards.append(avg_reward)

        # Print more detailed stats periodically
        if self.train_count % 10 == 0 or self.train_count < 10:
            print(
                f"Training #{self.train_count}, Avg Reward: {avg_reward:.2f}, Exploration: {self.exploration_rate:.4f} (was {old_rate:.4f})"
            )

            # Print loss and accuracy if available
            if "loss" in history.history:
                print(f"  Loss: {history.history['loss'][0]:.4f}", end="")
                if "accuracy" in history.history:
                    print(f", Accuracy: {history.history['accuracy'][0]:.4f}")
                else:
                    print()

        return history.history

    def save_model(self, filename=None):
        """Save the model to a file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"neural_models/neural_model_{timestamp}.keras"

        self.model.save(filename)
        print(f"Neural model saved to {filename}")

        # Save training stats
        stats_file = os.path.splitext(filename)[0] + "_stats.pkl"
        with open(stats_file, "wb") as f:
            pickle.dump(
                {
                    "rewards": self.rewards,
                    "train_count": self.train_count,
                    "exploration_rate": self.exploration_rate,
                },
                f,
            )

    def load_model(self, filename):
        """Load the model from a file"""
        self.model = load_model(filename)
        print(f"Neural model loaded from {filename}")

        # Try to load training stats
        stats_file = os.path.splitext(filename)[0] + "_stats.pkl"
        if os.path.exists(stats_file):
            with open(stats_file, "rb") as f:
                stats = pickle.load(f)
                self.rewards = stats.get("rewards", [])
                self.train_count = stats.get("train_count", 0)
                self.exploration_rate = stats.get(
                    "exploration_rate", self.exploration_rate
                )


def train_neural_model(episodes=100, ticks_per_episode=36000):
    """
    Train a neural model using the compressed state representation.

    Args:
        episodes: Number of training episodes
        ticks_per_episode: Number of ticks per episode

    Returns:
        Trained neural model
    """
    from simulation import simulate
    from state_compressor import StateCompressor
    import time

    # First check if we have a trained state compressor
    compressor = StateCompressor()
    if not compressor.is_ready():
        print("Error: State compressor must be trained first!")
        print("Run 'python state_compressor.py --train' to train the state compressor")
        return None

    # Create the neural model
    model = NeuralModel(
        input_dim=8,  # Default bottleneck size in state_compressor
        exploration_rate=0.5,  # Start with high exploration
    )

    print(f"Starting neural model training for {episodes} episodes")
    print(f"Initial exploration rate: {model.exploration_rate:.4f}")
    print(f"Buffer capacity: {model.max_buffer_size} experiences")
    print(f"Minimum action duration: {model.min_action_duration} ticks")

    total_reward = 0
    start_time = time.time()

    for episode in range(1, episodes + 1):
        episode_start = time.time()
        print(f"\n================================================")
        print(f"Episode {episode}/{episodes}")
        print(f"Current exploration rate: {model.exploration_rate:.4f}")
        print(
            f"Experience buffer size: {len(model.experience_buffer)}/{model.max_buffer_size}"
        )
        print(f"Training iterations so far: {model.train_count}")

        # Run one episode of simulation
        reward = simulate(
            model,
            TRAINING=True,
            TICKS_PER_SECOND=float("inf"),
            NO_OF_TICKS=ticks_per_episode,
            USE_ADVANCED_STATE=True,  # Use the compressed state
        )

        total_reward += reward
        avg_reward = total_reward / episode
        episode_duration = time.time() - episode_start

        # Calculate estimated time remaining
        elapsed_time = time.time() - start_time
        avg_episode_time = elapsed_time / episode
        remaining_episodes = episodes - episode
        est_time_remaining = avg_episode_time * remaining_episodes

        # Format as hours:minutes:seconds
        est_hours = int(est_time_remaining // 3600)
        est_minutes = int((est_time_remaining % 3600) // 60)
        est_seconds = int(est_time_remaining % 60)

        print(f"Episode {episode} completed in {episode_duration:.2f} seconds")
        print(f"Reward: {reward:.2f}, Average reward: {avg_reward:.2f}")
        print(
            f"Estimated time remaining: {est_hours:02d}:{est_minutes:02d}:{est_seconds:02d}"
        )

        # Save model periodically
        if episode % 10 == 0 or episode == episodes:
            save_path = f"neural_models/neural_model_episode_{episode}.keras"
            model.save_model(save_path)
            print(f"Model saved to {save_path}")

        # Summarize experience buffer stats
        if len(model.experience_buffer) > 0:
            rewards = [exp[2] for exp in model.experience_buffer]
            actions = [exp[1] for exp in model.experience_buffer]
            ns_percent = sum(1 for a in actions if a) / len(actions) * 100
            print(f"Experience buffer: {len(model.experience_buffer)} entries")
            print(
                f"Action distribution: {ns_percent:.1f}% NS, {100-ns_percent:.1f}% EW"
            )
            print(
                f"Reward stats: min={min(rewards):.2f}, max={max(rewards):.2f}, avg={np.mean(rewards):.2f}"
            )

    # Save the final model
    final_path = f"neural_models/neural_model_final.keras"
    model.save_model(final_path)

    # Calculate and format total training time
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print("\n================================================")
    print("Training completed!")
    print(f"Total training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"Final exploration rate: {model.exploration_rate:.4f}")
    print(f"Total training iterations: {model.train_count}")
    print(f"Final average reward: {avg_reward:.2f}")
    print(f"Final model saved to {final_path}")

    return model


def run_trained_neural_model(model_path):
    """
    Run a trained neural model in the simulation.

    Args:
        model_path: Path to the saved model
    """
    from simulation import simulate
    import time

    # Create and load the model
    model = NeuralModel()
    model.load_model(model_path)

    # Set exploration rate to 0 for deterministic policy
    model.exploration_rate = 0.0
    print(f"Running neural model {model_path} in simulation")
    print(
        f"Exploration rate set to {model.exploration_rate} for deterministic evaluation"
    )

    start_time = time.time()
    reward = simulate(
        model, TRAINING=False, TICKS_PER_SECOND=60, USE_ADVANCED_STATE=True
    )
    runtime = time.time() - start_time

    print(f"Simulation completed in {runtime:.2f} seconds")
    print(f"Final reward: {reward:.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train or run a neural traffic control model"
    )
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument(
        "--episodes", type=int, default=50, help="Number of training episodes"
    )
    parser.add_argument("--run", type=str, help="Path to a trained model to run")

    args = parser.parse_args()

    if args.train:
        train_neural_model(episodes=args.episodes)
    elif args.run:
        run_trained_neural_model(args.run)
    else:
        parser.print_help()
