import numpy as np
import random
import pickle
import os
import matplotlib.pyplot as plt
from datetime import datetime
from simulation import simulate
from base_model import Model
from simulation_config import DEFAULT_SCAN_ZONE_CONFIG, vehicleTypes


class TDLearningModel(Model):
    """
    TD-Learning Modell für die Ampelsteuerung - krasse KI!
    """

    def __init__(
        self,
        learning_rate=0.01,
        discount_factor=0.9,
        exploration_rate=0.02,
        exploration_decay=0.999,
        min_exploration_rate=0.01,
    ):
        super().__init__()
        self.learning_rate = learning_rate  # Alpha: Lern-Speed
        self.discount_factor = discount_factor  # Gamma: Wie wichtig future Rewards sind
        self.exploration_rate = (
            exploration_rate  # Epsilon: Neues ausprobieren vs. altes nutzen
        )
        self.exploration_decay = (
            exploration_decay  # Wie schnell weniger ausprobiert wird
        )
        self.min_exploration_rate = min_exploration_rate  # Minimum Exploration-Rate

        # Q-Table als Dict für State-Action Paare
        self.q_table = {}

        # Track vorheriger State, Action und Reward für TD-Learning
        self.previous_state = None
        self.previous_action = None
        self.tick_counter = 0
        self.min_action_duration = 200  # Minimum Dauer für eine Action in Ticks

    def state_to_key(self, state):
        """
        State-Dict in hashbaren Key für Q-Table umwandeln.
        Vereinfacht den State für besseres Lernen.
        """
        # Vereinfachte Version vom State bauen
        simplified_state = []

        # Fahrzeuge pro Richtung zählen
        for direction in ["right", "left", "up", "down"]:
            total = sum(state[direction].values())
            simplified_state.append(total)

        # Nord-Süd vs Ost-West Verkehrsverhältnis checken
        ns_traffic = simplified_state[2] + simplified_state[3]  # up + down
        ew_traffic = simplified_state[0] + simplified_state[1]  # right + left

        # Verkehrsverhältnis-Indikator (diskretisiert)
        if ns_traffic == 0 and ew_traffic == 0:
            ratio = 0
        elif ns_traffic == 0:
            ratio = 5  # Voll Ost-West Fokus
        elif ew_traffic == 0:
            ratio = -5  # Voll Nord-Süd Fokus
        else:
            ratio = int(5 * (ew_traffic - ns_traffic) / (ew_traffic + ns_traffic))

        simplified_state.append(ratio)

        # In Tuple für Hashing umwandeln
        return tuple(simplified_state)

    def get_action(self, state, history):
        """
        Action basierend auf aktuellem State mit Q-Table wählen.

        Args:
            state (dict): Dict mit Fahrzeugzahlen für jede Richtung und Typ
            history (list): Liste der letzten Actions vom Modell

        Returns:
            bool: True für Nord-Süd grün, False für Ost-West grün
        """
        # Tick-Zähler erhöhen
        self.tick_counter += 1

        # Wenn Mindestdauer nicht erreicht, bleib bei vorheriger Action
        if (
            self.previous_action is not None
            and self.tick_counter < self.min_action_duration
        ):
            return self.previous_action

        # State in hashbaren Key umwandeln
        state_key = self.state_to_key(state)

        # Wenn State nicht in Q-Table, füg ihn hinzu
        if state_key not in self.q_table:
            self.q_table[state_key] = {
                True: 0.0,
                False: 0.0,
            }  # Q-Werte für beide Actions initialisieren

        # Zwischen Exploration und Exploitation wählen
        if random.random() < self.exploration_rate:
            # Exploration: Random Action
            action = random.choice([True, False])
        else:
            # Exploitation: Beste Action laut Q-Table
            if self.q_table[state_key][True] > self.q_table[state_key][False]:
                action = True
            elif self.q_table[state_key][False] > self.q_table[state_key][True]:
                action = False
            else:
                # Bei gleichen Q-Werten, random wählen
                action = random.choice([True, False])

        # Tick-Zähler zurücksetzen wenn Action gewechselt
        if self.previous_action != action:
            self.tick_counter = 0

        # Aktuellen State und Action für nächstes Update speichern
        self.previous_state = state_key
        self.previous_action = action

        return action

    def update_q_value(self, state, action, reward, next_state):
        """
        Q-Wert für State-Action Paar mit TD-Learning Formel updaten.

        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]

        Args:
            state: State-Key
            action: Ausgeführte Action
            reward: Erhaltener Reward
            next_state: Resultierender State-Key
        """
        if state not in self.q_table:
            self.q_table[state] = {True: 0.0, False: 0.0}

        if next_state not in self.q_table:
            self.q_table[next_state] = {True: 0.0, False: 0.0}

        # Max Q-Wert für nächsten State berechnen
        max_next_q = max(self.q_table[next_state].values())

        # Aktueller Q-Wert
        current_q = self.q_table[state][action]

        # TD-Update Formel
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        # Q-Wert updaten
        self.q_table[state][action] = new_q

    def decay_exploration_rate(self):
        """Exploration-Rate reduzieren - von Ausprobieren zu Ausnutzen wechseln"""
        self.exploration_rate = max(
            self.min_exploration_rate, self.exploration_rate * self.exploration_decay
        )

    def save_model(self, filename=None):
        """Model in Datei speichern"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"td_model_{timestamp}.pkl"

        with open(filename, "wb") as f:
            pickle.dump(
                {
                    "q_table": self.q_table,
                    "learning_rate": self.learning_rate,
                    "discount_factor": self.discount_factor,
                    "exploration_rate": self.exploration_rate,
                    "exploration_decay": self.exploration_decay,
                    "min_exploration_rate": self.min_exploration_rate,
                },
                f,
            )

        print(f"Model saved to {filename}")

    def load_model(self, filename):
        """Model aus Datei laden"""
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.q_table = data["q_table"]
            self.learning_rate = data["learning_rate"]
            self.discount_factor = data["discount_factor"]
            self.exploration_rate = data["exploration_rate"]
            self.exploration_decay = data["exploration_decay"]
            self.min_exploration_rate = data["min_exploration_rate"]

        print(f"Model loaded from {filename}")


def train_model(episodes=100, ticks_per_episode=36000, save_interval=1):
    """
    TD-Learning Model mit Traffic-Sim trainieren.

    Args:
        episodes: Anzahl der Training-Episoden
        ticks_per_episode: Ticks pro Episode
        save_interval: Model alle X Episoden speichern

    Returns:
        Das trainierte Model
    """
    model = TDLearningModel()

    # Models-Ordner erstellen wenn nötig
    if not os.path.exists("models"):
        os.makedirs("models")

    # Rewards für Auswertung tracken
    rewards_history = []

    # Gesamt-Trainingsfortschritt tracken
    total_start_time = datetime.now()

    print(f"Starting training for {episodes} episodes")
    for episode in range(1, episodes + 1):
        # Gesamtfortschritt berechnen
        overall_progress = ((episode - 1) / episodes) * 100

        # Fortschrittsbalken bauen
        bar_length = 20
        filled_length = int(bar_length * (episode - 1) // episodes)
        progress_bar = "#" * filled_length + "-" * (bar_length - filled_length)

        print(f"\nOverall progress: |{progress_bar}| {overall_progress:.1f}%")
        print(f"Episode {episode}/{episodes}")
        episode_start_time = datetime.now()

        # Eine Episode simulieren
        reward = simulate(
            model,
            TRAINING=True,
            TICKS_PER_SECOND=float("inf"),
            NO_OF_TICKS=ticks_per_episode,
        )
        rewards_history.append(reward)

        episode_duration = (datetime.now() - episode_start_time).total_seconds()

        # Restzeit berechnen
        avg_episode_time = (datetime.now() - total_start_time).total_seconds() / episode
        est_time_remaining = avg_episode_time * (episodes - episode)

        print(f"Episode {episode} completed with reward: {reward}")
        print(f"Episode duration: {episode_duration:.2f} seconds")
        print(f"Estimated time remaining: {est_time_remaining/60:.2f} minutes")

        # Exploration-Rate reduzieren
        model.decay_exploration_rate()
        print(f"Current exploration rate: {model.exploration_rate:.4f}")

        # Model regelmäßig speichern
        if episode % save_interval == 0:
            model.save_model(f"models/td_model_episode_{episode}.pkl")

    # Gesamttrainingszeit ausgeben
    total_duration = (datetime.now() - total_start_time).total_seconds()
    print(f"\nTotal training time: {total_duration/60:.2f} minutes")

    # Finales Model speichern
    model.save_model(f"models/td_model_final.pkl")

    # Rewards-Verlauf plotten
    plot_rewards(rewards_history)

    # Training-Zusammenfassung
    print("\nTraining completed!")
    print(f"Final exploration rate: {model.exploration_rate:.4f}")
    print(
        f"Average reward over all episodes: {sum(rewards_history) / len(rewards_history):.2f}"
    )
    print(f"Best episode reward: {max(rewards_history):.2f}")

    return model


def evaluate_model(model_path, num_evaluations=5, ticks_per_evaluation=36000):
    """
    Trainiertes Model bewerten.

    Args:
        model_path: Pfad zum gespeicherten Model
        num_evaluations: Anzahl der Test-Runs
        ticks_per_evaluation: Ticks pro Test

    Returns:
        Durchschnittlicher Reward über alle Tests
    """
    model = TDLearningModel()
    model.load_model(model_path)

    # Exploration auf 0 setzen für deterministisches Verhalten
    model.exploration_rate = 0.0

    rewards = []

    print(f"Evaluating model {model_path}")
    for i in range(num_evaluations):
        print(f"Evaluation run {i+1}/{num_evaluations}")
        reward = simulate(
            model, TRAINING=True, TICKS_PER_SECOND=60, NO_OF_TICKS=ticks_per_evaluation
        )
        rewards.append(reward)
        print(f"Evaluation {i+1} completed with reward: {reward}")

    avg_reward = sum(rewards) / len(rewards)
    print(f"\nEvaluation completed!")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Min reward: {min(rewards):.2f}")
    print(f"Max reward: {max(rewards):.2f}")

    return avg_reward


def run_trained_model(model_path):
    """
    Trainiertes Model in der Sim laufen lassen zum Anschauen.

    Args:
        model_path: Pfad zum gespeicherten Model
    """
    model = TDLearningModel()
    model.load_model(model_path)

    # Exploration auf 0 für deterministisches Verhalten
    model.exploration_rate = 0.0

    print(f"Running model {model_path} in simulation")
    reward = simulate(model, TRAINING=False, TICKS_PER_SECOND=60)
    print(f"Simulation completed with reward: {reward}")


def plot_rewards(rewards_history):
    """
    Rewards über Episoden plotten um Trainingsfortschritt zu visualisieren.

    Args:
        rewards_history: Liste der Rewards aus jeder Training-Episode
    """
    # Plots-Ordner erstellen wenn nötig
    if not os.path.exists("plots"):
        os.makedirs("plots")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rewards_history) + 1), rewards_history, marker="o")
    plt.title("Reward Progress During Training")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)

    # Smoothed Trendlinie hinzufügen (gleitender Durchschnitt)
    if len(rewards_history) > 5:
        window_size = min(5, len(rewards_history) // 5)
        smoothed = np.convolve(
            rewards_history, np.ones(window_size) / window_size, mode="valid"
        )
        plt.plot(
            range(window_size, len(rewards_history) + 1), smoothed, "r-", linewidth=2
        )

    # Plot speichern
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"plots/rewards_plot_{timestamp}.png")
    print(f"Rewards plot saved to plots/rewards_plot_{timestamp}.png")

    # Plot anzeigen (für Headless-Environments auskommentieren)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TD Learning für Ampelsteuerung")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate", "run"],
        help="Mode: train, evaluate, oder run",
    )
    parser.add_argument(
        "--model", type=str, help="Pfad zur Model-Datei (für evaluate und run)"
    )
    parser.add_argument(
        "--episodes", type=int, default=50, help="Anzahl der Training-Episoden"
    )
    parser.add_argument("--ticks", type=int, default=36000, help="Ticks pro Episode")
    parser.add_argument(
        "--save_interval", type=int, default=10, help="Speicher-Intervall in Episoden"
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_model(
            episodes=args.episodes,
            ticks_per_episode=args.ticks,
            save_interval=args.save_interval,
        )
    elif args.mode == "evaluate":
        if args.model is None:
            print("Error: Model-Pfad für Evaluation-Mode benötigt")
        else:
            evaluate_model(args.model, ticks_per_evaluation=args.ticks)
    elif args.mode == "run":
        if args.model is None:
            print("Error: Model-Pfad für Run-Mode benötigt")
        else:
            run_trained_model(args.model)
