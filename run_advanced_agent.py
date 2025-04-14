import argparse
from advanced_traffic_agent import (
    AdvancedTrafficAgent,
    train_advanced_agent,
    evaluate_advanced_agent,
)
from simulation import simulate


def main():
    parser = argparse.ArgumentParser(
        description="Train or run an advanced traffic control agent"
    )

    # Add mode selection argument
    parser.add_argument(
        "mode",
        choices=["train", "evaluate", "simulate"],
        help="Mode to run: train a new agent, evaluate an existing one, or simulate with an agent",
    )

    # Training parameters
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of training episodes (default: 100)",
    )
    parser.add_argument(
        "--ticks", type=int, default=36000, help="Ticks per episode (default: 36000)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save model every N episodes (default: 10)",
    )

    # Evaluation parameters
    parser.add_argument(
        "--num-evals",
        type=int,
        default=5,
        help="Number of evaluation runs (default: 5)",
    )

    # Simulation parameters
    parser.add_argument(
        "--ticks-per-second",
        type=int,
        default=60,
        help="Simulation speed in ticks per second (default: 60)",
    )

    # Model parameters
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model file for evaluation or simulation",
    )

    # Agent hyperparameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Initial exploration rate (default: 1.0)",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
        help="Exploration rate decay (default: 0.995)",
    )
    parser.add_argument(
        "--min-epsilon",
        type=float,
        default=0.05,
        help="Minimum exploration rate (default: 0.05)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)",
    )
    parser.add_argument(
        "--action-duration",
        type=int,
        default=300,
        help="Minimum traffic light duration in ticks (default: 300)",
    )

    args = parser.parse_args()

    if args.mode == "train":
        # Create agent with custom hyperparameters
        agent = AdvancedTrafficAgent(
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            min_epsilon=args.min_epsilon,
            batch_size=args.batch_size,
            min_action_duration=args.action_duration,
        )

        # Train agent
        train_advanced_agent(
            agent=agent,
            episodes=args.episodes,
            ticks_per_episode=args.ticks,
            save_interval=args.save_interval,
        )

    elif args.mode == "evaluate":
        if not args.model_path:
            print("Error: --model-path must be specified for evaluation mode")
            return

        # Evaluate agent
        evaluate_advanced_agent(
            model_path=args.model_path,
            num_evaluations=args.num_evals,
            ticks_per_evaluation=args.ticks,
        )

    elif args.mode == "simulate":
        # Create agent
        agent = AdvancedTrafficAgent()

        # If model path is specified, load it
        if args.model_path:
            agent.load_model(args.model_path)
            # Set to evaluation mode (no exploration)
            agent.epsilon = 0.0

        # Run simulation
        reward = simulate(
            agent,
            TRAINING=False,
            TICKS_PER_SECOND=args.ticks_per_second,
            NO_OF_TICKS=args.ticks,
        )

        print(f"Simulation completed with reward: {reward}")


if __name__ == "__main__":
    main()
