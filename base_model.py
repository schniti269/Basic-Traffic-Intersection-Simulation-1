class Model:
    def __init__(self):
        """
        Initialize the base model.
        """
        pass

    def get_action(self, state, history):
        """
        Given a state and history of previous actions, return the action to take.

        Args:
            state (dict): A dictionary containing the current state of the traffic simulation
            history (list): A list of previous actions taken by the model

        Returns:
            bool: True for north-south green, False for east-west green
        """
        # Base implementation should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement get_action method")


# example of a model that flips its state every 400 ticks
class FlippingModel(Model):
    """
    A simple model that flips its state every 400 ticks.
    """

    def __init__(self):
        super().__init__()
        self.tick_counter = 400
        self.current_state = True  # Start with north-south green

    def get_action(self, state, history):
        """
        Returns True (north-south green) or False (east-west green)
        and flips the state every 400 ticks.
        """
        # Increment the tick counter
        self.tick_counter += 1

        # Check if it's time to flip the state
        if self.tick_counter >= 400:
            self.current_state = not self.current_state
            self.tick_counter = 0

        return self.current_state
