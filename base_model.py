class Model:
    def __init__(self):
        """
        Basis-Model starten.
        """
        pass

    def get_action(self, state, history):
        """
        Check die Action basierend auf State und History.

        Args:
            state (dict): Aktueller Zustand der Traffic-Sim
            history (list): Frühere Actions vom Model

        Returns:
            bool: True für Nord-Süd grün, False für Ost-West grün
        """
        # Muss von Unterklassen überschrieben werden!
        raise NotImplementedError("Subklassen müssen get_action implementieren")


# Beispiel-Model das alle 400 Ticks flippt
class FlippingModel(Model):
    """
    Simple Ampel die alle 400 Ticks umschaltet.
    """

    def __init__(self):
        super().__init__()
        self.tick_counter = 400
        self.current_state = True  # Start mit Nord-Süd grün

    def get_action(self, state, history):
        """
        Gibt True (Nord-Süd grün) oder False (Ost-West grün) zurück
        und flippt alle 400 Ticks.
        """
        # Tick-Zähler erhöhen
        self.tick_counter += 1

        # Zeit zum Umschalten?
        if self.tick_counter >= 400:
            self.current_state = not self.current_state
            self.tick_counter = 0

        return self.current_state
