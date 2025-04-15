import random
import pygame
import sys
import time
from simulation_config import *
from base_model import FlippingModel
from vehicle import Vehicle, generateVehicle, cleanup_vehicles
from traffic_signal import TrafficSignal, update_traffic_lights_Values


SHOW_FPS = False  # Checke FPS an/aus - True zum Anzeigen
TICKS_PER_SECOND = 1000000000  # Simulation-Speed in Ticks
VEHICLE_SPAWN_INTERVAL = 30  # Alle 30 Ticks neuer Wagen
ALL_RED_STATE_N_TICKS = 60  # Wie lang alles rot bleibt
# Belohnungsparams
REWARD_CO2_MULTIPLIER = 1  # CO2-Faktor für Score
REWARD_CROSSED_MULTIPLIER = 90  # Faktor für durchgefahrene Autos
# Bildschirmgrenzen mit zusätzlichem Puffer definieren


def calculate_reward(co2, crossed):
    # Belohnungsberechnung aus CO2 und durchgekommenen Fahrzeugen
    reward = (crossed * REWARD_CROSSED_MULTIPLIER) - (co2 * REWARD_CO2_MULTIPLIER)
    return reward


def get_state(vehicles):
    # Start-Dict für alle Scan-Zonen
    result = {
        "right": {"car": 0, "bus": 0, "truck": 0, "bike": 0},
        "left": {"car": 0, "bus": 0, "truck": 0, "bike": 0},
        "down": {"car": 0, "bus": 0, "truck": 0, "bike": 0},
        "up": {"car": 0, "bus": 0, "truck": 0, "bike": 0},
    }

    # Jede Zone checken
    for direction, config in DEFAULT_SCAN_ZONE_CONFIG.items():
        zone = config["zone"]
        zone_rect = pygame.Rect(
            zone["x1"], zone["y1"], zone["x2"] - zone["x1"], zone["y2"] - zone["y1"]
        )

        # Alle Fahrzeuge durchgehen
        for vehicle in vehicles:
            # Fahrzeug-Rechteck definieren
            vehicle_rect = pygame.Rect(
                vehicle.x,
                vehicle.y,
                vehicle.image.get_rect().width,
                vehicle.image.get_rect().height,
            )

            # Kollision checken
            if vehicle_rect.colliderect(zone_rect):
                # Fahrzeugtyp zählen
                result[direction][vehicle.vehicleClass] += 1

    return result


def simulate(Model, TRAINING=False, TICKS_PER_SECOND=60, NO_OF_TICKS=60 * 60 * 10):
    pygame.init()
    simulation = pygame.sprite.Group()
    # Ampeln initialisieren
    ts1 = TrafficSignal(0, defaultGreen[0])
    signals.append(ts1)
    ts2 = TrafficSignal(ts1.red + ts1.green, defaultGreen[1])
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultGreen[2])
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultGreen[3])
    signals.append(ts4)
    # Hintergrundbild laden
    background = pygame.image.load("images/intersection.png")

    screen = pygame.display.set_mode(screenSize)
    pygame.display.set_caption("SIMULATION")

    # Ampelbilder und Schrift laden
    redSignal = pygame.image.load("images/signals/red.png")
    greenSignal = pygame.image.load("images/signals/green.png")
    font = pygame.font.Font(None, 30)

    last_signal_state = True
    is_swtiching = False
    switch_time = 0

    clock = pygame.time.Clock()
    tick_count = 0

    output_hist = []  # Historie der Model-Outputs

    # Belohnungen
    crossed_vehicles = 0
    co2_emission = 0

    # Performance-Tracking
    start_time = time.time()
    last_update_time = start_time
    frames_since_last_update = 0
    iters_per_second = 0

    # Progress-Tracking im Training
    progress_update_interval = 1000  # Alle 1000 Ticks updaten

    # Unbegrenzter Speed im Training-Mode
    actual_ticks_per_second = float("inf") if TRAINING else TICKS_PER_SECOND

    while tick_count < NO_OF_TICKS or not TRAINING:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        # FPS anzeigen
        if SHOW_FPS:
            print("FPS: ", clock.get_fps())
        """
        KI-Ampel-Logik
        """
        # Aktuellen Simulations-State holen
        state = get_state(simulation)
        # Controller-Output vom Model bekommen
        controller_output = Model.get_action(state, output_hist)
        output_hist.append(controller_output)
        (
            northGreen,
            eastGreen,
            southGreen,
            westGreen,
            last_signal_state,
            switch_time,
            is_swtiching,
        ) = update_traffic_lights_Values(
            controller_output,
            last_signal_state,
            switch_time,
            is_swtiching,
            ALL_RED_STATE_N_TICKS,
        )

        """
        SIMULATIONS-LOGIK
        """
        # Fahrzeuge generieren nach Tick-Count
        tick_count += 1

        # Performance-Metriken updaten
        frames_since_last_update += 1
        current_time = time.time()
        time_elapsed = current_time - last_update_time

        # Progress im Training anzeigen
        if TRAINING and (
            tick_count % progress_update_interval == 0 or tick_count == NO_OF_TICKS
        ):
            progress_percent = (tick_count / NO_OF_TICKS) * 100
            iters_per_second = (
                frames_since_last_update / time_elapsed if time_elapsed > 0 else 0
            )

            # Fortschrittsbalken bauen
            bar_length = 30
            filled_length = int(bar_length * tick_count // NO_OF_TICKS)
            bar = "#" * filled_length + "-" * (bar_length - filled_length)

            # Fortschritt printen
            print(
                f"\rProgress: |{bar}| {progress_percent:.1f}% - Iterations/sec: {iters_per_second:.1f}",
                end="",
            )

            # Counter für nächstes Update resetten
            last_update_time = current_time
            frames_since_last_update = 0

        if tick_count % VEHICLE_SPAWN_INTERVAL == 0:
            generateVehicle(simulation)

        # Alle Fahrzeuge bewegen
        for vehicle in simulation:
            vehicle.move(northGreen, eastGreen, southGreen, westGreen)

        # Fahrzeuge außerhalb des Bildschirms entfernen
        crossed_vehicles, co2_emission = cleanup_vehicles(
            crossed_vehicles, co2_emission, simulation
        )
        """
        Simulations-Logik endet hier
        """
        """ BELOHNUNGS-BERECHNUNG 
        """
        reward = calculate_reward(co2_emission, crossed_vehicles)

        # Rendering
        if not TRAINING:
            screen.blit(background, (0, 0))  # Hintergrund anzeigen
            # Ampeln rendern
            if northGreen == 1:
                screen.blit(greenSignal, signalCoods[1])
            else:
                screen.blit(redSignal, signalCoods[1])
            if southGreen == 1:
                screen.blit(greenSignal, signalCoods[3])
            else:
                screen.blit(redSignal, signalCoods[3])
            if westGreen == 1:
                screen.blit(greenSignal, signalCoods[0])
            else:
                screen.blit(redSignal, signalCoods[0])
            if eastGreen == 1:
                screen.blit(greenSignal, signalCoods[2])
            else:
                screen.blit(redSignal, signalCoods[2])

            # CO2 und Fahrzeug-Counter anzeigen
            co2_emission_text = font.render(
                f"Co2: {co2_emission:.2f}g", True, white, black
            )
            screen.blit(co2_emission_text, (10, 10))
            crossed_vehicles_text = font.render(
                f"Crossed Vehicles: {crossed_vehicles}", True, white, black
            )
            screen.blit(crossed_vehicles_text, (10, 40))
            # Belohnung anzeigen
            reward_text = font.render(f"Reward: {reward:.2f}", True, white, black)
            screen.blit(reward_text, (10, 70))

            # Fahrzeuge anzeigen
            for vehicle in simulation:
                vehicle.render(screen)

                # Scan-Zonen zeichnen
            for direction, config in DEFAULT_SCAN_ZONE_CONFIG.items():
                # Zone als transparentes Rechteck
                zone = config["zone"]
                zone_rect = pygame.Rect(
                    zone["x1"],
                    zone["y1"],
                    zone["x2"] - zone["x1"],
                    zone["y2"] - zone["y1"],
                )
                zone_surface = pygame.Surface(
                    (zone_rect.width, zone_rect.height), pygame.SRCALPHA
                )
                zone_surface.fill((0, 255, 0, 64))  # Rot mit 25% Transparenz
                screen.blit(zone_surface, (zone_rect.x, zone_rect.y))

                # Rechteck-Umriss zeichnen
                pygame.draw.rect(screen, (0, 255, 0), zone_rect, 2)

                # Kamera als Kreis anzeigen
                camera = config["camera"]
                pygame.draw.circle(
                    screen, (255, 255, 0), (camera["x"], camera["y"]), 10
                )

                # Zonen-Label hinzufügen
                direction_label = font.render(
                    direction, True, (255, 255, 255), (0, 0, 0)
                )
                screen.blit(direction_label, (camera["x"] - 15, camera["y"] - 15))
            pygame.display.update()

        # Unbegrenzter FPS im Training, sonst mit TICKS_PER_SECOND
        clock.tick(actual_ticks_per_second)

    # Newline nach Training-Progress
    if TRAINING:
        print()  # Neue Zeile nach Fortschrittsbalken

        # Finale Stats berechnen
        total_time = time.time() - start_time
        avg_iters_per_second = tick_count / total_time if total_time > 0 else 0
        print(f"Training completed in {total_time:.2f} seconds")
        print(f"Average iterations per second: {avg_iters_per_second:.1f}")
        print(f"Final reward: {reward:.2f}")

    # Alle Fahrzeuge aus der Map teleportieren und aufräumen
    for vehicle in simulation:
        vehicle.x = -1000
        vehicle.y = -1000

    crossed_vehicles, co2_emission = cleanup_vehicles(
        crossed_vehicles, co2_emission, simulation
    )

    reward = calculate_reward(co2_emission, crossed_vehicles)

    return reward


if __name__ == "__main__":
    # Simulation starten
    from td_learning import TDLearningModel

    # Model erstellen und laden
    my_model = TDLearningModel()
    my_model.load_model(
        r"C:\Users\ian-s\Traffic_refactor\Basic-Traffic-Intersection-Simulation\models_old\td_model_final.pkl"
    )

    # Explorations-Rate auf 0 für Evaluation
    my_model.exploration_rate = 0.0

    simulate(my_model, TRAINING=False, TICKS_PER_SECOND=600)
