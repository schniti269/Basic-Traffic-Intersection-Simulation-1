import random
import pygame
import sys
from simulation_config import *
from base_model import FlippingModel
from vehicle import Vehicle, generateVehicle, cleanup_vehicles
from traffic_signal import TrafficSignal, update_traffic_lights_Values


SHOW_FPS = False  # Set to True to show frames per second
TICKS_PER_SECOND = 1000000000  # Ticks per second in the simulation (simulation speed)
VEHICLE_SPAWN_INTERVAL = 30  # Spawn vehicle every 60 ticks (1 second)
ALL_RED_STATE_N_TICKS = 60  # Duration of all red state in ticks
# rewardparams
REWARD_CO2_MULTIPLIER = 100  # Multiplier for CO2 emission in reward calculation
REWARD_CROSSED_MULTIPLIER = (
    1300  # Multiplier for crossed vehicles in reward calculation
)
# Bildschirmgrenzen mit zusätzlichem Puffer definieren


def calculate_reward(co2, crossed):
    # Calculate the reward based on CO2 emissions and crossed vehicles
    reward = (crossed * REWARD_CROSSED_MULTIPLIER) - (co2 * REWARD_CO2_MULTIPLIER)
    return reward


def get_state(vehicles):
    # Ergebnis-Dictionary für alle Scan-Zonen initialisieren
    result = {
        "right": {"car": 0, "bus": 0, "truck": 0, "bike": 0},
        "left": {"car": 0, "bus": 0, "truck": 0, "bike": 0},
        "down": {"car": 0, "bus": 0, "truck": 0, "bike": 0},
        "up": {"car": 0, "bus": 0, "truck": 0, "bike": 0},
    }

    # Für jede Scan-Zone prüfen, ob das Fahrzeug darin liegt
    for direction, config in DEFAULT_SCAN_ZONE_CONFIG.items():
        zone = config["zone"]
        zone_rect = pygame.Rect(
            zone["x1"], zone["y1"], zone["x2"] - zone["x1"], zone["y2"] - zone["y1"]
        )

        # Für jedes Fahrzeug in der Simulation
        for vehicle in vehicles:
            # Fahrzeugdimensionen bestimmen (Rechteck)
            vehicle_rect = pygame.Rect(
                vehicle.x,
                vehicle.y,
                vehicle.image.get_rect().width,
                vehicle.image.get_rect().height,
            )

            # Überprüfen, ob das Fahrzeug mit der Zone überlappt
            if vehicle_rect.colliderect(zone_rect):
                # Fahrzeugklasse zählen
                result[direction][vehicle.vehicleClass] += 1

    return result


def simulate(Model, TRAINING=False, TICKS_PER_SECOND=60, NO_OF_TICKS=60 * 60 * 10):
    pygame.init()
    simulation = pygame.sprite.Group()
    # Initialize traffic signals
    ts1 = TrafficSignal(0, defaultGreen[0])
    signals.append(ts1)
    ts2 = TrafficSignal(ts1.red + ts1.green, defaultGreen[1])
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultGreen[2])
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultGreen[3])
    signals.append(ts4)
    # Setting background image i.e. image of intersection
    background = pygame.image.load("images/intersection.png")

    screen = pygame.display.set_mode(screenSize)
    pygame.display.set_caption("SIMULATION")

    # Loading signal images and font
    redSignal = pygame.image.load("images/signals/red.png")
    greenSignal = pygame.image.load("images/signals/green.png")
    font = pygame.font.Font(None, 30)

    last_signal_state = True
    is_swtiching = False
    switch_time = 0

    clock = pygame.time.Clock()
    tick_count = 0

    output_hist = []  # history of the output of the model

    # rewards
    crossed_vehicles = 0
    co2_emission = 0

    while tick_count < NO_OF_TICKS or not TRAINING:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        # show fps
        if SHOW_FPS:
            print("FPS: ", clock.get_fps())
        """
        AI Traffic light LOGIC
        """
        # get a vectorized state of the current simulation state
        state = get_state(simulation)
        # get the controller output from the model
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
        SIMULATION LOGIC
        """
        # Generate vehicles based on tick count
        tick_count += 1
        if tick_count % VEHICLE_SPAWN_INTERVAL == 0:
            generateVehicle(simulation)

        # Move all vehicles
        for vehicle in simulation:
            vehicle.move(northGreen, eastGreen, southGreen, westGreen)

        # Cleanup vehicles that have left the screen
        crossed_vehicles, co2_emission = cleanup_vehicles(
            crossed_vehicles, co2_emission, simulation
        )
        """
        Simulation logic ends here
        """
        """ REWARD CALCULATION 
        """
        reward = calculate_reward(co2_emission, crossed_vehicles)

        # Rendering
        if not TRAINING:
            screen.blit(background, (0, 0))  # display background in simulation
            # render the signals
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

            # display the co2 emission and crossed vehicles
            co2_emission_text = font.render(
                f"Co2: {co2_emission:.2f}g", True, white, black
            )
            screen.blit(co2_emission_text, (10, 10))
            crossed_vehicles_text = font.render(
                f"Crossed Vehicles: {crossed_vehicles}", True, white, black
            )
            screen.blit(crossed_vehicles_text, (10, 40))
            # display the reward
            reward_text = font.render(f"Reward: {reward:.2f}", True, white, black)
            screen.blit(reward_text, (10, 70))

            # display the vehicles
            for vehicle in simulation:
                vehicle.render(screen)

                # Zeichne die Scan-Zonen
            for direction, config in DEFAULT_SCAN_ZONE_CONFIG.items():
                # Zeichne die Zone als transparentes Rechteck
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

                # Zeichne den Umriss des Rechtecks
                pygame.draw.rect(screen, (0, 255, 0), zone_rect, 2)

                # Zeichne die Kameraposition als Kreis
                camera = config["camera"]
                pygame.draw.circle(
                    screen, (255, 255, 0), (camera["x"], camera["y"]), 10
                )

                # Optional: Beschriftung für jede Zone hinzufügen
                direction_label = font.render(
                    direction, True, (255, 255, 255), (0, 0, 0)
                )
                screen.blit(direction_label, (camera["x"] - 15, camera["y"] - 15))
            pygame.display.update()

        clock.tick(TICKS_PER_SECOND)

    # do a post run cleanup by teleporting all vehicles off map and running cleanup_vehicles
    # so that the vehicles are removed from the simulation
    for vehicle in simulation:
        vehicle.x = -1000
        vehicle.y = -1000

    cleanup_vehicles(crossed_vehicles, co2_emission)

    reward = calculate_reward(co2_emission, crossed_vehicles)

    return reward


if __name__ == "__main__":
    # Run the simulation
    my_model = FlippingModel()
    simulate(my_model, TRAINING=False, TICKS_PER_SECOND=60)
