import random
import pygame
import sys
from simulation_config import *


SHOW_FPS = False  # Set to True to show frames per second
TICKS_PER_SECOND = 1000000000  # Ticks per second in the simulation (simulation speed)
VEHICLE_SPAWN_INTERVAL = 30  # Spawn vehicle every 60 ticks (1 second)
ALL_RED_STATE_N_TICKS = 60  # Duration of all red state in ticks
# rewardparams
REWARD_CO2_MULTIPLIER = 1  # Multiplier for CO2 emission in reward calculation
REWARD_CROSSED_MULTIPLIER = 130  # Multiplier for crossed vehicles in reward calculation
# Bildschirmgrenzen mit zusätzlichem Puffer definieren


# Funktion zum Zurücksetzen der Startpositionen
def reset_spawn_positions():
    global x, y
    # Kopieren Sie die ursprünglichen Werte zurück
    for direction in x:
        for lane in range(len(x[direction])):
            x[direction][lane] = original_x[direction][lane]
            y[direction][lane] = original_y[direction][lane]


# Entfernt Fahrzeuge, die außerhalb des Bildschirms sind
def cleanup_vehicles(this_crossed_vehicles, this_unnecessary_co2_emission, simulation):
    vehicles_to_remove = []
    # Alle Fahrzeuge überprüfen
    for vehicle in simulation:
        # Prüfen, ob das Fahrzeug außerhalb des Bildschirms ist
        if (
            vehicle.x < SCREEN_BOUNDS["left"]
            or vehicle.x > SCREEN_BOUNDS["right"]
            or vehicle.y < SCREEN_BOUNDS["top"]
            or vehicle.y > SCREEN_BOUNDS["bottom"]
        ):

            # Fahrzeug für Entfernung markieren
            vehicles_to_remove.append(vehicle)
            # add unnecessary co2 emission to the total co2 emission
            this_unnecessary_co2_emission += vehicle.unnecessary_co2_emission
            # add crossed vehicles to the total crossed vehicles
            this_crossed_vehicles += 1

    # Fahrzeuge aus beiden Listen entfernen
    for vehicle in vehicles_to_remove:
        # Aus der simulation-Gruppe entfernen
        simulation.remove(vehicle)

        # Aus der vehicles-Datenstruktur entfernen
        if vehicle.index < len(vehicles[vehicle.direction][vehicle.lane]):
            vehicles[vehicle.direction][vehicle.lane].pop(vehicle.index)

            # Indizes der nachfolgenden Fahrzeuge in der gleichen Spur aktualisieren
            for i in range(
                vehicle.index, len(vehicles[vehicle.direction][vehicle.lane])
            ):
                vehicles[vehicle.direction][vehicle.lane][i].index = i

    # Wenn eine bestimmte Anzahl von Fahrzeugen entfernt wurde, setze die Spawn-Positionen zurück
    if len(vehicles_to_remove) > 0:
        reset_spawn_positions()

    return this_crossed_vehicles, this_unnecessary_co2_emission


class TrafficSignal:
    def __init__(self, red, green):
        self.red = red
        self.green = green
        self.signalText = ""


class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction, simulation):
        pygame.sprite.Sprite.__init__(self)
        self.unnecessary_co2_emission = 0
        self.this_tick_unnecessary_co2_emission = 0
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        self.is_moving = True
        self.was_moving_previous_tick = True
        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1
        path = "images/" + direction + "/" + vehicleClass + ".png"
        self.image = pygame.image.load(path)

        if (
            len(vehicles[direction][lane]) > 1
            and vehicles[direction][lane][self.index - 1].crossed == 0
        ):  # if more than 1 vehicle in the lane of vehicle before it has crossed stop line
            if direction == "right":
                self.stop = (
                    vehicles[direction][lane][self.index - 1].stop
                    - vehicles[direction][lane][self.index - 1].image.get_rect().width
                    - stoppingGap
                )  # setting stop coordinate as: stop coordinate of next vehicle - width of next vehicle - gap
            elif direction == "left":
                self.stop = (
                    vehicles[direction][lane][self.index - 1].stop
                    + vehicles[direction][lane][self.index - 1].image.get_rect().width
                    + stoppingGap
                )
            elif direction == "down":
                self.stop = (
                    vehicles[direction][lane][self.index - 1].stop
                    - vehicles[direction][lane][self.index - 1].image.get_rect().height
                    - stoppingGap
                )
            elif direction == "up":
                self.stop = (
                    vehicles[direction][lane][self.index - 1].stop
                    + vehicles[direction][lane][self.index - 1].image.get_rect().height
                    + stoppingGap
                )
        else:
            self.stop = defaultStop[direction]

        # Set new starting and stopping coordinate
        if direction == "right":
            temp = self.image.get_rect().width + stoppingGap
            x[direction][lane] -= temp
        elif direction == "left":
            temp = self.image.get_rect().width + stoppingGap
            x[direction][lane] += temp
        elif direction == "down":
            temp = self.image.get_rect().height + stoppingGap
            y[direction][lane] -= temp
        elif direction == "up":
            temp = self.image.get_rect().height + stoppingGap
            y[direction][lane] += temp
        simulation.add(self)

    def render(self, screen):
        screen.blit(self.image, (self.x, self.y))
        # add a counter to the vehicle to show current unnecessary co2 emission onto the car
        font = pygame.font.Font(None, 20)
        vehicle_counter = font.render(
            str(self.unnecessary_co2_emission), True, white, black
        )
        screen.blit(vehicle_counter, (self.x, self.y))

    def move(self, northGreen, eastGreen, southGreen, westGreen):
        # Speichern des vorherigen Bewegungsstatus
        self.was_moving_previous_tick = self.is_moving
        # Standardmäßig auf "nicht bewegt" setzen und später aktualisieren wenn nötig
        self.is_moving = False

        if self.direction == "right":
            if (
                self.crossed == 0
                and self.x + self.image.get_rect().width > stopLines[self.direction]
            ):  # if the image has crossed stop line now
                self.crossed = 1
            if (
                self.x + self.image.get_rect().width <= self.stop
                or self.crossed == 1
                or (westGreen == 1)
            ) and (
                self.index == 0
                or self.x + self.image.get_rect().width
                < (vehicles[self.direction][self.lane][self.index - 1].x - movingGap)
            ):
                # (if the image has not reached its stop coordinate or has crossed stop line or has green signal) and (it is either the first vehicle in that lane or it is has enough gap to the next vehicle in that lane)
                self.x += self.speed  # move the vehicle
                self.is_moving = True
        elif self.direction == "down":
            if (
                self.crossed == 0
                and self.y + self.image.get_rect().height > stopLines[self.direction]
            ):
                self.crossed = 1
            if (
                self.y + self.image.get_rect().height <= self.stop
                or self.crossed == 1
                or (northGreen == 1)
            ) and (
                self.index == 0
                or self.y + self.image.get_rect().height
                < (vehicles[self.direction][self.lane][self.index - 1].y - movingGap)
            ):
                self.y += self.speed
                self.is_moving = True
        elif self.direction == "left":
            if self.crossed == 0 and self.x < stopLines[self.direction]:
                self.crossed = 1
            if (self.x >= self.stop or self.crossed == 1 or (eastGreen == 1)) and (
                self.index == 0
                or self.x
                > (
                    vehicles[self.direction][self.lane][self.index - 1].x
                    + vehicles[self.direction][self.lane][self.index - 1]
                    .image.get_rect()
                    .width
                    + movingGap
                )
            ):
                self.x -= self.speed
                self.is_moving = True
        elif self.direction == "up":
            if self.crossed == 0 and self.y < stopLines[self.direction]:
                self.crossed = 1
            if (self.y >= self.stop or self.crossed == 1 or (southGreen == 1)) and (
                self.index == 0
                or self.y
                > (
                    vehicles[self.direction][self.lane][self.index - 1].y
                    + vehicles[self.direction][self.lane][self.index - 1]
                    .image.get_rect()
                    .height
                    + movingGap
                )
            ):
                self.y -= self.speed
                self.is_moving = True

        # CO2-Emissionen aktualisieren
        self.update_co2_emissions()

    def update_co2_emissions(self):
        # Wenn das Fahrzeug nicht in Bewegung ist, erhöhe die Wartezeitwerte
        if not self.is_moving:
            self.this_tick_unnecessary_co2_emission = co2_emissions[self.vehicleClass][
                "waiting"
            ]
            self.unnecessary_co2_emission += self.this_tick_unnecessary_co2_emission

        # Wenn das Fahrzeug von Bewegung zu Stillstand wechselt, füge Bremsemissionen hinzu
        if self.was_moving_previous_tick and not self.is_moving:
            self.this_tick_unnecessary_co2_emission = co2_emissions[self.vehicleClass][
                "stopping"
            ]
            self.this_tick_unnecessary_co2_emission += co2_emissions[self.vehicleClass][
                "acceleration"
            ]
            self.unnecessary_co2_emission += self.this_tick_unnecessary_co2_emission
            # wenn das fahrzeug steht wird es später wieder beschleunigt also werden die emissionswerte für beschleunigung nicht hinzugefügt


# Update values of the signal timers after every tick
def update_traffic_lights_Values(
    north_south_green, last_north_south_green, swtich_time, is_swtiching
):
    westGreen = 0
    eastGreen = 0
    southGreen = 0
    northGreen = 0
    if north_south_green != last_north_south_green:
        # initialize a signal switch
        swtich_time = 0
        is_swtiching = True
        # if the signal is switching, set the si#gnal to red for ALL the signals
        westGreen = 0
        eastGreen = 0
        southGreen = 0
        northGreen = 0

    if is_swtiching:
        # if the signal is switching, increment the switch time
        swtich_time += 1
        if swtich_time >= ALL_RED_STATE_N_TICKS:
            if north_south_green:
                # swithc the signal to green for north south direction
                northGreen = 1
                southGreen = 1
                westGreen = 0
                eastGreen = 0

            if not north_south_green:
                # swithc the signal to green for east west direction
                northGreen = 0
                southGreen = 0
                westGreen = 1
                eastGreen = 1

    return (
        northGreen,
        eastGreen,
        southGreen,
        westGreen,
        north_south_green,
        swtich_time,
        is_swtiching,
    )


# Generate a new vehicle based on simulation parameters
def generateVehicle(simulation):
    vehicle_type = random.randint(0, 3)
    lane_number = random.randint(1, 2)
    direction_number = random.randint(0, 3)
    Vehicle(
        lane_number,
        vehicleTypes[vehicle_type],
        direction_number,
        directionNumbers[direction_number],
        simulation,
    )


def calculate_reward(co2, crossed):
    # Calculate the reward based on CO2 emissions and crossed vehicles
    reward = (crossed * REWARD_CROSSED_MULTIPLIER) - (co2 * REWARD_CO2_MULTIPLIER)
    return reward


def get_state(vehicles):

    return vehicles


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

    contr = True

    while tick_count < NO_OF_TICKS or not TRAINING:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        # all 400 ticks switch the signal state
        if tick_count % 400 == 0:
            contr = not contr

        #

        # show fps
        if SHOW_FPS:
            print("FPS: ", clock.get_fps())
        """
        AI Traffic light LOGIC
        """
        # get a vectorized state of the current simulation state
        state = get_state(simulation)
        # get the controller output from the model
        controller_output = contr  # Model.get_action(state, output_hist)
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
            controller_output, last_signal_state, switch_time, is_swtiching
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
                "CO2 Emission: " + str(co2_emission), True, white, black
            )
            screen.blit(co2_emission_text, (10, 10))
            crossed_vehicles_text = font.render(
                "Crossed Vehicles: " + str(crossed_vehicles), True, white, black
            )
            screen.blit(crossed_vehicles_text, (10, 40))
            # display the reward
            reward_text = font.render("Reward: " + str(reward), True, white, black)
            screen.blit(reward_text, (10, 70))

            # display the vehicles
            for vehicle in simulation:
                vehicle.render(screen)
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
    simulate(None)
