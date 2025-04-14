import random
import pygame
import sys
from simulation_config import *

pygame.init()
simulation = pygame.sprite.Group()


SHOW_FPS = False  # Set to True to show frames per second
TICKS_PER_SECOND = 1000000000  # Ticks per second in the simulation (simulation speed)
VEHICLE_SPAWN_INTERVAL = 30  # Spawn vehicle every 60 ticks (1 second)
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
def cleanup_vehicles(this_crossed_vehicles, this_unnecessary_co2_emission):
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
    def __init__(self, red, yellow, green):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.signalText = ""


# Initialization of signals with default values
def initialize():
    ts1 = TrafficSignal(0, defaultYellow, defaultGreen[0])
    signals.append(ts1)
    ts2 = TrafficSignal(
        ts1.red + ts1.yellow + ts1.green, defaultYellow, defaultGreen[1]
    )
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen[2])
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen[3])
    signals.append(ts4)


class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction):
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

    def move(self):
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
                or (currentGreen == 0 and currentYellow == 0)
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
                or (currentGreen == 1 and currentYellow == 0)
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
            if (
                self.x >= self.stop
                or self.crossed == 1
                or (currentGreen == 2 and currentYellow == 0)
            ) and (
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
            if (
                self.y >= self.stop
                or self.crossed == 1
                or (currentGreen == 3 and currentYellow == 0)
            ) and (
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


# Update values of the signal timers after every tick this will later be AI
def update_traffic_lights_Values():
    global currentGreen, currentYellow, nextGreen

    for i in range(0, noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                signals[i].green -= 1
                if signals[i].green == 0:
                    currentYellow = 1
                    # Reset stop coordinates of lanes and vehicles
                    for j in range(0, 3):
                        for vehicle in vehicles[directionNumbers[currentGreen]][j]:
                            vehicle.stop = defaultStop[directionNumbers[currentGreen]]
            else:
                signals[i].yellow -= 1
                if signals[i].yellow == 0:
                    currentYellow = 0
                    # Reset all signal times of current signal to default times
                    signals[i].green = defaultGreen[i]
                    signals[i].yellow = defaultYellow
                    signals[i].red = defaultRed

                    currentGreen = nextGreen  # Set next signal as green signal
                    nextGreen = (
                        currentGreen + 1
                    ) % noOfSignals  # Set next green signal
                    signals[nextGreen].red = (
                        signals[currentGreen].yellow + signals[currentGreen].green
                    )  # Set the red time of next to next signal as (yellow time + green time) of next signal
        else:
            signals[i].red -= 1


# Generate a new vehicle based on simulation parameters
def generateVehicle():
    vehicle_type = random.randint(0, 3)
    lane_number = random.randint(1, 2)
    direction_number = random.randint(0, 3)
    Vehicle(
        lane_number,
        vehicleTypes[vehicle_type],
        direction_number,
        directionNumbers[direction_number],
    )


def calculate_reward(co2, crossed):
    # Calculate the reward based on CO2 emissions and crossed vehicles
    reward = (crossed * REWARD_CROSSED_MULTIPLIER) - (co2 * REWARD_CO2_MULTIPLIER)
    return reward


def simulate(Model, TRAINING=False, TICKS_PER_SECOND=60, NO_OF_TICKS=60 * 60 * 10):
    # Initialize traffic signals
    initialize()
    # Setting background image i.e. image of intersection
    background = pygame.image.load("images/intersection.png")

    screen = pygame.display.set_mode(screenSize)
    pygame.display.set_caption("SIMULATION")

    # Loading signal images and font
    redSignal = pygame.image.load("images/signals/red.png")
    yellowSignal = pygame.image.load("images/signals/yellow.png")
    greenSignal = pygame.image.load("images/signals/green.png")
    font = pygame.font.Font(None, 30)

    clock = pygame.time.Clock()
    tick_count = 0

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
        # vectorized_state = get_current_state()
        # get the predicted action from the AI model
        # predicted_action = get_action(vectorized_state)
        # Update the signals based on AI logic
        update_traffic_lights_Values()

        """
        SIMULATION LOGIC
        """
        # Generate vehicles based on tick count
        tick_count += 1
        if tick_count % VEHICLE_SPAWN_INTERVAL == 0:
            generateVehicle()

        # Move all vehicles
        for vehicle in simulation:
            vehicle.move()

        # Cleanup vehicles that have left the screen
        crossed_vehicles, co2_emission = cleanup_vehicles(
            crossed_vehicles, co2_emission
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
            for i in range(
                0, noOfSignals
            ):  # display signal and set timer according to current status: green, yellow, or red
                if i == currentGreen:
                    if currentYellow == 1:
                        signals[i].signalText = signals[i].yellow
                        screen.blit(yellowSignal, signalCoods[i])
                    else:
                        signals[i].signalText = signals[i].green
                        screen.blit(greenSignal, signalCoods[i])
                else:
                    if signals[i].red <= 10:
                        signals[i].signalText = signals[i].red
                    else:
                        signals[i].signalText = "---"
                    screen.blit(redSignal, signalCoods[i])
            signalTexts = ["", "", "", ""]

            # display signal timer
            for i in range(0, noOfSignals):
                signalTexts[i] = font.render(
                    str(signals[i].signalText), True, white, black
                )
                screen.blit(signalTexts[i], signalTimerCoods[i])
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
