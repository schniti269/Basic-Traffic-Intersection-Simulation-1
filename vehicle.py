import pygame
from simulation_config import *
import random


#
#  Funktion zum Zurücksetzen der Startpositionen
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
            f"{self.unnecessary_co2_emission:.2f}", True, white, black
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
