import random
import pygame
import sys
from simulation_config import *

pygame.init()
simulation = pygame.sprite.Group()


TRAINING = True  # Set to True for training mode
TICKS_PER_SECOND = 100000  # Ticks per second in the simulation (simulation speed)
VEHICLE_SPAWN_INTERVAL = 30  # Spawn vehicle every 60 ticks (1 second)


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
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
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

    def move(self):
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


# Update values of the signal timers after every tick
def updateValues():
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


class Main:
    # Initialize traffic signals
    initialize()

    # Colours
    black = (0, 0, 0)
    white = (255, 255, 255)

    # Screensize
    screenWidth = 1400
    screenHeight = 800
    screenSize = (screenWidth, screenHeight)

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

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        """
        SIMULATION LOGIC
        """
        # Update the signals based on tick count
        updateValues()

        # Generate vehicles based on tick count
        tick_count += 1
        if tick_count % VEHICLE_SPAWN_INTERVAL == 0:
            generateVehicle()

        # Move all vehicles
        for vehicle in simulation:
            vehicle.move()
        """
            Simulation logic ends here
        """
        # show fps

        print("FPS: ", clock.get_fps())

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

            # display the vehicles
            for vehicle in simulation:
                screen.blit(vehicle.image, [vehicle.x, vehicle.y])
            pygame.display.update()
        clock.tick(TICKS_PER_SECOND)


if __name__ == "__main__":
    Main()
