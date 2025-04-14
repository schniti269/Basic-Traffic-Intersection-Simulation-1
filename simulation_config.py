# Default values of signal timers
defaultGreen = {0: 10, 1: 10, 2: 10, 3: 10}
defaultRed = 150


signals = []
noOfSignals = 4
northGreen = 0  # Indicates which signal is currently green
southGreen = 0  # Indicates which signal is currently green
westGreen = 0  # Indicates which signal is currently green
eastGreen = 0  # Indicates which signal is currently green

speeds = {
    "car": 3,
    "bus": 3,
    "truck": 3,
    "bike": 3,
}  # average speeds of vehicles

# CO2 emission values for vehicles (in arbitrary units)
co2_emissions = {
    "car": {"stopping": 100, "acceleration": 150, "waiting": 2},
    "bus": {"stopping": 250, "acceleration": 300, "waiting": 4},
    "truck": {"stopping": 300, "acceleration": 350, "waiting": 5},
    "bike": {"stopping": 30, "acceleration": 50, "waiting": 1},
}

# Coordinates of vehicles' start
x = {
    "right": [0, 0, 0],
    "down": [755, 727, 697],
    "left": [1400, 1400, 1400],
    "up": [602, 627, 657],
}
y = {
    "right": [348, 370, 398],
    "down": [0, 0, 0],
    "left": [498, 466, 436],
    "up": [800, 800, 800],
}

vehicles = {
    "right": {0: [], 1: [], 2: [], "crossed": 0},
    "down": {0: [], 1: [], 2: [], "crossed": 0},
    "left": {0: [], 1: [], 2: [], "crossed": 0},
    "up": {0: [], 1: [], 2: [], "crossed": 0},
}
vehicleTypes = {0: "car", 1: "bus", 2: "truck", 3: "bike"}
directionNumbers = {0: "right", 1: "down", 2: "left", 3: "up"}

# Coordinates of signal image, timer, and vehicle count
signalCoods = [(530, 230), (810, 230), (810, 570), (530, 570)]
signalTimerCoods = [(530, 210), (810, 210), (810, 550), (530, 550)]

# Coordinates of stop lines
stopLines = {"right": 590, "down": 330, "left": 800, "up": 535}
defaultStop = {"right": 580, "down": 320, "left": 810, "up": 545}
# stops = {'right': [580,580,580], 'down': [320,320,320], 'left': [810,810,810], 'up': [545,545,545]}

# Gap between vehicles
stoppingGap = 15  # stopping gap
movingGap = 15  # moving gap


SCREEN_BOUNDS = {"left": -200, "right": 1600, "top": -200, "bottom": 1000}

# Ursprüngliche Startpositionen speichern
original_x = {
    "right": [0, 0, 0],
    "down": [755, 727, 697],
    "left": [1400, 1400, 1400],
    "up": [602, 627, 657],
}
original_y = {
    "right": [348, 370, 398],
    "down": [0, 0, 0],
    "left": [498, 466, 436],
    "up": [800, 800, 800],
}


# Colours
black = (0, 0, 0)
white = (255, 255, 255)

# Screensize
screenWidth = 1400
screenHeight = 800
screenSize = (screenWidth, screenHeight)
