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
    "car": {"stopping": 0.3, "acceleration": 2.0, "waiting": 0.4},
    "bus": {"stopping": 0.6, "acceleration": 4.0, "waiting": 0.9},
    "truck": {"stopping": 0.5, "acceleration": 3.5, "waiting": 0.7},
    "bike": {"stopping": 0.1, "acceleration": 0.5, "waiting": 0.15},
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

DEFAULT_SCAN_ZONE_CONFIG = {
    "right": {
        "zone": {
            "x1": 811,  # Left edge
            "y1": 427,  # Top edge
            "x2": 1400,  # Right edge
            "y2": 512,  # Bottom edge
        },
        "camera": {
            "x": 787,  # Camera x position
            "y": 464,  # Camera y position
        },
    },
    "left": {
        "zone": {
            "x1": 0,  # Left edge
            "y1": 370,  # Top edge
            "x2": 580,  # Right edge
            "y2": 424,  # Bottom edge
        },
        "camera": {
            "x": 600,  # Camera x position
            "y": 400,  # Camera y position
        },
    },
    "down": {
        "zone": {
            "x1": 600,  # Left edge
            "y1": 546,  # Top edge
            "x2": 681,  # Right edge
            "y2": 800,  # Bottom edge
        },
        "camera": {
            "x": 650,  # Camera x position
            "y": 530,  # Camera y position
        },
    },
    "up": {
        "zone": {
            "x1": 688,  # Left edge
            "y1": 0,  # Top edge
            "x2": 767,  # Right edge
            "y2": 321,  # Bottom edge
        },
        "camera": {
            "x": 730,  # Camera x position
            "y": 320,  # Camera y position
        },
    },
}
