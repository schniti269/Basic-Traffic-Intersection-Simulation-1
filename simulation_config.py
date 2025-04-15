# Standard-Timer für Ampeln
defaultGreen = {0: 10, 1: 10, 2: 10, 3: 10}
defaultRed = 150


signals = []
noOfSignals = 4
northGreen = 0  # Welche Ampeln grad grün sind
southGreen = 0  # Welche Ampeln grad grün sind
westGreen = 0  # Welche Ampeln grad grün sind
eastGreen = 0  # Welche Ampeln grad grün sind

speeds = {
    "car": 3,
    "bus": 3,
    "truck": 3,
    "bike": 3,
}  # Durchschnitts-Speed der Fahrzeuge

# CO2-Werte für Fahrzeuge (ausgedachte Einheiten)
co2_emissions = {
    "car": {"stopping": 0.3, "acceleration": 2.0, "waiting": 0.4},
    "bus": {"stopping": 0.6, "acceleration": 4.0, "waiting": 0.9},
    "truck": {"stopping": 0.5, "acceleration": 3.5, "waiting": 0.7},
    "bike": {"stopping": 0.1, "acceleration": 0.5, "waiting": 0.15},
}


# Startkoordinaten der Fahrzeuge
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

# Koordinaten für Ampelbild, Timer und Fahrzeugzähler
signalCoods = [(530, 230), (810, 230), (810, 570), (530, 570)]
signalTimerCoods = [(530, 210), (810, 210), (810, 550), (530, 550)]

# Koordinaten der Stopplinien
stopLines = {"right": 590, "down": 330, "left": 800, "up": 535}
defaultStop = {"right": 580, "down": 320, "left": 810, "up": 545}
# stops = {'right': [580,580,580], 'down': [320,320,320], 'left': [810,810,810], 'up': [545,545,545]}

# Abstand zwischen Fahrzeugen
stoppingGap = 15  # Abstand beim Stoppen
movingGap = 15  # Abstand bei Bewegung


SCREEN_BOUNDS = {"left": -200, "right": 1600, "top": -200, "bottom": 1000}

# Original-Startpositionen für Reset
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


# Farben
black = (0, 0, 0)
white = (255, 255, 255)

# Bildschirmgröße
screenWidth = 1400
screenHeight = 800
screenSize = (screenWidth, screenHeight)

DEFAULT_SCAN_ZONE_CONFIG = {
    "right": {
        "zone": {
            "x1": 811,  # Linke Kante
            "y1": 427,  # Obere Kante
            "x2": 1400,  # Rechte Kante
            "y2": 512,  # Untere Kante
        },
        "camera": {
            "x": 787,  # Kamera X-Pos
            "y": 464,  # Kamera Y-Pos
        },
    },
    "left": {
        "zone": {
            "x1": 0,  # Linke Kante
            "y1": 370,  # Obere Kante
            "x2": 580,  # Rechte Kante
            "y2": 424,  # Untere Kante
        },
        "camera": {
            "x": 600,  # Kamera X-Pos
            "y": 400,  # Kamera Y-Pos
        },
    },
    "down": {
        "zone": {
            "x1": 600,  # Linke Kante
            "y1": 546,  # Obere Kante
            "x2": 681,  # Rechte Kante
            "y2": 800,  # Untere Kante
        },
        "camera": {
            "x": 650,  # Kamera X-Pos
            "y": 530,  # Kamera Y-Pos
        },
    },
    "up": {
        "zone": {
            "x1": 688,  # Linke Kante
            "y1": 0,  # Obere Kante
            "x2": 767,  # Rechte Kante
            "y2": 321,  # Untere Kante
        },
        "camera": {
            "x": 730,  # Kamera X-Pos
            "y": 320,  # Kamera Y-Pos
        },
    },
}
