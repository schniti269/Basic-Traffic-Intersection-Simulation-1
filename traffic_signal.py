class TrafficSignal:
    def __init__(self, red, green):
        self.red = red
        self.green = green
        self.signalText = ""


# Ampel-Timer nach jedem Tick updaten
def update_traffic_lights_Values(
    north_south_green,
    last_north_south_green,
    swtich_time,
    is_swtiching,
    all_red_state_n_ticks,
):
    westGreen = 0
    eastGreen = 0
    southGreen = 0
    northGreen = 0
    if north_south_green != last_north_south_green:
        # Switch initialisieren
        swtich_time = 0
        is_swtiching = True
        # Beim Wechsel alle Ampeln auf Rot
        westGreen = 0
        eastGreen = 0
        southGreen = 0
        northGreen = 0

    if is_swtiching:
        # Switch-Zeit hochzählen
        swtich_time += 1
        if swtich_time >= all_red_state_n_ticks:
            if north_south_green:
                # Nord-Süd auf Grün
                northGreen = 1
                southGreen = 1
                westGreen = 0
                eastGreen = 0

            if not north_south_green:
                # Ost-West auf Grün
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
