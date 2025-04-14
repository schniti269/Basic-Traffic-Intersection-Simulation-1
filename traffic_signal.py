class TrafficSignal:
    def __init__(self, red, green):
        self.red = red
        self.green = green
        self.signalText = ""


# Update values of the signal timers after every tick
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
        if swtich_time >= all_red_state_n_ticks:
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
