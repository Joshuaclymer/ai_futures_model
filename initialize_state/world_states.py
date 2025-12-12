from classes.simulations import WorldState, Time

def initialize_world_states() -> dict[Time, WorldState]:
    """
    Initializes and returns a dictionary of predefined world states.

    Returns:
        dict[str, WorldState]: A dictionary mapping state names to WorldState instances.
    """
    world_states = {
        Time(year=2026): WorldState(
            name="example",
        )
    }
    return world_states