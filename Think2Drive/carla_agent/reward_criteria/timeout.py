class Timeout:
    # In their email they say in their first paper version they use 50 seconds as timeout but later they use 100
    def __init__(self, speed_threshold=1.0, timeout_duration=100, simulation_timestep=0.1):
        """
        Initialize the timeout manager.
        speed_threshold: float representing the speed threshold in m/s, which defines the vehicle as moving
        timeout_duration: float representing the time in seconds before the episode is terminated
        simulation_timestep: float representing the time in seconds between each step
        """
        self._speed_threshold = speed_threshold
        self._timeout_duration = timeout_duration
        self._simulation_timestep = simulation_timestep

        self._vehicle = None
        self._stationary_ticks = 0

    def reset(self, vehicle):
        """
        Reset for a new episode.
        """
        self._vehicle = vehicle
        self._stationary_ticks = 0

    def step(self):
        """
        Update the timeout manager for one simulation step.
        """

        # Get vehicle velocity magnitude
        velocity_vector = self._vehicle.get_velocity()
        current_speed = velocity_vector.length()

        # Update stationary tick counter
        if current_speed < self._speed_threshold:
            self._stationary_ticks += 1
        else:
            self._stationary_ticks = 0

        # Calculate total stationary time
        stationary_time = self._stationary_ticks * self._simulation_timestep
        timeout_reached = stationary_time > self._timeout_duration

        if timeout_reached:
            print(f"Timeout reached: Vehicle stationary for {stationary_time:.1f}s ", flush=True)

        return timeout_reached, stationary_time

    def destroy(self):
        """
        Clean up resources and reset internal state.
        """
        pass
