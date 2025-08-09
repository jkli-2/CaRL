class RouteCompletion:
    """
    Detects when a route is nearing completion based on remaining waypoints.
    """

    def __init__(self, min_route_length=10, point_distance=0.1):
        """
        min_route_length: float, minimum distance of remaining route in meters
        point_distance: float, distance between waypoints in meters
        """

        self._min_remaining_waypoints = min_route_length / point_distance
        self._min_route_length = min_route_length
        self._point_distance = point_distance

    def reset(self):
        pass

    def step(self, remaining_route):
        """
        remaining_route: np.array of shape (n, 3) where n is the number of waypoints
        """

        num_remaining_waypoints = remaining_route.shape[0]
        route_complete = num_remaining_waypoints < self._min_remaining_waypoints

        if route_complete:
            print("Route completion detected.", flush=True)

        return route_complete

    def destroy(self):
        pass
