# `roadsim`

Run a simulation of a public transport circuit.

The simulation consists of a single circuit (which, in this case, means a list
of stops).  The number of stops along the circuit as well as their positions
can be specified.  The simulation performs some basic statistical measurements.

Roadsim is simple, but it can be used as a starting point to handle more
complicated scenarios.


## Dependencies

To run the simulation, the following packages are required:

  - scipy
  - numpy
  - matplotlib


## Example

Start by determining the number of bus stops and their positions.

    stop_pos = np.arange(0, 30, 2)
    nb_stops = len(stop_pos)

Determine how far passengers want to go.  Here, we suppose the number of
stops a passenger wants to travel is normally distributed with mean 7.5 and
standard deviation 3.75.  Obviously, a passenger can't travel a negative number
of stops nor travel past the last stop.  This is taken care of by using a
truncated normal function.

    import numpy as np
    from scipy.stats import truncnorm

    mean_stops = nb_stops / 2.0
    std_stops = nb_stops / 4.0
    a, b = (1 - mean_stops) / std_stops, (nb_stops - mean_stops) / std_stops
    stops_to_dest = lambda: np.round(truncnorm.rvs(a, b, loc=mean_stops,
                                                   scale=std_stops))

The simulation is initialized using mostly default values (see below).  The
time between each bus is set to 40 minutes and passenger arrivals at the stops
follow a Poisson process with mean interarrival time set to 5 minutes.
Finally, the number of buses in the simulation is set to 40 (which means the
simulations will run for roughly 1600 minutes, or 27 hours).

    sim = Simulation(bus_stop_positions=stop_pos,
                     time_between_buses=lambda: 40,
                     nb_stops_to_dest=stops_to_dest,
                     passenger_arrival_times=lambda: np.random.exponential(5),
                     nb_buses=40)

Then the simulation can be started.

    sim.run()

After completion, the `sim` object contains information about various
statistics.  It is possible to produce a figure containing lots of interesting
metrics as well as to compute some other statistics.

    import matplotlib.pyplot as plt

    sim.stats.plot()
    plt.show()

    print('Mean satisfaction: {:.3f}'.format(np.mean(sim.stats.satisfaction)))
    print('Mean number of passengers per bus: '
          '{:.2f}'.format(np.mean(sim.stats.total_passengers[-1])))


## Initialization

The default values for the simulation are defined to give something
reasonable, but they should be tuned based on available data.  By default,
a simulation assumes time is measured in minutes, distance in kilometers
and speed in kilometers per minute.  The full list of initialization
options follows.

bus_stop_positions: list or array
:   Position of all bus stops in the simulation.  The number of bus stops
    is set equal to the length of this list.

    Default: 10 stops separated by 3 kilometers; ``np.arange(0, 30, 3)``

passenger_arrival_times: function
:   Time between two successive arrivals of passengers at a bus stop.  This
    function takes no arguments.  For arrivals according to a Poisson process,
    this should be set to an exponential random variable.

    Default: Poisson process with mean interarrival time of 10 minutes; ``lambda: np.random.exponential(10.0)``

hop_in_time: function
:   Time for a passenger to hop into a bus.  This function takes no
    arguments.

    Default: ``lambda: truncnorm.rvs(-1, 8, loc=0.3, scale=0.2)``

hop_out_time: function
:   Time for a passenger to hop out of a bus.  This function takes no
    arguments.

    Default: ``lambda: truncnorm.rvs(-1, 8, loc=0.3, scale=0.2)``

nb_stops_to_dest: function
:   Number of stops between origin and destination for a passenger.  This
    function takes no arguments.

    Default: ``lambda: np.round(truncnorm.rvs(-1, 4, loc=4, scale=3))``

bus_speed: function
:   Bus speed between two successive stops.  This function takes no
    argument.

    Default: ``lambda: truncnorm.rvs(-2, 2, loc=0.83, scale=0.1)``

nb_buses: int
:   Number of buses in the simulation.

    Default: 50

time_between_buses: function
:   Time between the arrival of two successive buses at the first bus stop.
    This function takes no argument.

    Default: ``lambda: 25``

stats_time: float
:   Time between measurement of statistics.

    Default: 5.0

## Statistics

Once the simulation has finished running, the statistics are available in
the ``Stats`` object ``sim.stats``.

## Animation

The `anim_simul.py` script will produce a mp4 movie showing the length of
queues, the positions of the buses and the number of passengers in each bus.
To use the script, create a directory named `anim` and execute the script with

    python anim_simul.py

The animation is saved to a file named `movie.mp4` in the current directory.
