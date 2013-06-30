import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm
import heapq
import mpl_toolkits.axisartist as AA


class Passenger:
    def __init__(self, source=None, dest=None, start_time=0):
        self.source = source
        self.dest = dest
        self.time_waited_for_bus = 0.0
        self.satisfaction = 1.0
        self.start_time = start_time
        self.end_time = None
        
    def satisfy(self):
        """Sigmoid satisfaction function."""
        self.satisfaction = 0.5 * (1 - np.tanh(0.1*(self.time_waited_for_bus - 30)))


class Bus:
    def __init__(self):
        self.active = False
        self.position = 0.0
        self.next_stop = 0
        self.passengers = []
        self.size = 100
        self.last = False
        self.total_passengers = 0
        self.vjitter = np.random.uniform(-2, 2)
        
    @property
    def nb_free_places(self):
        return self.size - len(self.passengers)
    
    def hop_in(self, passengers):
        """Passengers at the bus stop hop into the bus."""
        if hasattr(passengers, '__getitem__'):
            self.passengers.extend(passengers)
            self.total_passengers += len(passengers)
        else:
            self.passengers.append(passengers)
            self.total_passengers += 1
            
    def hop_out(self, stop_index, cur_time, hop_out_time):
        """Passengers that reached their destination leave the bus."""
        out_passengers = [passenger for passenger in self.passengers
                                    if passenger.dest==stop_index]
        t = 0
        for passenger in out_passengers:
            passenger.end_time = cur_time
            self.passengers.remove(passenger)
            t += hop_out_time()
        return t, out_passengers
        
    def empty(self, cur_time):
        """This function is called when the bus reaches the last stop."""
        for passenger in self.passengers:
            passenger.end_time = cur_time
            passenger.dest = self.next_stop
        out_passengers = self.passengers
        self.passengers = []
        self.active = False
        return out_passengers


class BusStop:
    def __init__(self, position=0, arrival_func=lambda : 10.0, index=0):
        self.passengers = []
        self.position = position
        self.next_arrival_time = arrival_func  # time between arrivals
        self.index = index
        
    def passenger_arrival(self, cur_time, dest):
        """Add a new passenger to the queue."""
        passenger = Passenger(source=self.index, dest=dest, start_time=cur_time)
        self.passengers.append(passenger)

    def hop_in_bus(self, cur_time, hop_in_time, bus):
        """A bus arrives, all passengers that fit into the bus hop in, others
        stay at the bus stop. Return the time it takes for passengers to hop
        into the bus."""
        nb_to_bus = min(bus.nb_free_places, len(self.passengers))
        t = 0.0
        for i in range(nb_to_bus):
            passenger = self.passengers.pop(0)
            passenger.time_waited_for_bus = cur_time - passenger.start_time
            passenger.satisfy()
            bus.hop_in(passenger)
            t += hop_in_time()
        return t


class Stats:
    def __init__(self):
        self.t = []
        self.nb_active_buses = []
        self.nb_passengers_in_active_buses = []
        self.len_queues_at_stops = []
        self.waited_times = []
        self.nb_stops_traveled = []
        self.satisfaction = []
        self.total_passengers = []
        
    def measure(self, cur_time, buses, stops, passengers):
        self.t.append(cur_time)
        active_buses = [bus for bus in buses if bus.active]
        self.nb_active_buses.append(len(active_buses))
        self.nb_passengers_in_active_buses.append([len(bus.passengers)
                                                   for bus in active_buses])
        self.len_queues_at_stops.append([len(stop.passengers) for stop in stops])
        self.total_passengers.append([bus.total_passengers for bus in buses])
        self.travel_times = [passenger.end_time - passenger.start_time
                             for passenger in passengers]
        self.waited_times = [passenger.time_waited_for_bus
                             for passenger in passengers]
        self.nb_stops_traveled = [passenger.dest - passenger.source
                                  for passenger in passengers]
        self.satisfaction = [passenger.satisfaction for passenger in passengers]
    
    def _nb_bins(self, x):
        """Use Rice rule for number of bins."""
        x = np.array(x)
        return np.ceil(2 * x ** (1.0/3.0))
    
    def plot(self):
        kwargs = {'edgecolor': 'w', 'alpha': 0.7}
        plt.figure()
        plt.hist(self.travel_times,
                 bins=self._nb_bins(len(self.travel_times)),
                 **kwargs)
        plt.xlabel('Total travel time')
        plt.ylabel('Frequency')
        plt.figure()
        plt.hist(self.waited_times,
                 bins=self._nb_bins(len(self.waited_times)),
                 **kwargs)
        plt.xlabel('Time waited for bus')
        plt.ylabel('Frequency')
        
        queues = np.array(self.len_queues_at_stops)
        nb_stops = np.size(queues, 1)
        plt.figure()
        plt.hist(self.nb_stops_traveled,
                 bins=len(set(self.nb_stops_traveled)),
                 **kwargs)
        plt.xlabel('Number of stops to destination')
        plt.ylabel('Frequency')
        avg_nb_passengers = [mean(nb_passengers)
                             for nb_passengers in self.nb_passengers_in_active_buses]
        plt.figure()
        plt.plot(self.t, avg_nb_passengers)
        plt.xlabel('Time (min)')
        plt.ylabel('Average number of passengers in buses')
        plt.figure()
        plt.plot(self.t, np.sum(self.total_passengers, axis=1))
        plt.xlabel('Time (min)')
        plt.ylabel('Total number of passengers served')
        plt.figure()
        plt.plot(self.t, self.nb_active_buses)
        y0, y1 = plt.ylim()
        plt.ylim(y0 - 0.1, y1 + 0.1)
        plt.xlabel('Time (min)')
        plt.ylabel('Number of active buses')
        plt.figure()
        plt.bar(np.arange(nb_stops) - 0.5, np.mean(queues, axis=0),
                width=1.0, **kwargs)
        plt.xlim(-0.75, nb_stops - 1.25)
        plt.xlabel('Stop index')
        plt.ylabel('Average length of queue')
        plt.figure()
        plt.hist(self.total_passengers[-1],
                 bins=self._nb_bins(len(self.total_passengers[-1])),
                 **kwargs)
        plt.xlabel('Total number of passengers per bus')
        plt.ylabel('Frequency')
        plt.figure()
        plt.hist(self.satisfaction,
                 bins=self._nb_bins(len(self.satisfaction)),
                 **kwargs)
        plt.xlabel('Satisfaction')
        plt.ylabel('Frequency')


class Simulation:
    def __init__(self,
                 bus_stop_positions=np.arange(0, 30, 3),
                 passenger_arrival_times=lambda : np.random.exponential(10.0),
                 hop_in_time=lambda : truncnorm.rvs(-1, 8, loc=0.3, scale=0.2),
                 hop_out_time=lambda : truncnorm.rvs(-1, 8, loc=0.3, scale=0.2),
                 nb_stops_to_dest=lambda : np.round(truncnorm.rvs(-1, 4, loc=4, scale=3)),
                 bus_speed=lambda : truncnorm.rvs(-2, 2, loc=0.83, scale=0.1),
                 nb_buses=50,
                 time_between_buses=lambda : 25,
                 stats_time=5):
        self.bus_stop_positions = bus_stop_positions
        self.stops = [BusStop(position=pos, index=i, arrival_func=passenger_arrival_times)
                      for i, pos in enumerate(self.bus_stop_positions)]
        self.stops[-1].next_arrival_time = lambda : np.Inf # last stop, no one hops in
        self.hop_in_time = hop_in_time
        self.hop_out_time = hop_out_time
        self.nb_stops_to_dest = nb_stops_to_dest
        self.bus_speed = bus_speed
        self.nb_buses = nb_buses
        self.time_between_buses = time_between_buses
        self.stats = None
        self.stats_time = stats_time
        
    def run(self):
        moved_passengers = []
        events = []
        global counter

        # Initialize events queue.
        for stop in self.stops:
            heapq.heappush(events, (stop.next_arrival_time(), stop))

        buses = []
        t = 0.5 * self.time_between_buses()
        for i in range(self.nb_buses):
            bus = Bus()
            buses.append(bus)
            heapq.heappush(events, (t, bus))
            t += self.time_between_buses()
        buses[-1].last = True

        # Initialize statistics collection.
        self.stats = Stats()
        heapq.heappush(events, (self.stats_time, self.stats))

        while events:
            t, obj = heapq.heappop(events)
            if isinstance(obj, BusStop):
                # New arrival at a bus stop.
                dest = obj.index + self.nb_stops_to_dest()
                obj.passenger_arrival(t, dest=dest)
                heapq.heappush(events, (t + obj.next_arrival_time(), obj))
            elif isinstance(obj, Bus):
                if not obj.active:
                    obj.active = True
                if obj.next_stop >= len(self.stops):
                    # Bus reached terminal: it empties and becomes inactive.
                    moved_passengers.extend(obj.empty(t))
                    if obj.last:
                        break
                elif self.stops[obj.next_stop].position == obj.position:
                    # Bus reached a bus stop.
                    bus_stop = self.stops[obj.next_stop]
                    wait_out, out_passengers = obj.hop_out(stop_index=bus_stop.index,
                                                           cur_time=t,
                                                           hop_out_time=self.hop_out_time)
                    moved_passengers.extend(out_passengers)
                    wait_in = bus_stop.hop_in_bus(t, self.hop_in_time, obj)
                    obj.next_stop += 1
                    heapq.heappush(events, (t + wait_out + wait_in, obj))
                else:
                    # Bus finished loading passengers, move to next stop.
                    dist = self.stops[obj.next_stop].position - obj.position
                    heapq.heappush(events, (t + self.bus_speed() * dist, obj))
                    obj.position += dist
            elif isinstance(obj, Stats):
                obj.measure(t, buses, self.stops, moved_passengers)
                heapq.heappush(events, (t + self.stats_time, obj))
                fig = plt.figure()
                ax = AA.Subplot(fig, 111)
                fig.add_axes(ax)
                ax.axis['right'].set_visible(False)
                ax.axis['top'].set_visible(False)
                plt.bar([stop.position for stop in self.stops],
                        [len(stop.passengers) for stop in self.stops],
                        width=1.5)
                for bus in [bus for bus in buses if bus.active]:
                    plt.plot(bus.position, 16 + bus.vjitter, '>',
                             color='#A60628',
                             markersize=3 + len(bus.passengers)**(2.0/3.0))
                plt.text(26, 19, '{:.0f} min'.format(t))
                plt.xlim(-0.5, 30)
                plt.ylim(0, 20)
                plt.xlabel('Position (km)')
                plt.ylabel('Nombre de passagers')
                plt.savefig('anim/fig{:05d}.png'.format(counter))
                print 'anim/fig{:05d}.png'.format(counter)
                counter += 1


counter = 1

if __name__ == '__main__':
    stop_pos = np.arange(0, 30, 2)
    nb_stops = len(stop_pos)
    mean_stops = nb_stops/2.0
    std_stops = nb_stops/4.0
    a, b = (1 - mean_stops)/std_stops, (nb_stops - mean_stops)/std_stops
    stops_to_dest = lambda: np.round(truncnorm.rvs(a, b, loc=mean_stops, scale=std_stops))
    sim = Simulation(bus_stop_positions=stop_pos,
                     time_between_buses=lambda: 40,
                     nb_stops_to_dest=stops_to_dest,
                     passenger_arrival_times=lambda : np.random.exponential(5.0),
                     stats_time=2)
    sim.run()
