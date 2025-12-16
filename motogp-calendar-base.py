import unittest
import math
import csv
import random
import numpy as np
import pyswarms as ps
import os
from simanneal import Annealer
from deap import base, creator, tools

# safe path resolver
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# constants that define the likelihood of two individuals having crossover
# performed and the probability that a child will be mutated. needed for the
# DEAP library
CXPB = 0.5
MUTPB = 0.2

# define a spurious best solution and cost. We will updat
# e this as we update the cost of each particle
swarm_best_cost = 1000000

swarm_best_itinerary = []
itineraries = []
home = 8

TRACKS = None
RACE_WEEKENDS = None

# the unit tests to check that the simulation has been implemented correctly
class UnitTests (unittest.TestCase):
    # this will read in the track locations file and will pick out 5 fields to see if the file has been read correctly
    def testReadCSV(self):
        # read in the locations file
        rows = readCSVFile('track-locations.csv')

        # test that the corners and a middle value are read in correctly
        self.assertEqual('GP', rows[0][0])
        self.assertEqual('Valencia', rows[0][22])
        self.assertEqual('Temp week 52', rows[55][0])
        self.assertEqual('16.25', rows[55][22])
        self.assertEqual('12.5', rows[11][8])
    
    # this will test to see if the row conversion works. here we will convert the latitude rwo and will test 5 values
    # as we are dealing with floating point we will use almost equals rather than a direct equality
    def testRowToFloat(self):
        # read in the locations file and convert the latitude column to floats
        rows = readCSVFile('track-locations.csv')
        convertRowToFloat(rows, 2)

        # check that 5 of the values have converted correctly
        self.assertAlmostEqual(14.957883, rows[2][1], delta=0.0001)
        self.assertAlmostEqual(39.484786, rows[2][22], delta=0.0001)
        self.assertAlmostEqual(36.532176, rows[2][17], delta=0.0001)
        self.assertAlmostEqual(-38.502284, rows[2][19], delta=0.0001)
        self.assertAlmostEqual(36.709896, rows[2][5], delta=0.0001)

        # check that the conversion of a temperature row to floating point is also correct
        convertRowToFloat(rows, 5)

        # check that 5 of the values have converted correctly
        self.assertAlmostEqual(31.5, rows[5][1], delta=0.0001)
        self.assertAlmostEqual(16.5, rows[5][22], delta=0.0001)
        self.assertAlmostEqual(8.5, rows[5][17], delta=0.0001)
        self.assertAlmostEqual(23.5, rows[5][19], delta=0.0001)
        self.assertAlmostEqual(16.5, rows[5][5], delta=0.0001)
    
    # # this will test to see if the file conversion overall is successful for the track locations
    # # it will read in the file and will test a string, float, and int from 2 rows to verify it worked correctly
    def testReadTrackLocations(self):
        # read in the locations file
        rows = readTrackLocations()

        # check the name, latitude, and final temp of the first race
        self.assertEqual(rows[0][0], 'Thailand')
        self.assertAlmostEqual(rows[2][0], 14.957883, delta=0.0001)
        self.assertAlmostEqual(rows[55][0], 30.75, delta=0.0001)

        # check the name, longitude, and initial temp of the last race        
        self.assertEqual(rows[0][21], 'Valencia')
        self.assertAlmostEqual(rows[2][21], 39.484786, delta=0.0001)
        self.assertAlmostEqual(rows[4][21], 16, delta=0.0001)
    
    # # tests to see if the race weekends file is read in correctly
    def testReadRaceWeekends(self):
        # read in the race weekends file
        weekends = readRaceWeekends()

        # check that thailand is weekend 8 and valencia is weekend 45
        self.assertEqual(weekends[0], 8)
        self.assertEqual(weekends[21], 45)

        # check that Austria is weekend 32
        self.assertEqual(weekends[12], 32)

    # # this will test to see if the haversine function will work correctly we will test 4 sets of locations
    def testHaversine(self):
        # read in the locations file with conversion
        rows = readTrackLocations()

        # check the distance of Thailand against itself this should be zero
        self.assertAlmostEqual(haversine(rows, 0, 0), 0.0, delta=0.01)
        
        # check the distance of Thailand against Silverstone this should be 9632.57 km
        self.assertAlmostEqual(haversine(rows, 0, 6), 9632.57, delta=0.01)

        # check the distance of silverstone against Mugello this should be 1283.1 Km
        self.assertAlmostEqual(haversine(rows, 6, 8), 1283.12, delta=0.01)

        # check the distance of Mugello to the red bull ring this should be 445.06 Km
        self.assertAlmostEqual(haversine(rows, 8, 12), 445.06, delta=0.01)
    
    # # will test to see if the season distance calculation is correct using the 2025 calendar
    def testDistanceCalculation(self):
        # read in the locations & race weekends, generate the weekends, and calculate the season distance
        tracks = readTrackLocations()
        weekends = readRaceWeekends()
        
        # calculate the season distance using Mugello as the home track as this will be the case for almost all of the teams we will use silverstone for the others
        self.assertAlmostEqual(calculateSeasonDistance(tracks, weekends, 8), 146768.1778, delta=0.0001)
        self.assertAlmostEqual(calculateSeasonDistance(tracks, weekends, 6), 151481.2754, delta=0.0001)
    
    # # will test that the temperature constraint is working this should fail as Azerbijan should fail the test
    def testTempConstraint(self):
        # load in the tracks, race weekends, and the sundays
        tracks = readTrackLocations()
        weekends1 = [8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 27, 28, 32, 33, 35, 36, 38, 39, 41, 42, 44, 45]
        weekends2 = [8, 10, 12, 14, 16, 18, 30, 22, 24, 25, 27, 28, 32, 33, 35, 36, 38, 39, 48, 42, 40, 41]

        # the test with the default calendar should be false because of Great Britian at 17.25
        self.assertEqual(checkTemperatureConstraint(tracks, weekends1, 20, 35), False)
        self.assertEqual(checkTemperatureConstraint(tracks, weekends2, 20, 35), True)
        
    # # will test that we can detect a period for a summer shutdown in the prescribed weeks
    def testSummerShutdown(self):
        # weekend patterns the first has a summer shutdown the second doesn't
        weekends1 = [8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 27, 28, 32, 33, 35, 36, 38, 39, 41, 42, 44, 45]
        weekends2 = [8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 27, 28, 31, 33, 35, 36, 38, 39, 41, 42, 44, 45]

        # the first should pass and the second should fail
        self.assertEqual(checkSummerShutdown(weekends1), True)
        self.assertEqual(checkSummerShutdown(weekends2), False)

# implementation of an annealer that will attempt to come up with a better calendar. it will allow the free movement of weekends with the exception of.
# monaco. bahrain and abu dhabi will open and close the season respectively. the summer shutdown should still be respected. Double and triple header weekends
# are permitted but for or more races in a row is not allowed.
class CalendarAnnealer(Annealer):
    # used to initialise the annealer the state here will be the list of race weekends and the locations. we take in all the extra parameters
    # to help calculate the distance and temperature requirements on the way
    def __init__(self, weekends, home, tracks):
        # weekends: list of 22 weeks
        # home: home race indice
        # tracks: output of readTrackLocations()
        self.tracks = tracks
        self.home = home
        self.min_temp = 15
        self.max_temp = 35
        super().__init__(weekends) # inicial state = weekends list

    # used to make a move this will take two race weekends and will swap their locations as this is the smallest change we can make
    def move(self):
        weekends = self.state

        # 1) if there is too many cold races, move them first
        cold_idx = indexLowestTemp(self.tracks, weekends, self.min_temp)
        if cold_idx != -1:
            j = random.randrange(len(weekends))
            if j != cold_idx:
                weekends[cold_idx], weekends[j] = weekends[j], weekends[cold_idx]
            return

        # 2) if there is too many hot races, move them
        hot_idx = indexHighestTemp(self.tracks, weekends, self.max_temp)
        if hot_idx != -1:
            j = random.randrange(len(weekends))
            if j != hot_idx:
                weekends[hot_idx], weekends[j] = weekends[j], weekends[hot_idx]
            return

        # 3) otherwise, swap any 2 random races (except valencia)
        valencia_index = 21
        valid_indices = [i for i in range(len(weekends)) if i != valencia_index]
        i, j = random.sample(valid_indices, 2)
        weekends[i], weekends[j] = weekends[j], weekends[i]

    # used to calculate the energy. smaller values represent smaller distances. A default value of 1,000,000 will be returned if the
    # temperature requirement is not satisfied
    def energy(self):
        weekends = self.state
        return calculateSeasonDistancePenalties(
            self.tracks,
            weekends,
            self.home,
            self.min_temp,
            self.max_temp
        )

# class that will hold the weekends for genetic algorithms
class CalendarGA(list):
    """Simple list subclass used by DEAP for the calendar."""
    pass
        
# function that will calculate the total distance for the season assuming a given racetrack as the home racetrack
# the following will be assumed:
# - on a weekend where there is no race the team will return home
# - on a weekend in a double or triple header a team will travel straight to the next race and won't go back home
# - the preseason test will always take place in Bahrain
# - for the summer shutdown and off season the team will return home
def calculateSeasonDistance(tracks, weekends, home):
    # Build a week-by-week schedule: 1..52
    # schedule[w] = race index (0..21) or -1 if no race
    NUM_WEEKS = 52
    schedule = [-1] * (NUM_WEEKS + 1)  # index 0 unused

    # weekends list: index = race index, value = week number
    for race_index, week in enumerate(weekends):
        schedule[week] = race_index

    total_distance = 0.0
    current_location = home  # track index
    at_home = True

    for week in range(1, NUM_WEEKS + 1):
        race = schedule[week]

        if race == -1:
            # No race this week, nothing to add here.
            # Returning home is handled right after race weeks.
            continue

        # There is a race this week at track index = race
        if at_home:
            total_distance += haversine(tracks, home, race)
        else:
            total_distance += haversine(tracks, current_location, race)

        current_location = race
        at_home = False

        # Look ahead to next week
        if week == NUM_WEEKS or schedule[week + 1] == -1:
            # No race next week (or end of season) -> go home after this race
            total_distance += haversine(tracks, current_location, home)
            current_location = home
            at_home = True

    return total_distance

# # function that will calculate the season distance and will include the cost of penalties in the calculation
def calculateSeasonDistancePenalties(tracks, weekends, home, min, max):
    distance = calculateSeasonDistance(tracks, weekends, home)
    penalty = 0

    # 1) Temperature constraints
    if not checkTemperatureConstraint(tracks, weekends, min, max):
        penalty += 100000

    # 2) Without summer shutdown (week 29-31)
    if not checkSummerShutdown(weekends):
        penalty += 100000

    # 3) Calendar with duplicate or missing weeks
    if len(weekends) != 22 or len(set(weekends)) != len(weekends):
        penalty += 100000

    # 4)  Valencia on the wrong week
    original = readRaceWeekends()
    original_valencia_week = original[21]
    if weekends[21] != original_valencia_week:
        penalty += 100000

    # 5) No triple headers (3+ consecutive races)
    if not checkNoTripleHeader(weekends):
        penalty += 100000

    return distance + penalty

# function that will check to see if the temperature constraint for all races is satisfied. The temperature
# constraint is that a minimum temperature of min degrees for the month is required for a race to run
def checkTemperatureConstraint(tracks, weekends, min, max):
    """
    tracks: output of readTrackLocations()
    weekends: list of weekend numbers for each race (index = race index)
    min, max: allowed temperature range
    """
    for race_index, week in enumerate(weekends):
        # week 1 -> temp row index 4; week 52 -> row 55
        temp_row = 3 + week
        temp = tracks[temp_row][race_index]

        if temp < min or temp > max:
            return False

    return True

# never 3 or more races in a row
def checkNoTripleHeader(weekends):
    race_weeks = sorted(weekends)
    consecutive = 1
    for i in range(1, len(race_weeks)):
        if race_weeks[i] == race_weeks[i-1] + 1:
            consecutive += 1
            if consecutive >= 3:
                return False
        else:
            consecutive = 1
    return True

# function that will check to see if there is a four week gap anywhere in july and august. we will need this for the summer shutdown.
# the way this is defined is that we have a gap of three weekends between successive races. this will be weeks 29, 30, and 31, they are not
# permitted to have a race during these weekends
def checkSummerShutdown(weekends):
    # No race is allowed on weeks 29, 30, 31
    forbidden = {29, 30, 31}
    for week in weekends:
        if week in forbidden:
            return False
    return True

# will go through the genetic code of this child and will make sure that all the required weekends are in it.
# it's highly likely that with crossover that there will be weekends missing and others duplicated. we will
# randomly replace the duplicated ones with the missing ones
def childGeneticCodeFix(child):
    # base calendar, all weeks validated
    base_weeks = readRaceWeekends()
    remaining = base_weeks[:]
    duplicate_indices = []

    # identify duplicated and remove from the list of "remaining" the races showed once
    for idx, w in enumerate(child):
        if w in remaining:
            remaining.remove(w)
        else:
            duplicate_indices.append(idx)

    # what left of "remaining" are the ones who is coming in child
    random.shuffle(remaining)

    for idx in duplicate_indices:
        if remaining:
            child[idx] = remaining.pop()

    return child

# function that will take in the set of rows and will convert the given row index into floating point values
# this assumes the header in the CSV file is still present so it will skip the first column
def convertRowToFloat(rows, row_index):
    row = rows[row_index]
    # skip column 0 (header, e.g. "Latitude", "Temp week X", etc.)
    for i in range(1, len(row)):
        try:
            rows[row_index][i] = float(row[i])
        except ValueError:
            # If it can't be converted, just leave it as a string
            pass

# function that will count how many elements in the given array are greater equal a specific value
def countGreaterEqual(array, value):
    return sum(1 for x in array if x >= value)

# function that will perform roulette wheel crossover to generate children
def crossoverStrategy(ind1, ind2):
    base_weeks = readRaceWeekends()
    valencia_index = 21
    valencia_week = base_weeks[valencia_index]

    for i in range(len(ind1)):
        if i == valencia_index:
            ind1[i] = valencia_week
            ind2[i] = valencia_week
            continue

        a = ind1[i]
        b = ind2[i]

        ind1[i] = rouletteWheel(a, b)
        ind2[i] = rouletteWheel(a, b)

    # fix duplicated/missing
    childGeneticCodeFix(ind1)
    childGeneticCodeFix(ind2)

    return ind1, ind2

# function that will evaluate the strategy for a stock #UNUSED
def evaluateStrategy(individual):
    """ Placeholder: simple evaluation of a stock strategy.
       Returns the negative sum so that 'larger is better' if you ever use it."""
    total = sum(individual)
    return (-total,)

# function that will generate the initial itineraries for particle swarm optimisation
# this will take the initial solution and will shuffle it to create new solutions
def generateInitialItineraries(num_particles, initial_solution):
    global itineraries
    itineraries = []

    for i in range(num_particles):
        if i == 0:
            # first particle: original calendar
            new_itinerary = initial_solution[:]
        else:
            # others: shuffled versions (keeping Valencia fixed)
            new_itinerary = generateShuffledItinerary(initial_solution)
        itineraries.append(new_itinerary)

    return itineraries

# function that will generate a shuffled itinerary. However, this will make sure that the bahrain, abu dhabi, and monaco
# will retain their fixed weeks in the calendar
def generateShuffledItinerary(weekends):
    new_weekends = weekends[:]

    valencia_index = 21 # Valencia GP 22º on te list
    valencia_week = new_weekends[valencia_index]

    # Other races indices
    other_indices = [i for i in range(len(new_weekends)) if i != valencia_index]
    other_weeks = [new_weekends[i] for i in other_indices]

    random.shuffle(other_weeks)

    for idx, week in zip(other_indices, other_weeks):
        new_weekends[idx] = week

    # Make sure Valencia remain on the original week
    new_weekends[valencia_index] = valencia_week

    return new_weekends

# function that will use the haversine formula to calculate the distance in Km given two latitude/longitude pairs
# it will take in an index to two rows, and extract the latitude and longitude before the calculation.
def haversine(rows, location1, location2):
    # rows: output of readTrackLocations()
    # location1, location2: column indices of the two tracks

    # Latitude and longitude rows after readTrackLocations()
    LAT_ROW = 2
    LON_ROW = 3

    lat1 = math.radians(rows[LAT_ROW][location1])
    lon1 = math.radians(rows[LON_ROW][location1])
    lat2 = math.radians(rows[LAT_ROW][location2])
    lon2 = math.radians(rows[LON_ROW][location2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)

    c = 2 * math.asin(math.sqrt(a))
    R = 6371  # radius of Earth in km

    return R * c

# function that will initialise a strategy for the stocks. we will randomise all the weights here
def initIndividual(ind_class):
    base_weekends = readRaceWeekends()
    shuffled = generateShuffledItinerary(base_weekends)
    return ind_class(shuffled)

# function that will give us the index of the lowest temp below min. will return -1 if none found
def indexHighestTemp(tracks, weekends, max):
    worst_index = -1
    worst_temp = max

    for race_index, week in enumerate(weekends):
        temp_row = 3 + week
        temp = tracks[temp_row][race_index]
        if temp > max and temp > worst_temp:
            worst_temp = temp
            worst_index = race_index

    return worst_index

# function that will give us the index of the lowest temp below min. will return -1 if none found
def indexLowestTemp(tracks, weekends, min):
    worst_index = -1
    worst_temp = min

    for race_index, week in enumerate(weekends):
        temp_row = 3 + week
        temp = tracks[temp_row][race_index]
        if temp < min and temp < worst_temp:
            worst_temp = temp
            worst_index = race_index

    return worst_index

# function that will mutate an individual
def mutateIndividual(individual, indpb):
    global TRACKS
    tracks = TRACKS
    if tracks is None:
        tracks = readTrackLocations()
        TRACKS = tracks

    min_temp = 15
    max_temp = 35

    weekends = list(individual)
    valencia_index = 21

    # 1) too cold races
    cold_idx = indexLowestTemp(tracks, weekends, min_temp)
    if cold_idx != -1 and cold_idx != valencia_index:
        # chose a different indice
        choices = [i for i in range(len(weekends)) if i not in (cold_idx, valencia_index)]
        if choices:
            j = random.choice(choices)
            weekends[cold_idx], weekends[j] = weekends[j], weekends[cold_idx]
            individual[:] = weekends
            return individual,

    # 2) too hot races
    hot_idx = indexHighestTemp(tracks, weekends, max_temp)
    if hot_idx != -1 and hot_idx != valencia_index:
        choices = [i for i in range(len(weekends)) if i not in (hot_idx, valencia_index)]
        if choices:
            j = random.choice(choices)
            weekends[hot_idx], weekends[j] = weekends[j], weekends[hot_idx]
            individual[:] = weekends
            return individual,

    # 3) neither too hot/cold -> swap two random indices (apart valencia)
    valid_indices = [i for i in range(len(weekends)) if i != valencia_index]
    if len(valid_indices) >= 2:
        i, j = random.sample(valid_indices, 2)
        weekends[i], weekends[j] = weekends[j], weekends[i]
        individual[:] = weekends

    return individual,

# objective function for particle swarm optimisation
def objectiveCalendar(particles):
    """Particles: array (n_particles, dimensions) from pyswarms.
       For each particle we apply the swaps to its current itinerary
       and compute the distance + penalties."""
    global itineraries, swarm_best_cost, swarm_best_itinerary, TRACKS

    tracks = TRACKS
    if tracks is None:
        tracks = readTrackLocations()
        TRACKS = tracks

    base_itinerary = readRaceWeekends()

    costs = []

    for particle in particles:
        candidate = swapElements(base_itinerary, particle)

        cost = calculateSeasonDistancePenalties(tracks, candidate, home, 15, 35)
        costs.append(cost)

        if cost < swarm_best_cost:
            swarm_best_cost = cost
            swarm_best_itinerary = candidate[:]

    return np.array(costs)

# prints out the itinerary that was generated on a weekend by weekend basis starting from the preaseason test
def printItinerary(tracks, weekends, home):
    NUM_WEEKS = 52

    # Week to week agenda: week -> race_index or -1
    schedule = [-1] * (NUM_WEEKS + 1)
    for race_index, week in enumerate(weekends):
        schedule[week] = race_index

    current_location = home
    at_home = True

    for week in range(1, NUM_WEEKS + 1):
        race = schedule[week]

        if at_home and race == -1:
            print("Staying at home thus no travel this weekend")

        elif at_home and race != -1:
            temp_row = 3 + week
            temp = tracks[temp_row][race]
            name = tracks[0][race]
            print(f"Travelling from home to {name}. "
                  f"Race temperature is expected to be {temp:.2f} degrees")
            current_location = race
            at_home = False

        elif not at_home and race != -1:
            temp_row = 3 + week
            temp = tracks[temp_row][race]
            current_name = tracks[0][current_location]
            next_name = tracks[0][race]
            print(f"Travelling directly from {current_name} to {next_name}. "
                  f"Race temperature is expected to be {temp:.2f} degrees")
            current_location = race

        elif not at_home and race == -1:
            current_name = tracks[0][current_location]
            print(f"Travelling home from {current_name}")
            current_location = home
            at_home = True

# function that will take in the given CSV file and will read in its entire contents
# and return a list of lists
def readCSVFile(file):
    """
    Read a CSV file and return its rows as a list of lists.
    Uses context manager for safe file handling.
    """
    path = os.path.join(BASE_DIR, file)
    rows = []
    with open(path, newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            rows.append(row)
    return rows

# function that will read in the race weekends file and will perform all necessary conversions on it
def readRaceWeekends():
    """
    Reads and parses race-weekends.csv.
    Cached after first load to avoid repeated disk I/O.
    """
    global RACE_WEEKENDS
    if RACE_WEEKENDS is not None:
        return RACE_WEEKENDS

    rows = readCSVFile('race-weekends.csv')

    weekends = []
    for row in rows[1:]:
        if len(row) < 2 or row[1].strip() == '':
            continue
        weekends.append(int(row[1]))

    RACE_WEEKENDS = weekends
    return RACE_WEEKENDS

# function that will read the track locations file and will perform all necessary conversions on it.
# this should also strip out the first column on the left which is the header information for all the rows
def readTrackLocations():
    """
        Reads and parses track-locations.csv.
        Cached after first load to avoid repeated disk I/O.
        """
    global TRACKS
    if TRACKS is not None:
        return TRACKS

    raw_rows = readCSVFile('track-locations.csv')

    # Strip the first (label) column from every row
    rows = [row[1:] for row in raw_rows]

    # Convert numeric rows (lat/long + all temp rows) fully to float
    # Row 0: GP names (strings)
    # Row 1: probably some header info, leave as-is
    for r in range(2, len(rows)):
        for c in range(len(rows[r])):
            try:
                rows[r][c] = float(rows[r][c])
            except ValueError:
                # if it's not numeric, leave it
                pass

    TRACKS = rows
    return TRACKS

# function that performs a roulette wheel randomisation on the two given values and returns the chosen on
def rouletteWheel(a, b):
    # 50/50 choice between a and b
    r = random.random()  # in [0, 1)
    return a if r < 0.5 else b

# function that will take an itinerary and will swap the elements based on the values in the particle.
# If only one element is selected for swapping it will be randomly swapped with another element.
# if two or more are selected we will take those elements and shuffle them
def swapElements(itinerary, particle):
    valencia_index = 21
    new_itinerary = itinerary[:]

    indices = swapIndexes(particle, k=5)

    # organise the indices by particle's value
    ranked = sorted(indices, key=lambda i: particle[i], reverse=True)

    # get the current indices values
    vals = [new_itinerary[i] for i in ranked]

    # rotation: move the last one to the front position
    vals = [vals[-1]] + vals[:-1]

    # apply and return
    for i, v in zip(ranked, vals):
        new_itinerary[i] = v

    # lock valencia (just to make sure)
    new_itinerary[valencia_index] = itinerary[valencia_index]
    return new_itinerary

# function that will return a list of the indexes to be swaped according to the particle
def swapIndexes(particle, k=7):
    """
    PSO 'particle' is continuous. We interpret it as:
    - pick the top-k dimensions and swap/shuffle those positions.
    This avoids relying on thresholds (e.g., >0.5) when particles can be negative.
    Valencia index is excluded.
    """
    valencia_index = 21
    idx_sorted = np.argsort(np.abs(particle))[::-1]  # bigger's first
    return [int(i) for i in idx_sorted if int(i) != valencia_index][:k]

# function that will take an itinerary and will swap a pair of weekends without changing valencia
def swapPair(itinerary):
    # Swap two random positions in the itinerary, never moving Valencia.
    valencia_index = 21
    indices = [i for i in range(len(itinerary)) if i != valencia_index]

    if len(indices) < 2:
        return itinerary

    i, j = random.sample(indices, 2)
    itinerary[i], itinerary[j] = itinerary[j], itinerary[i]
    return itinerary

# function that will swap the values at the given index with another randomly chosen index
def swapIndex(itinerary, index):
    # Swap the weekend at 'index' with a random other index (not Valencia).
    valencia_index = 21
    n = len(itinerary)

    choices = [i for i in range(n) if i != index and i != valencia_index]
    if not choices:
        return itinerary

    j = random.choice(choices)
    itinerary[index], itinerary[j] = itinerary[j], itinerary[index]
    return itinerary

# function that will run the simulated annealing case for shortening the distance sepparately for both silverstone and monza
def SAcases():
    tracks = readTrackLocations()
    base_weekends = readRaceWeekends()

    # generate an inicial shuffled itinerary (apart valencia)
    initial = generateShuffledItinerary(base_weekends)
    # mugello as home
    annealer = CalendarAnnealer(initial, home, tracks)

    # suggested parameters
    annealer.Tmax = 25000.0
    annealer.Tmin = 2.5
    annealer.steps = 100000
    annealer.updates = 50

    best_state, best_energy = annealer.anneal()

    print("\n=== Simulated Annealing result ===")
    print("Best energy (distance + penalties):", best_energy)
    print("Best calendar (weekends):", best_state)
    print("Distance (no penalties):",
          calculateSeasonDistance(tracks, best_state, home))

    return best_state, best_energy

# function that will run the genetic algorithms cases for all four situations
def GAcases():
    tracks = readTrackLocations()

    # create types of DEAP only once
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", CalendarGA, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # individual and population
    toolbox.register("individual", initIndividual, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # checking function: distance + penalities
    def eval_calendar(individual):
        weekends = list(individual)
        value = calculateSeasonDistancePenalties(tracks, weekends, home, 15, 35)
        return (value,)

    toolbox.register("evaluate", eval_calendar)
    toolbox.register("mate", crossoverStrategy)
    toolbox.register("mutate", mutateIndividual, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # GA parameters
    POP_SIZE = 300
    NGEN = 1000

    pop = toolbox.population(n=POP_SIZE)

    # check initial population
    for ind, fit in zip(pop, map(toolbox.evaluate, pop)):
        ind.fitness.values = fit

    for gen in range(NGEN):
        # selection
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))

        # crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # mutation
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # check the individual that lost the fitness
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        pop[:] = offspring

        # log simples
        if (gen + 1) % 100 == 0:
            best = tools.selBest(pop, 1)[0]
            print(f"Gen {gen+1}: best energy = {best.fitness.values[0]:.2f}")

    best = tools.selBest(pop, 1)[0]
    best_weekends = list(best)

    print("\n=== Genetic Algorithm Result ===")
    print("Best energy (distance + penalties):", best.fitness.values[0])
    print("Best calendar (weekends):", best_weekends)
    print("Distance (no penalties):",
          calculateSeasonDistance(tracks, best_weekends, home))

    return best_weekends, best.fitness.values[0]

# function that will run particle swarm optimisation in an attempt to find a solution
def PSOcases(seed_calendar=None):
    global itineraries, swarm_best_cost, swarm_best_itinerary, TRACKS

    tracks = readTrackLocations()
    TRACKS = tracks
    base_weekends = seed_calendar[:] if seed_calendar else readRaceWeekends()

    n_particles = 100
    dimensions = len(base_weekends)

    # reset global best
    swarm_best_cost = 1_000_000.0
    swarm_best_itinerary = base_weekends[:]

    # initial itineraries for each particle
    bounds = (np.full(dimensions, -2.0), np.full(dimensions, 2.0))

    # PSO options (inertia + components cognitive/social)
    options = {'c1': 1.7, 'c2': 1.7, 'w': 0.6}

    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=dimensions,
        options=options,
        bounds=bounds
    )

    # run the PSO
    best_cost, best_pos = optimizer.optimize(objectiveCalendar, iters=1000)

    best_distance = calculateSeasonDistance(tracks, swarm_best_itinerary, home)
    penalties = swarm_best_cost - best_distance

    print("\n=== Particle Swarm Optimisation Result ===")
    print(f"pyswarms best_cost (continuous space): {best_cost:.4f}")
    print(f"Best energy (distance + penalties): {swarm_best_cost:.4f}")
    print(f"Distance (no penalties): {best_distance:.4f}")
    print(f"Penalties applied: {penalties:.4f}")
    print("Best calendar (weekends):", swarm_best_itinerary)

    ok_temp = checkTemperatureConstraint(tracks, swarm_best_itinerary, 15, 35)
    ok_shutdown = checkSummerShutdown(swarm_best_itinerary)
    ok_unique = (len(swarm_best_itinerary) == 22 and len(set(swarm_best_itinerary)) == 22)
    ok_valencia = (swarm_best_itinerary[21] == readRaceWeekends()[21])
    ok_triple = checkNoTripleHeader(swarm_best_itinerary)

    print("\n--- Constraint check ---")
    print("Temp ok:", ok_temp)
    print("Summer shutdown ok:", ok_shutdown)
    print("No duplicates ok:", ok_unique)
    print("Valencia fixed ok:", ok_valencia, "| valencia week:", swarm_best_itinerary[21])
    print("No triple header ok:", ok_triple)

    base = readRaceWeekends()
    base_dist = calculateSeasonDistance(tracks, base, home)

    print(f"\nBaseline (original) distance: {base_dist:.4f}")
    print(f"PSO best distance: {best_distance:.4f}")
    print(f"Improvement: {base_dist - best_distance:.4f} km")

    return swarm_best_itinerary, swarm_best_cost

def compareStrategies():
    tracks = readTrackLocations()

    sa_cal, sa_cost = SAcases()
    ga_cal, ga_cost = GAcases()
    pso_cal, pso_cost = PSOcases(seed_calendar=sa_cal)

    results = [
        ("Simulated Annealing", sa_cal, sa_cost),
        ("Genetic Algorithm", ga_cal, ga_cost),
        ("Particle Swarm", pso_cal, pso_cost),
    ]
    results.sort(key=lambda x: x[2])

    name, cal, cost = results[0]
    dist = calculateSeasonDistance(tracks, cal, home)

    print("\n=== Best Overall Strategy ===")
    print("Winner:", name)
    print("Best energy (distance + penalties):", cost)
    print("Distance (no penalties):", dist)
    print("Best calendar (weekends):", cal)
    printItinerary(tracks, cal, home)

if __name__ == '__main__':
    # uncomment this run all the unit tests. when you have satisfied all the unit tests you will have a working simulation
    # you can then comment this out and move onto your SA and GA solutions
    unittest.main(exit=False)
    compareStrategies()

    # just to check that the itinerary printing mechanism works. we will assume that silverstone is the home track for this
    #weekends = readRaceWeekends()
    #print(generateShuffledItinerary(weekends))
    #tracks = readTrackLocations()
    #printItinerary(tracks, weekends, 11)

    # run the cases for simulated annealing
    #SAcases()

    # run the cases for genetic algorithms
    #GAcases()

    # run the cases for particle swarm optimisation
    #PSOcases()
