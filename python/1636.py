# # Computing optimal road trips on a limited budget
# 
# This notebook provides the methodology and code used in the blog post, [Computing optimal road trips on a limited budget](http://www.randalolson.com/2016/06/05/computing-optimal-road-trips-on-a-limited-budget/).
# 
# ### Notebook by [Randal S. Olson](http://www.randalolson.com)
# 
# Please see the [repository README file](https://github.com/rhiever/Data-Analysis-and-Machine-Learning-Projects#license) for the licenses and usage terms for the instructional material and code in this notebook. In general, I have licensed this material so that it is as widely useable and shareable as possible.
# 
# ### Required Python libraries
# 
# If you don't have Python on your computer, you can use the [Anaconda Python distribution](http://continuum.io/downloads) to install most of the Python packages you need. Anaconda provides a simple double-click installer for your convenience.
# 
# This code uses base Python libraries except for the `googlemaps`, `pandas`, `deap`, and `tqdm` packages. You can install these packages using `pip` by typing the following commands into your command line:
# 
# > pip install googlemaps pandas deap tqdm
# 
# If you're on a Mac, Linux, or Unix machine, you may need to type `sudo` before the command to install the package with administrator privileges.
# 
# ### Construct a list of road trip waypoints
# 
# The first step is to decide where you want to stop on your road trip.
# 
# Make sure you look all of the locations up on [Google Maps](http://maps.google.com) first so you have the correct address, city, state, etc. If the text you use to look up the location doesn't work on Google Maps, then it won't work here either.
# 
# Add all of your waypoints to the list below. Make sure they're formatted the same way as in the example below.
# 
# *Technical note: Due to daily usage limitations of the Google Maps API, you can only have a maximum of 70 waypoints. You will have to pay Google for an increased API limit if you want to add more waypoints.*
# 

# https://en.wikipedia.org/wiki/List_of_state_capitols_in_the_United_States

all_waypoints = ['Alabama State Capitol, 600 Dexter Avenue, Montgomery, AL 36130',
                 #'Alaska State Capitol, Juneau, AK',
                 'Arizona State Capitol, 1700 W Washington St, Phoenix, AZ 85007',
                 'Arkansas State Capitol, 500 Woodlane Street, Little Rock, AR 72201',
                 'L St & 10th St, Sacramento, CA 95814',
                 '200 E Colfax Ave, Denver, CO 80203',
                 'Connecticut State Capitol, 210 Capitol Ave, Hartford, CT 06106',
                 'Legislative Hall: The State Capitol, Legislative Avenue, Dover, DE 19901',
                 '402 S Monroe St, Tallahassee, FL 32301',
                 'Georgia State Capitol, Atlanta, GA 30334',
                 #'Hawaii State Capitol, 415 S Beretania St, Honolulu, HI 96813'
                 '700 W Jefferson St, Boise, ID 83720',
                 'Illinois State Capitol, Springfield, IL 62756',
                 'Indiana State Capitol, Indianapolis, IN 46204',
                 'Iowa State Capitol, 1007 E Grand Ave, Des Moines, IA 50319',
                 '300 SW 10th Ave, Topeka, KS 66612',
                 'Kentucky State Capitol Building, 700 Capitol Avenue, Frankfort, KY 40601',
                 'Louisiana State Capitol, Baton Rouge, LA 70802',
                 'Maine State House, Augusta, ME 04330',
                 'Maryland State House, 100 State Cir, Annapolis, MD 21401',
                 'Massachusetts State House, Boston, MA 02108',
                 'Michigan State Capitol, Lansing, MI 48933',
                 'Minnesota State Capitol, St Paul, MN 55155',
                 '400-498 N West St, Jackson, MS 39201',
                 'Missouri State Capitol, Jefferson City, MO 65101',
                 'Montana State Capitol, 1301 E 6th Ave, Helena, MT 59601',
                 'Nebraska State Capitol, 1445 K Street, Lincoln, NE 68509',
                 'Nevada State Capitol, Carson City, NV 89701',
                 'State House, 107 North Main Street, Concord, NH 03303',
                 'New Jersey State House, Trenton, NJ 08608',
                 'New Mexico State Capitol, Santa Fe, NM 87501',
                 'New York State Capitol, State St. and Washington Ave, Albany, NY 12224',
                 'North Carolina State Capitol, Raleigh, NC 27601',
                 'North Dakota State Capitol, Bismarck, ND 58501',
                 'Ohio State Capitol, 1 Capitol Square, Columbus, OH 43215',
                 'Oklahoma State Capitol, Oklahoma City, OK 73105',
                 'Oregon State Capitol, 900 Court St NE, Salem, OR 97301',
                 'Pennsylvania State Capitol Building, North 3rd Street, Harrisburg, PA 17120',
                 'Rhode Island State House, 82 Smith Street, Providence, RI 02903',
                 'South Carolina State House, 1100 Gervais Street, Columbia, SC 29201',
                 '500 E Capitol Ave, Pierre, SD 57501',
                 'Tennessee State Capitol, 600 Charlotte Avenue, Nashville, TN 37243',
                 'Texas Capitol, 1100 Congress Avenue, Austin, TX 78701',
                 'Utah State Capitol, Salt Lake City, UT 84103',
                 'Vermont State House, 115 State Street, Montpelier, VT 05633',
                 'Virginia State Capitol, Richmond, VA 23219',
                 'Washington State Capitol Bldg, 416 Sid Snyder Ave SW, Olympia, WA 98504',
                 'West Virginia State Capitol, Charleston, WV 25317',
                 '2 E Main St, Madison, WI 53703',
                 'Wyoming State Capitol, Cheyenne, WY 82001']

len(all_waypoints)


# Next you'll have to register this script with the Google Maps API so they know who's hitting their servers with hundreds of Google Maps routing requests.
# 
# 1) Enable the Google Maps Distance Matrix API on your Google account. Google explains how to do that [here](https://github.com/googlemaps/google-maps-services-python#api-keys).
# 
# 2) Copy and paste the API key they had you create into the code below.
# 

import googlemaps

gmaps = googlemaps.Client(key='ENTER YOUR GOOGLE MAPS KEY HERE')


# Now we're going to query the Google Maps API for the shortest route between all of the waypoints.
# 
# This is equivalent to doing Google Maps directions lookups on the Google Maps site, except now we're performing hundreds of lookups automatically using code.
# 
# If you get an error on this part, that most likely means one of the waypoints you entered couldn't be found on Google Maps. Another possible reason for an error here is if it's not possible to drive between the points, e.g., finding the driving directions between Hawaii and Florida will return an error until we invent flying cars.
# 
# ### Gather the distance traveled on the shortest route between all waypoints
# 

from itertools import combinations

waypoint_distances = {}
waypoint_durations = {}

for (waypoint1, waypoint2) in combinations(all_waypoints, 2):
    try:
        route = gmaps.distance_matrix(origins=[waypoint1],
                                      destinations=[waypoint2],
                                      mode='driving', # Change this to 'walking' for walking directions,
                                                      # 'bicycling' for biking directions, etc.
                                      language='English',
                                      units='metric')

        # 'distance' is in meters
        distance = route['rows'][0]['elements'][0]['distance']['value']

        # 'duration' is in seconds
        duration = route['rows'][0]['elements'][0]['duration']['value']

        waypoint_distances[frozenset([waypoint1, waypoint2])] = distance
        waypoint_durations[frozenset([waypoint1, waypoint2])] = duration
    
    except Exception as e:
        print('Error with finding the route between {} and {}.'.format(waypoint1, waypoint2))


# Now that we have the routes between all of our waypoints, let's save them to a text file so we don't have to bother Google about them again.
# 

with open('my-waypoints-dist-dur.tsv', 'w') as out_file:
    out_file.write('\t'.join(['waypoint1',
                              'waypoint2',
                              'distance_m',
                              'duration_s']))
    
    for (waypoint1, waypoint2) in waypoint_distances.keys():
        out_file.write('\n' +
                       '\t'.join([waypoint1,
                                  waypoint2,
                                  str(waypoint_distances[frozenset([waypoint1, waypoint2])]),
                                  str(waypoint_durations[frozenset([waypoint1, waypoint2])])]))


# ### Use a genetic algorithm to optimize the order to visit the waypoints in
# 
# Instead of exhaustively looking at every possible solution, genetic algorithms start with a handful of random solutions and continually tinkers with these solutions — always trying something slightly different from the current solutions and keeping the best ones — until they can’t find a better solution any more.
# 
# Below, all you need to do is make sure that the file name above matches the file name below (both currently `my-waypoints-dist-dur.tsv`) and run the code. The code will read in your route information and use a genetic algorithm to discover an optimized driving route.
# 

import pandas as pd
import numpy as np

waypoint_distances = {}
waypoint_durations = {}
all_waypoints = set()

waypoint_data = pd.read_csv('my-waypoints-dist-dur.tsv', sep='\t')

for i, row in waypoint_data.iterrows():
    # Distance = meters
    waypoint_distances[frozenset([row.waypoint1, row.waypoint2])] = row.distance_m
    
    # Duration = hours
    waypoint_durations[frozenset([row.waypoint1, row.waypoint2])] = row.duration_s / (60. * 60.)
    all_waypoints.update([row.waypoint1, row.waypoint2])


import random
import numpy as np
import copy
from tqdm import tqdm

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

creator.create('FitnessMulti', base.Fitness, weights=(1.0, -1.0))
creator.create('Individual', list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register('waypoints', random.sample, all_waypoints, random.randint(2, 20))
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.waypoints)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

def eval_capitol_trip(individual):
    """
        This function returns the total distance traveled on the current road trip
        as well as the number of waypoints visited in the trip.
        
        The genetic algorithm will favor road trips that have shorter
        total distances traveled and more waypoints visited.
    """
    trip_length = 0.
    individual = list(individual)
    
    # Adding the starting point to the end of the trip forces it to be a round-trip
    individual += [individual[0]]
    
    for index in range(1, len(individual)):
        waypoint1 = individual[index - 1]
        waypoint2 = individual[index]
        trip_length += waypoint_distances[frozenset([waypoint1, waypoint2])]
        
    return len(set(individual)), trip_length

def pareto_selection_operator(individuals, k):
    """
        This function chooses what road trips get copied into the next generation.
        
        The genetic algorithm will favor road trips that have shorter
        total distances traveled and more waypoints visited.
    """
    return tools.selNSGA2(individuals, int(k / 5.)) * 5

def mutation_operator(individual):
    """
        This function applies a random change to one road trip:
        
            - Insert: Adds one new waypoint to the road trip
            - Delete: Removes one waypoint from the road trip
            - Point: Replaces one waypoint with another different one
            - Swap: Swaps the places of two waypoints in the road trip
    """
    possible_mutations = ['swap']
    
    if len(individual) < len(all_waypoints):
        possible_mutations.append('insert')
        possible_mutations.append('point')
    if len(individual) > 2:
        possible_mutations.append('delete')
    
    mutation_type = random.sample(possible_mutations, 1)[0]
    
    # Insert mutation
    if mutation_type == 'insert':
        waypoint_to_add = individual[0]
        while waypoint_to_add in individual:
            waypoint_to_add = random.sample(all_waypoints, 1)[0]
            
        index_to_insert = random.randint(0, len(individual) - 1)
        individual.insert(index_to_insert, waypoint_to_add)
    
    # Delete mutation
    elif mutation_type == 'delete':
        index_to_delete = random.randint(0, len(individual) - 1)
        del individual[index_to_delete]
    
    # Point mutation
    elif mutation_type == 'point':
        waypoint_to_add = individual[0]
        while waypoint_to_add in individual:
            waypoint_to_add = random.sample(all_waypoints, 1)[0]
        
        index_to_replace = random.randint(0, len(individual) - 1)
        individual[index_to_replace] = waypoint_to_add
        
    # Swap mutation
    elif mutation_type == 'swap':
        index1 = random.randint(0, len(individual) - 1)
        index2 = index1
        while index2 == index1:
            index2 = random.randint(0, len(individual) - 1)
            
        individual[index1], individual[index2] = individual[index2], individual[index1]
    
    return individual,


toolbox.register('evaluate', eval_capitol_trip)
toolbox.register('mutate', mutation_operator)
toolbox.register('select', pareto_selection_operator)

def pareto_eq(ind1, ind2):
    return np.all(ind1.fitness.values == ind2.fitness.values)

pop = toolbox.population(n=1000)
hof = tools.ParetoFront(similar=pareto_eq)
stats = tools.Statistics(lambda ind: (int(ind.fitness.values[0]), round(ind.fitness.values[1], 2)))
stats.register('Minimum', np.min, axis=0)
stats.register('Maximum', np.max, axis=0)
# This stores a copy of the Pareto front for every generation of the genetic algorithm
stats.register('ParetoFront', lambda x: copy.deepcopy(hof))
# This is a hack to make the tqdm progress bar work
stats.register('Progress', lambda x: pbar.update())

# How many iterations of the genetic algorithm to run
# The more iterations you allow it to run, the better the solutions it will find
total_gens = 5000

pbar = tqdm(total=total_gens)
pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0., mutpb=1.0, ngen=total_gens, 
                               stats=stats, halloffame=hof, verbose=False)
pbar.close()


# ## Animated road trip map
# 
# Now that we've optimized the road trip, let's visualize it!
# 
# The function below will take the results of the genetic algorithm and generate an animated map showing the Pareto optimized road trips.
# 

def create_animated_road_trip_map(optimized_routes):
    """
        This function takes a list of optimized road trips and generates
        an animated map of them using the Google Maps API.
    """
    
    # This line makes the road trips round trips
    optimized_routes = [list(route) + [route[0]] for route in optimized_routes]

    Page_1 = """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="initial-scale=1.0, user-scalable=no">
        <meta name="description" content="Randy Olson uses machine learning to find the optimal road trip across the U.S.">
        <meta name="author" content="Randal S. Olson">
        
        <title>An optimized road trip across the U.S. according to machine learning</title>
        <style>
          html, body, #map-canvas {
              height: 100%;
              margin: 0px;
              padding: 0px
          }
          #panel {
              position: absolute;
              top: 5px;
              left: 50%;
              margin-left: -180px;
              z-index: 5;
              background-color: #fff;
              padding: 10px;
              border: 1px solid #999;
          }
        </style>
        <script src="https://maps.googleapis.com/maps/api/js?v=3"></script>
        <script>
            var routesList = [];
            var markerOptions = {icon: "http://maps.gstatic.com/mapfiles/markers2/marker.png"};
            var directionsDisplayOptions = {preserveViewport: true,
                                            markerOptions: markerOptions};
            var directionsService = new google.maps.DirectionsService();
            var map;
            var mapNum = 0;
            var numRoutesRendered = 0;
            var numRoutes = 0;
            
            function initialize() {
                var center = new google.maps.LatLng(39, -96);
                var mapOptions = {
                    zoom: 5,
                    center: center
                };
                map = new google.maps.Map(document.getElementById("map-canvas"), mapOptions);
                for (var i = 0; i < routesList.length; i++) {
                    routesList[i].setMap(map); 
                }
            }
            function calcRoute(start, end, routes) {
                var directionsDisplay = new google.maps.DirectionsRenderer(directionsDisplayOptions);
                var waypts = [];
                for (var i = 0; i < routes.length; i++) {
                    waypts.push({
                        location:routes[i],
                        stopover:true});
                    }

                var request = {
                    origin: start,
                    destination: end,
                    waypoints: waypts,
                    optimizeWaypoints: false,
                    travelMode: google.maps.TravelMode.DRIVING
                };
                directionsService.route(request, function(response, status) {
                    if (status == google.maps.DirectionsStatus.OK) {
                        directionsDisplay.setDirections(response);
                        directionsDisplay.setMap(map);
                        numRoutesRendered += 1;
                        
                        if (numRoutesRendered == numRoutes) {
                            mapNum += 1;
                            if (mapNum < 47) {
                                setTimeout(function() {
                                    return createRoutes(allRoutes[mapNum]);
                                }, 5000);
                            }
                        }
                    }
                });
                
                routesList.push(directionsDisplay);
            }
            function createRoutes(route) {
                // Clear the existing routes (if any)
                for (var i = 0; i < routesList.length; i++) {
                    routesList[i].setMap(null);
                }
                routesList = [];
                numRoutes = Math.floor((route.length - 1) / 9 + 1);
                numRoutesRendered = 0;
            
                // Google's free map API is limited to 10 waypoints so need to break into batches
                var subset = 0;
                while (subset < route.length) {
                    var waypointSubset = route.slice(subset, subset + 10);
                    var startPoint = waypointSubset[0];
                    var midPoints = waypointSubset.slice(1, waypointSubset.length - 1);
                    var endPoint = waypointSubset[waypointSubset.length - 1];
                    calcRoute(startPoint, endPoint, midPoints);
                    subset += 9;
                }
            }
            
            allRoutes = [];
            """
    Page_2 = """
            createRoutes(allRoutes[mapNum]);
            google.maps.event.addDomListener(window, "load", initialize);
        </script>
      </head>
      <body>
        <div id="map-canvas"></div>
      </body>
    </html>
    """

    with open('us-state-capitols-animated-map.html', 'w') as output_file:
        output_file.write(Page_1)
        for route in optimized_routes:
            output_file.write('allRoutes.push({});'.format(str(route)))
        output_file.write(Page_2)

create_animated_road_trip_map(reversed(hof))


get_ipython().system('open us-state-capitols-animated-map.html')


# ## Individual road trip maps
# 
# We can also visualize single trips at a time instead of
# 

def create_individual_road_trip_maps(optimized_routes):
    """
        This function takes a list of optimized road trips and generates
        individual maps of them using the Google Maps API.
    """
    
    # This line makes the road trips round trips
    optimized_routes = [list(route) + [route[0]] for route in optimized_routes]

    for route_num, route in enumerate(optimized_routes):
        Page_1 = """
        <!DOCTYPE html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="initial-scale=1.0, user-scalable=no">
            <meta name="description" content="Randy Olson uses machine learning to find the optimal road trip across the U.S.">
            <meta name="author" content="Randal S. Olson">

            <title>An optimized road trip across the U.S. according to machine learning</title>
            <style>
              html, body, #map-canvas {
                  height: 100%;
                  margin: 0px;
                  padding: 0px
              }
              #panel {
                  position: absolute;
                  top: 5px;
                  left: 50%;
                  margin-left: -180px;
                  z-index: 5;
                  background-color: #fff;
                  padding: 10px;
                  border: 1px solid #999;
              }
            </style>
            <script src="https://maps.googleapis.com/maps/api/js?v=3"></script>
            <script>
                var routesList = [];
                var markerOptions = {icon: "http://maps.gstatic.com/mapfiles/markers2/marker.png"};
                var directionsDisplayOptions = {preserveViewport: true,
                                                markerOptions: markerOptions};
                var directionsService = new google.maps.DirectionsService();
                var map;

                function initialize() {
                    var center = new google.maps.LatLng(39, -96);
                    var mapOptions = {
                        zoom: 5,
                        center: center
                    };
                    map = new google.maps.Map(document.getElementById("map-canvas"), mapOptions);
                    for (var i = 0; i < routesList.length; i++) {
                        routesList[i].setMap(map); 
                    }
                }
                function calcRoute(start, end, routes) {
                    var directionsDisplay = new google.maps.DirectionsRenderer(directionsDisplayOptions);
                    var waypts = [];
                    for (var i = 0; i < routes.length; i++) {
                        waypts.push({
                            location:routes[i],
                            stopover:true});
                        }

                    var request = {
                        origin: start,
                        destination: end,
                        waypoints: waypts,
                        optimizeWaypoints: false,
                        travelMode: google.maps.TravelMode.DRIVING
                    };
                    directionsService.route(request, function(response, status) {
                        if (status == google.maps.DirectionsStatus.OK) {
                            directionsDisplay.setDirections(response);
                            directionsDisplay.setMap(map);
                        }
                    });

                    routesList.push(directionsDisplay);
                }
                function createRoutes(route) {
                    // Google's free map API is limited to 10 waypoints so need to break into batches
                    var subset = 0;
                    while (subset < route.length) {
                        var waypointSubset = route.slice(subset, subset + 10);
                        var startPoint = waypointSubset[0];
                        var midPoints = waypointSubset.slice(1, waypointSubset.length - 1);
                        var endPoint = waypointSubset[waypointSubset.length - 1];
                        calcRoute(startPoint, endPoint, midPoints);
                        subset += 9;
                    }
                }

                """
        Page_2 = """
                createRoutes(optimized_route);
                google.maps.event.addDomListener(window, "load", initialize);
            </script>
          </head>
          <body>
            <div id="map-canvas"></div>
          </body>
        </html>
        """

        with open('optimized-us-capitol-trip-{}-states.html'.format(route_num + 2), 'w') as output_file:
            output_file.write(Page_1)
            output_file.write('optimized_route = {};'.format(str(route)))
            output_file.write(Page_2)

create_individual_road_trip_maps(reversed(hof))


get_ipython().system('open optimized-us-capitol-trip-48-states.html')


# ### Some technical notes
# 
# As I mentioned in the [original article](http://www.randalolson.com/2015/03/08/computing-the-optimal-road-trip-across-the-u-s/), by the end of 5,000 generations, the genetic algorithm will very likely find a *good* but probably not the *absolute best* solution to the optimal routing problem. It is in the nature of genetic algorithms that we never know if we found the absolute best solution.
# 
# However, there exist some brilliant analytical solutions to the optimal routing problem such as the [Concorde TSP solver](http://en.wikipedia.org/wiki/Concorde_TSP_Solver). If you're interested in learning more about Concorde and how it's possible to find a perfect solution to the routing problem, I advise you check out [Nathan Brixius' article](https://nathanbrixius.wordpress.com/2016/06/16/finding-optimal-state-capitol-tours-on-the-cloud-with-neos/) on the topic.
# 
# ### If you have any questions
# 
# Please feel free to:
# 
# * [Email me](http://www.randalolson.com/contact/),
# 
# * [Tweet](https://twitter.com/randal_olson) at me, or
# 
# * comment on the [blog post](http://www.randalolson.com/2016/06/05/computing-optimal-road-trips-on-a-limited-budget/)
# 
# I'm usually pretty good about getting back to people within a day or two.
# 

# # TPOT: A Python tool for automating data science
# 
# ### Notebook by [Randal S. Olson](http://www.randalolson.com/)
# 
# This notebook is a demo for the [TPOT](https://github.com/rhiever/tpot) data science automation tool under development at the [Penn Institute for Biomedical Informatics](http://upibi.org/).
# 
# Below are code samples demonstrating why designing machine learning pipelines is difficult, and how TPOT automates that process.
# 
# ## License
# 
# Please see the [repository README file](https://github.com/rhiever/Data-Analysis-and-Machine-Learning-Projects#license) for the licenses and usage terms for the instructional material and code in this notebook. In general, I have licensed this material so that it is as widely usable and shareable as possible.
# 

# ## Why automate data science?
# 
# ### Model hyperparameter tuning is important
# 

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

mnist_data = pd.read_csv('https://raw.githubusercontent.com/rhiever/Data-Analysis-and-Machine-Learning-Projects/master/tpot-demo/mnist.csv.gz', sep='\t', compression='gzip')
mnist_data.head()


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sb

plt.figure(figsize=(8, 8))

for record_num in range(1, 65):
    plt.subplot(8, 8, record_num)
    
    digit_features = mnist_data.iloc[record_num].drop('class').values
    sb.heatmap(digit_features.reshape((28, 28)),
               cmap='Greys',
               square=True, cbar=False,
               xticklabels=[], yticklabels=[])

plt.tight_layout()
("")


cv_scores = cross_val_score(RandomForestClassifier(n_estimators=10, n_jobs=-1),
                            X=mnist_data.drop('class', axis=1).values,
                            y=mnist_data.loc[:, 'class'].values,
                            cv=10)

print(cv_scores)
print(np.mean(cv_scores))


cv_scores = cross_val_score(RandomForestClassifier(n_estimators=100, n_jobs=-1),
                            X=mnist_data.drop('class', axis=1).values,
                            y=mnist_data.loc[:, 'class'].values,
                            cv=10)

print(cv_scores)
print(np.mean(cv_scores))


# ### Model selection is important
# 

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score

hill_valley_data = pd.read_csv('https://raw.githubusercontent.com/rhiever/Data-Analysis-and-Machine-Learning-Projects/master/tpot-demo/Hill_Valley_without_noise.csv.gz', sep='\t', compression='gzip')
hill_valley_data.head()


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sb

with plt.style.context('seaborn-notebook'):
    plt.figure(figsize=(6, 6))
    for record_num in range(1, 11):
        plt.subplot(10, 1, record_num)
        hv_record_features = hill_valley_data.loc[record_num].drop('class').values
        plt.plot(hv_record_features)
        plt.yticks([])
        plt.xticks([])

    plt.tight_layout()
("")


cv_scores = cross_val_score(RandomForestClassifier(n_estimators=100, n_jobs=-1),
                            X=hill_valley_data.drop('class', axis=1).values,
                            y=hill_valley_data.loc[:, 'class'].values,
                            cv=10)

print(cv_scores)
print(np.mean(cv_scores))


cv_scores = cross_val_score(LogisticRegression(),
                            X=hill_valley_data.drop('class', axis=1).values,
                            y=hill_valley_data.loc[:, 'class'].values,
                            cv=10)

print(cv_scores)
print(np.mean(cv_scores))


# ### Feature preprocessing is important
# 

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import cross_val_score

hill_valley_noisy_data = pd.read_csv('https://raw.githubusercontent.com/rhiever/Data-Analysis-and-Machine-Learning-Projects/master/tpot-demo/Hill_Valley_with_noise.csv.gz', sep='\t', compression='gzip')
hill_valley_noisy_data.head()


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sb

with plt.style.context('seaborn-notebook'):
    plt.figure(figsize=(6, 6))
    for record_num in range(1, 11):
        plt.subplot(10, 1, record_num)
        hv_noisy_record_features = hill_valley_noisy_data.loc[record_num].drop('class').values
        plt.plot(hv_noisy_record_features)
        plt.yticks([])
        plt.xticks([])

    plt.tight_layout()
("")


cv_scores = cross_val_score(RandomForestClassifier(n_estimators=100, n_jobs=-1),
                            X=hill_valley_noisy_data.drop('class', axis=1).values,
                            y=hill_valley_noisy_data.loc[:, 'class'].values,
                            cv=10)

print(cv_scores)
print(np.mean(cv_scores))


cv_scores = cross_val_score(make_pipeline(PCA(n_components=10), RandomForestClassifier(n_estimators=100, n_jobs=-1)),
                            X=hill_valley_noisy_data.drop('class', axis=1).values,
                            y=hill_valley_noisy_data.loc[:, 'class'].values,
                            cv=10)

print(cv_scores)
print(np.mean(cv_scores))


# # Automating data science with TPOT
# 

import pandas as pd
from sklearn.cross_validation import train_test_split
from tpot import TPOTClassifier

hill_valley_noisy_data = pd.read_csv('https://raw.githubusercontent.com/rhiever/Data-Analysis-and-Machine-Learning-Projects/master/tpot-demo/Hill_Valley_with_noise.csv.gz', sep='\t', compression='gzip')
hill_valley_noisy_data.head()


X = hill_valley_noisy_data.drop('class', axis=1).values
y = hill_valley_noisy_data.loc[:, 'class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

my_tpot = TPOTClassifier(generations=10, verbosity=2)
my_tpot.fit(X_train, y_train)
print(my_tpot.score(X_test, y_test))


# # Rethinking the population pyramid
# 
# This notebook provides the methodology and code used in the blog post, [Rethinking the population pyramid](http://www.randalolson.com/2015/07/14/rethinking-the-population-pyramid/).
# 
# ### Notebook by [Randal S. Olson](http://www.randalolson.com/)
# 
# Please see the [repository README file](https://github.com/rhiever/Data-Analysis-and-Machine-Learning-Projects#license) for the licenses and usage terms for the instructional material and code in this notebook. In general, I have licensed this material so that it is widely useable and shareable as possible.
# 

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# This is a custom matplotlib style that I use for most of my charts
plt.style.use('https://gist.githubusercontent.com/rhiever/d0a7332fe0beebfdc3d5/raw/205e477cf231330fe2f265070f7c37982fd3130c/tableau10.mplstyle')

age_gender_data = pd.read_csv('http://www.randalolson.com/wp-content/uploads/us-age-gender-breakdown.csv')
age_gender_data.head()


# #Problems with the population pyramid
# 
# ###1) Violates the [standard expectation](http://mathbench.umd.edu/modules/visualization_graph/page02.htm) of having the causal variable on the x-axis.
# 

fig = plt.figure(figsize=(15, 7))
ax1 = fig.add_subplot(111)

for (i, row) in age_gender_data.iterrows():
    plt.bar([i, i], [row['Females_2010'], -row['Males_2010']],
            color=['#CC6699', '#008AB8'], width=0.8, align='center', edgecolor='none')
    
plt.xlim(-0.6, 20.6)
plt.ylim(-12.1e6, 12.1e6)
#plt.grid(False, axis='x')
plt.xticks(np.arange(0, 21), age_gender_data['Age_Range'], fontsize=11)
plt.yticks(np.arange(-12e6, 13e6, 2e6),
           ['{}m'.format(int(abs(x) / 1e6)) if x != 0 else 0 for x in np.arange(-12e6, 13e6, 2e6)])
plt.xlabel('Age group')
plt.ylabel('Number of people (millions)')

plt.savefig('pop_pyramid_rotated.pdf')
("")


# ###2) Doesn't allow direct comparisons between the two categories.
# 

fig = plt.figure(figsize=(15, 7))
ax1 = fig.add_subplot(111)

for (i, row) in age_gender_data.iterrows():
    plt.bar([i - 0.2, i + 0.2], [row['Females_2010'], row['Males_2010']],
            color=['#CC6699', '#008AB8'], width=0.4, align='center', edgecolor='none')
    
plt.xlim(-0.6, 20.6)
plt.xticks(range(0, 21), age_gender_data['Age_Range'], fontsize=11)
plt.grid(False, axis='x')
plt.yticks(np.arange(0, 13e6, 1e6),
           ['{}m'.format(int(x / 1e6)) if x > 0 else 0 for x in np.arange(0, 13e6, 1e6)])
plt.xlabel('Age group')
plt.ylabel('Number of people (millions)')

plt.savefig('pop_pyramid_grouped.pdf')
("")


# ###3) Relative trends between the categories can be masked by displaying absolute values.
# 

age_gender_data['Male_Pct'] = age_gender_data['Males_2010'] / age_gender_data['Total_Pop_2010']
age_gender_data['Female_Pct'] = age_gender_data['Females_2010'] / age_gender_data['Total_Pop_2010']

plt.figure(figsize=(15, 7))

for (i, row) in age_gender_data.iterrows():
    plt.bar([i], [row['Male_Pct']],
            color=['#008AB8'], width=0.9, align='center', edgecolor='none')
    plt.bar([i], [row['Female_Pct']], bottom=[row['Male_Pct']],
            color=['#CC6699'], width=0.9, align='center', edgecolor='none')
    
plt.xlim(-0.6, 20.6)
plt.ylim(0, 1)
plt.xticks(range(0, 21), age_gender_data['Age_Range'], fontsize=11)
plt.grid(False)
plt.yticks(np.arange(0, 1.01, 0.25),
           ['{}%'.format(int(x * 100)) for x in np.arange(0, 1.01, 0.25)])
plt.xlabel('Age group')
plt.ylabel('Percentage of age group')

plt.plot([-0.425, 20.425], [0.5, 0.5], lw=2, color='black')
plt.plot([-0.425, 20.425], [0.25, 0.25], lw=2, color='black')

plt.savefig('pop_pyramid_stacked.pdf')
("")





# # Computing the optimal path for finding Waldo
# 

# This notebook provides the methodology and code used in the blog post, [Here’s Waldo: Computing the optimal search strategy for finding Waldo](http://www.randalolson.com/2015/02/03/heres-waldo-computing-the-optimal-search-strategy-for-finding-waldo/).
# 
# ###Notebook by [Randal S. Olson](http://www.randalolson.com/)
# 
# Please see the [repository README file](https://github.com/rhiever/Data-Analysis-and-Machine-Learning-Projects#license) for the licenses and usage terms for the instructional material and code in this notebook. In general, I have licensed this material so that it is widely useable and shareable as possible.
# 

# ### Read the data into a pandas DataFrame
# 

# Since we already have all 68 locations of Waldo from [Slate](http://www.slate.com/content/dam/slate/articles/arts/culturebox/2013/11/131111_heresWaldo920_1.jpg.CROP.original-original.jpg), we can jump right into analyzing them. 
# 

from __future__ import print_function
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import random
import math

sb.set_style("white")
plt.style.use("tableau10")

wheres_waldo_locations = pd.read_csv("wheres-waldo-locations.csv")
wheres_waldo_locations.describe()


# ### Plot the dots according to the book they came from
# 

# The first basic visualization that we can make is to plot all of the points according to the book that they came from.
# 
# The dashed line in the center represents the crease of the book, as "Where's Waldo" illustrations always stretched over two pages.
# 

plt.figure(figsize=(12.75, 8))
plt.plot([6.375, 6.375], [0, 8], "--", color="black", alpha=0.4, lw=1.25)

for book, group in wheres_waldo_locations.groupby("Book"):
    plt.plot(group.X, group.Y, "o", label="Book %d" % (book))

plt.xlim(0, 12.75)
plt.ylim(0, 8)
plt.xticks([])
plt.yticks([])
plt.legend(loc="upper center", ncol=7, frameon=True, fancybox=True, bbox_to_anchor=(0.5, 1.1))
("")


# ### Kernel Density Estimation
# 

# Next, we can use the [Seaborn library](http://stanford.edu/~mwaskom/software/seaborn/examples/index.html) to perform a [kernel density estimation](http://en.wikipedia.org/wiki/Kernel_density_estimation) (KDE) of the points.
# 
# A KDE will show us Waldo's hot spots, i.e., where he's most likely to appear.
# 

sb.kdeplot(wheres_waldo_locations.X, wheres_waldo_locations.Y, shade=True, cmap="Blues")
plt.plot([6.375, 6.375], [0, 8], "--", color="black", alpha=0.4, lw=1.25)
plt.xlim(0, 12.75)
plt.ylim(0, 8)
plt.xlabel("")
plt.ylabel("")
plt.xticks([])
plt.yticks([])
("")


# ### Genetic Algorithm
# 

# Now on to the real fun! I decided to approach this problem as a [traveling salesman problem](http://en.wikipedia.org/wiki/Travelling_salesman_problem): We need to check every possible location that Waldo could be at while taking as little time as possible. That means we need to cover as much ground as possible without any backtracking.
# 
# In computer terms, that means we’re making a list of all 68 points that Waldo could be at, then sorting them based on the order that we’re going to visit them. We can use a [genetic algorithm](http://en.wikipedia.org/wiki/Genetic_algorithm) (GA) to try out hundreds of possible arrangements and continually build upon the best ones. Note that because GAs are stochastic, the end result will not always be the same each time you run it.
# 

# #### Reorganize the DataFrame Xs and Ys into a lookup table
# 

# Constantly looking up values in a DataFrame is quite slow, so it's better to use a dictionary.
# 

waldo_location_map = {}

for i, record in wheres_waldo_locations.iterrows():
    key = "B%dP%d" % (record.Book, record.Page)
    waldo_location_map[key] = (record.X, record.Y)


# #### Basic functions for the Genetic Algorithm
# 

def calculate_distance(x1, y1, x2, y2):
    """
        Returns the Euclidean distance between points (x1, y1) and (x2, y2)
    """
    return math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )

def compute_fitness(solution):
    """
        Computes the distance that the Waldo-seeking solution covers.
        
        Lower distance is better, so the GA should try to minimize this function.
    """
    solution_fitness = 0.0
    
    for index in range(1, len(solution)):
        w1 = solution[index]
        w2 = solution[index - 1]
        solution_fitness += calculate_distance(waldo_location_map[w1][0], waldo_location_map[w1][1],
                                               waldo_location_map[w2][0], waldo_location_map[w2][1])
        
    return solution_fitness

def generate_random_agent():
    """
        Creates a random Waldo-seeking path.
    """
    new_random_agent = waldo_location_map.keys()
    random.shuffle(new_random_agent)
    return tuple(new_random_agent)

def mutate_agent(agent_genome, max_mutations=3):
    """
        Applies 1 - `max_mutations` point mutations to the given Waldo-seeking path.
        
        A point mutation swaps the order of two locations in the Waldo-seeking path.
    """
    agent_genome = list(agent_genome)
    num_mutations = random.randint(1, max_mutations)
    
    for mutation in range(num_mutations):
        swap_index1 = random.randint(0, len(agent_genome) - 1)
        swap_index2 = swap_index1

        while swap_index1 == swap_index2:
            swap_index2 = random.randint(0, len(agent_genome) - 1)

        agent_genome[swap_index1], agent_genome[swap_index2] = agent_genome[swap_index2], agent_genome[swap_index1]
            
    return tuple(agent_genome)

def shuffle_mutation(agent_genome):
    """
        Applies a single shuffle mutation to the given Waldo-seeking path.
        
        A shuffle mutation takes a random sub-section of the path and moves it to
        another location in the path.
    """
    agent_genome = list(agent_genome)
    
    start_index = random.randint(0, len(agent_genome) - 1)
    length = random.randint(2, 20)
    
    genome_subset = agent_genome[start_index:start_index + length]
    agent_genome = agent_genome[:start_index] + agent_genome[start_index + length:]
    
    insert_index = random.randint(0, len(agent_genome) + len(genome_subset) - 1)
    agent_genome = agent_genome[:insert_index] + genome_subset + agent_genome[insert_index:]
    
    return tuple(agent_genome)

def generate_random_population(pop_size):
    """
        Generates a list with `pop_size` number of random Waldo-seeking paths.
    """
    random_population = []
    for agent in range(pop_size):
        random_population.append(generate_random_agent())
    return random_population

def plot_trajectory(agent_genome):
    """
        Create a visualization of the given Waldo-seeking path.
    """
    agent_xs = []
    agent_ys = []
    agent_fitness = compute_fitness(agent_genome)

    for waldo_loc in agent_genome:
        agent_xs.append(waldo_location_map[waldo_loc][0])
        agent_ys.append(waldo_location_map[waldo_loc][1])

    plt.figure()
    plt.title("Fitness: %f" % (agent_fitness))
    plt.plot(agent_xs[:18], agent_ys[:18], "-o", markersize=7)
    plt.plot(agent_xs[17:35], agent_ys[17:35], "-o", markersize=7)
    plt.plot(agent_xs[34:52], agent_ys[34:52], "-o", markersize=7)
    plt.plot(agent_xs[51:], agent_ys[51:], "-o", markersize=7)
    plt.plot(agent_xs[0], agent_ys[0], "^", color="#1f77b4", markersize=15)
    plt.plot(agent_xs[-1], agent_ys[-1], "v", color="#d62728", markersize=15)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
def run_genetic_algorithm(generations=10000, population_size=100):
    """
        The core of the Genetic Algorithm.
        
        `generations` and `population_size` must be a multiple of 10.
    """
    
    population_subset_size = int(population_size / 10.)
    generations_10pct = int(generations / 10.)
    
    # Create a random population of `population_size` number of solutions.
    population = generate_random_population(population_size)

    # For `generations` number of repetitions...
    for generation in range(int(generations)):
        
        # Compute the fitness of the entire current population
        population_fitness = {}

        for agent_genome in population:
            if agent_genome in population_fitness:
                continue

            population_fitness[agent_genome] = compute_fitness(agent_genome)

        # Take the top 10% shortest paths and produce offspring from each of them
        new_population = []
        for rank, agent_genome in enumerate(sorted(population_fitness,
                                                   key=population_fitness.get)[:population_subset_size]):

            if (generation % generations_10pct == 0 or generation == (generations - 1)) and rank == 0:
                print("Generation %d best: %f" % (generation, population_fitness[agent_genome]))
                print(agent_genome)
                plot_trajectory(agent_genome)

            # Create 1 exact copy of each top path
            new_population.append(agent_genome)

            # Create 4 offspring with 1-3 mutations
            for offspring in range(4):
                new_population.append(mutate_agent(agent_genome, 3))
                
            # Create 5 offspring with a single shuffle mutation
            for offspring in range(5):
                new_population.append(shuffle_mutation(agent_genome))

        # Replace the old population with the new population of offspring
        for i in range(len(population))[::-1]:
            del population[i]

        population = new_population


run_genetic_algorithm(generations=10000, population_size=100)


# ### Some notes on the methodology for the nerdy folk
# 

# Several people have pointed out that this method is very likely [overfit](http://en.wikipedia.org/wiki/Overfitting), i.e., that the path is trained so much on the 68 points that it likely wouldn't work well on a new 8th book. While this criticism is possibly true, it also misses the purpose of this project.
# 
# **The purpose of this project is to find an optimal search path for the first 7 'Where's Waldo?' books only.**
# 
# If the goal were to produce an optimal search path for any future books, it would become necessary to divide the 68 points into a training and testing set so some form of cross-validation could be performed. However, because the data is already quite sparse (68 points), this would likely prove to be a fruitless endeavor anyway.
# 
# That said, please feel free to continue working on this data set and discover better methods to find Waldo efficiently. I've provided the data set along with this notebook if you would like to work on it with your own methods.
# 
# If you find a better solution, please [contact me](http://www.randalolson.com/contact/) so I can take a look.
# 

# # Percentage of Bachelor’s degrees conferred to women in the U.S., by major
# 
# This notebook provides the methodology and code used in the blog post, [Percentage of Bachelor’s degrees conferred to women in the U.S., by major (1970-2014)](http://www.randalolson.com/2014/06/14/percentage-of-bachelors-degrees-conferred-to-women-by-major-1970-2012/).
# 
# ### Notebook by [Randal S. Olson](http://www.randalolson.com)
# 
# Please see the [repository README file](https://github.com/rhiever/Data-Analysis-and-Machine-Learning-Projects#license) for the licenses and usage terms for the instructional material and code in this notebook. In general, I have licensed this material so that it is as widely useable and shareable as possible.
# 
# ### Required Python libraries
# 
# If you don't have Python on your computer, you can use the [Anaconda Python distribution](http://continuum.io/downloads) to install most of the Python packages you need. Anaconda provides a simple double-click installer for your convenience.
# 
# This code uses base Python libraries except for the `BeautifulSoup`, `pandas`, and `matplotlib` packages. You can install these packages using `pip` by typing the following command into your command line:
# 
# > pip install beautifulsoup4 pandas matplotlib 
# 
# If you're on a Mac, Linux, or Unix machine, you may need to type `sudo` before the command to install the package with administrator privileges.
# 
# ### Scraping the NCES database
# 
# To acquire the data used in the blog post, we need to scrape the [NCES database](http://nces.ed.gov/programs/digest/current_tables.asp). The NCES database is available for download as Excel files, but we didn't want to deal with a bunch of Excel files. Instead, let's scrape the NCES database web pages directly using [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/).
# 
# Running the code below will create a file called `gender_degree_data.tsv` that contains all data about the gender breakdowns of various degree majors in the US.
# 

from bs4 import BeautifulSoup
import requests

with open('gender_degree_data.tsv', 'w') as out_file:

    out_file.write('\t'.join(['Year', 'Degree_Major',
                          'Total_Bachelors',
                          'Percent_change_Bachelors',
                          'Male_Bachelors', 'Female_Bachelors', 'Female_percent_Bachelors',
                          'Total_Masters', 'Male_Masters', 'Female_Masters',
                          'Total_Doctorates', 'Male_Doctorates', 'Female_Doctorates']) + '\n')

    table_list_response = requests.get('http://nces.ed.gov/programs/digest/current_tables.asp')
    table_list_response = BeautifulSoup(table_list_response.text, 'lxml')

    for link in table_list_response.find_all('a', href=True):
        # We only want the tables that stratify the data by degree and gender, which are in table group 325
        if 'dt15_325' in link['href'] and int(link.text.split('.')[1]) % 5 == 0:
            url = 'http://nces.ed.gov/programs/digest/{}'.format(link['href'])
            url_response = requests.get(url)
            url_response = BeautifulSoup(url_response.text, 'lxml')
            degree_major = url_response.find('title').text.split('Degrees in')[1].split('conferred')[0].strip()

            all_trs = url_response.find_all('tr')
            for tr in all_trs:
                # We only want to parse entries that correspond to a certain year
                year_header = tr.find('th')
                if year_header is None:
                    continue

                # Stop parsing after all of the years are listed
                if 'Percent change' in year_header.text:
                    break

                # Years always have a dash (-) in them
                if '-' not in year_header.text:
                    continue

                year = str(int(year_header.text.split('-')[0]) + 1)
                year_vals = [x.text.replace(',', '').replace('†', '0').replace('#', '0') for x in tr.find_all('td')]

                out_text = '\t'.join([year, degree_major] + year_vals) + '\n'
                out_file.write(out_text)


# ### Visualizing the gender breakdowns for the various degree programs
# 
# Next, let's use [matplotlib](http://matplotlib.org) to visualize the trends in the gender breakdowns.
# 

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import pandas as pd

# This is my custom style that does most of the plot formatting
plt.style.use('https://gist.githubusercontent.com/rhiever/a4fb39bfab4b33af0018/raw/b25b4ba478c2e163dd54fd5600e80ca7453459da/tableau20.mplstyle')

degree_gender_data = pd.read_csv('gender_degree_data.tsv', sep='\t')
degree_gender_data = degree_gender_data[degree_gender_data['Year'] >= 1970]
degree_gender_data.set_index('Year', inplace=True)

# Create a list of the degree majors ranked by their last value in the time series
# We'll use this list to determine what colors the degree majors are assigned
degree_major_order = degree_gender_data.groupby('Degree_Major')['Female_percent_Bachelors'].last()
degree_major_order = degree_major_order.sort_values(ascending=False).index.values
degree_major_order_dict = dict(zip(degree_major_order, range(len(degree_major_order))))

degree_gender_data['Degree_Major_Order'] = degree_gender_data[
    'Degree_Major'].apply(lambda major: degree_major_order_dict[major])

degree_gender_data.groupby('Degree_Major_Order')['Female_percent_Bachelors'].plot(figsize=(10, 12))

plt.xlabel('')
plt.yticks(range(0, 91, 10), ['{}%'.format(x) for x in range(0, 91, 10)])

plt.xlim(1969, 2014)
plt.ylim(-1, 90)

plt.title('Percentage of Bachelor\'s degrees conferred to women'
          'in the U.S.A., by major (1970-2014)\n', fontsize=14)
plt.grid(False, axis='x')

degree_major_pcts = dict(degree_gender_data.groupby(
        'Degree_Major')['Female_percent_Bachelors'].last().iteritems())

degree_major_color_map = [x['color'] for x in plt.rcParams['axes.prop_cycle']]

# We use this dictionary to rename the degree majors to shorter names
degree_major_name_map = {
    'the social sciences and history': 'Social Sciences and History',
    'the health professions and related programs': 'Health Professions',
    'visual and performing arts': 'Art and Performance',
    'foreign languages and literatures': 'Foreign Languages',
    'engineering and engineering technologies': 'Engineering',
    'the biological and biomedical sciences': 'Biology',
    'mathematics and statistics': 'Math and Statistics',
    'agriculture and natural resources': 'Agriculture',
    'the physical sciences and science technologies': 'Physical Sciences',
    'communication, journalism, and related '
    'programs and in communications technologies': 'Communications\nand Journalism',
    'public administration and social services': 'Public Administration',
    'psychology': 'Psychology',
    'English language and literature/letters': 'English',
    'computer and information sciences': 'Computer Science',
    'education': 'Education',
    'business': 'Business',
    'architecture and related services': 'Architecture',
}

# We use these offsets to prevent the degree major labels from overlapping
degree_major_offset_map = {
    'foreign languages and literatures': 1.0,
    'English language and literature/letters': -0.5,
    'agriculture and natural resources': 0.5,
    'business': -0.5,
    'architecture and related services': 0.75,
    'mathematics and statistics': -0.75,
    'engineering and engineering technologies': 0.75,
    'computer and information sciences': -0.75,
}

# Draw the degree major labels at the end of the time series lines
for degree_major in degree_major_pcts:
    plt.text(2014.5, degree_major_pcts[degree_major] - 0.5 + degree_major_offset_map.get(degree_major, 0),
             degree_major_name_map[degree_major],
             color=degree_major_color_map[degree_major_order_dict[degree_major]])

plt.text(1967, -9,
         '\nData source: nces.ed.gov/programs/digest/current_tables.asp (Tables 325.*)\n'
         'Author: Randy Olson (randalolson.com / @randal_olson)',
         ha='left', fontsize=10)

plt.savefig('pct-bachelors-degrees-women-usa-1970-2014.png')
("")


