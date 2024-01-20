import numpy as np # for most calculations
import pandas as pd # for working with tabular data
from matplotlib import pyplot as plt # for visualization
import geopy.distance # for calculating geographic distances
import streamlit as st

st.set_page_config(layout="wide")


@st.cache_data
def plotIndividualProbs(hyd_lons, hyd_lats, longitudes, latitudes, toas):
    ## visualize the ToA
    fig, axs = plt.subplots(1, len(hyd_lons), figsize=(25, 5))
    for h1 in range(len(hyd_lons)):
        pcol = axs[h1].pcolormesh(longitudes, latitudes, toas[h1].T, cmap='Blues_r', linewidth=0, rasterized=True, vmin=0, vmax=1)
        
        for h2 in range(len(hyd_lons)):
            axs[h1].scatter(hyd_lons[h2], hyd_lats[h2], c='k', s=15)
            axs[h1].text(hyd_lons[h2], hyd_lats[h2], h2, color='k', fontsize=15)
    
    fig.tight_layout()
    return fig

@st.cache_data
def plotProbabilityLandscape(lons, lats, probs):
    fig, axs = plt.subplots(1, 1, figsize=(7, 5))
    pcol = axs.pcolormesh(longitudes, latitudes, probs.T, cmap='Blues_r', linewidth=0, rasterized=True, vmin=0, vmax=1)

    for h2 in range(len(lons)):
        axs.scatter(lons[h2], lats[h2], c='k', s=15)
        axs.text(lons[h2], lats[h2], h2 + 1, color='k', fontsize=15)
    
    fig.colorbar(pcol)
    return fig

# plot detection function
def plotDetFunction(t, det_function):
    fig_det_function, axs_det_function = plt.subplots(1, 1, figsize=(7, 5))
    axs_det_function.plot(t, det_function(t))
    axs_det_function.set_xlabel('Distance')
    axs_det_function.set_ylabel('Probability of Detection')
    axs_det_function.set_title('Detection Function')
    return fig_det_function


def calculateProbs(_det_function, distances):
    probs = {i: _det_function(distances[i]) for i in range(len(distances))}
    return probs

@st.cache_data
def precalculate():
    lats = np.array([41.933717, 41.9466, 41.8413, 41.839717, 41.8949])
    lons = np.array([-70.2286, -70.40015, -70.469317, -70.3148, -70.359167])

    positions = {'Latitude': lats, 'Longitude': lons}
    positions = pd.DataFrame(positions)

    num_lon = 150
    num_lat = 150
    num_units = len(positions)

    center_lon = np.mean(lons)
    center_lat = np.mean(lats)

    # these are the max/min limits for longitude and latitude
    lon_limits = [center_lon + 2 * (np.min(lons) - center_lon), center_lon + 2 * (np.max(lons) - center_lon)]
    lat_limits = [center_lat + 2 * (np.min(lats) - center_lat), center_lat + 2 * (np.max(lats) - center_lat)]

    # define somme parameters 
    speed_of_sound = 1.484 # estimated average speed of sound for Cape Cod Bay based on salinity/temperature
        
    # get a linear spacing 
    latitudes = np.linspace(lat_limits[0], lat_limits[1], num_lat)
    longitudes = np.linspace(lon_limits[0], lon_limits[1], num_lon)

    # unwrap the grid
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    lon_grid = lon_grid.ravel()
    lat_grid = lat_grid.ravel()

    distances = {}

    # calculate time-of-arrival for each hydrophone
    for hyd in range(num_units):
        distances[hyd] = np.zeros((num_lon, num_lat))
        for i in range(num_lat):
            for j in range(num_lon):
                distances[hyd][j, i] = geopy.distance.distance((latitudes[i], longitudes[j]),
                                                        (lats[hyd], lons[hyd])).km
                
    return positions, lats, lons, latitudes, longitudes, distances


@st.cache_data
def detectionProbabilities(probs):
    num_units = len(probs)
    (num_lon, num_lat) = probs[0].shape

    no_detection = np.ones((num_lon, num_lat))
    detection_at_one_unit = np.zeros((num_lon, num_lat))
    detection_at_two_units = np.zeros((num_lon, num_lat))

    for h1 in range(num_units):
        no_detection = no_detection * (1 - probs[h1])

    for h1 in range(num_units):
        current_product = probs[h1]
        print('----')
        print(h1)
        for h2 in range(num_units):
            if (h1 != h2):
                current_product = current_product * (1 - probs[h2])
                print(h2)
            
        print('Maximum: %.2f' % np.max(current_product))
        detection_at_one_unit = detection_at_one_unit + current_product

    for h1 in range(num_units):
        for h2 in range(num_units):
            if (h1 != h2):
                current_product = probs[h1] * probs[h2]
                print('prob: %d and %d. Current max: %.2f' % (h1, h2, np.max(current_product)))
                for h3 in range(num_units):
                    if (h3 != h1) and (h3 != h2):
                        current_product = current_product * (1 - probs[h3])
                        print('1 - prob: %d. Current max: %.2f' % (h3, np.max(current_product)))
            
        print('Final maximum: %.2f' % np.max(current_product))
        detection_at_two_units = detection_at_two_units + current_product
    
    detection_at_three_or_more_units = np.ones((num_lon, num_lat)) - no_detection - detection_at_one_unit - detection_at_two_units

    return no_detection, detection_at_one_unit, detection_at_two_units, detection_at_three_or_more_units


## ---- define all coordinates / parameters ---- ##

positions, lats, lons, latitudes, longitudes, distances = precalculate()

st.header('Visualizing the probability of detection across an array ~')

st.text('Sample lat/lon coordinates:')

c0, c1, c2 = st.columns((4, 4, 2))
c1.dataframe(positions)


## ---- define all coordinates / parameters ---- ##

st.text('Sample array geometry:')
fig_array = plotProbabilityLandscape(lons, lats, np.ones((len(longitudes), len(latitudes))))

# make two columns
c0, c1, c2 = st.columns((3, 4, 3))
c1.pyplot(fig_array)


## ---- user input for detection function ---- ##

max_distance = np.max(distances[0])

function_type = st.radio('Choose a detection function:', ['Half-normal', 'Hazard-rate'])

t = np.linspace(0, max_distance, 1000)

if (function_type == 'Half-normal'):
    sd_parameter = st.slider('SD:', value=5.0, min_value=0.0, max_value = max_distance, step=0.01, format="%.3f")
    det_function = lambda x: (np.exp(-x**(2) / (2 * sd_parameter**2)))
else:
    c1, c2 = st.columns(2)
    beta_parameter = c1.slider('Exponent (beta):', value=1.0, min_value=0.0, max_value = 10.0, step=0.01, format="%.3f")
    sd_parameter = c2.slider('SD:', value=5.0, min_value=0.0, max_value = max_distance, step=0.01, format="%.3f")
    det_function = lambda x: (1 - np.exp(-(x / sd_parameter)**(-beta_parameter)))


## ---- plots and calculations ---- ##

# plot detection function
fig_det_function = plotDetFunction(t, det_function)
c0, c1, c2 = st.columns((3, 4, 3))
c1.pyplot(fig_det_function)

# calculate probabilities
probs = calculateProbs(det_function, distances)

# calculate time-difference-of-arrival for each hydrophone pair
st.text('Probability of detection for each unit individually:')
fig_toa = plotIndividualProbs(lons, lats, longitudes, latitudes, probs)
st.pyplot(fig_toa)

# calculate all detection probabilities
no_detection, detection_at_one_unit, detection_at_two_units, detection_at_three_or_more_units = detectionProbabilities(probs)

st.text('Probability that a call is not detected at any units:')

fig_no_detection = plotProbabilityLandscape(lons, lats, no_detection)
c0, c1, c2 = st.columns((3, 4, 3))
c1.pyplot(fig_no_detection)

st.text('Probability that a call is detected at exactly one unit:')

fig_det_at_one_unit = plotProbabilityLandscape(lons, lats, detection_at_one_unit)
c0, c1, c2 = st.columns((3, 4, 3))
c1.pyplot(fig_det_at_one_unit)

st.text('Probability that a call is detected at exactly two units:')

fig_det_at_two_units = plotProbabilityLandscape(lons, lats, detection_at_two_units)
c0, c1, c2 = st.columns((3, 4, 3))
c1.pyplot(fig_det_at_two_units)

st.text('Probability that a call is detected at three or more units:')

fig_det_at_three_plus_units = plotProbabilityLandscape(lons, lats, detection_at_three_or_more_units)
c0, c1, c2 = st.columns((3, 4, 3))
c1.pyplot(fig_det_at_three_plus_units)
