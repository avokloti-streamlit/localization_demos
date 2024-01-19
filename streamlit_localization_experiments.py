import numpy as np # for most calculations
import pandas as pd # for working with tabular data
from matplotlib import pyplot as plt # for visualization
import geopy.distance # for calculating geographic distances
import streamlit as st

st.set_page_config(layout="wide")


@st.cache_data
def plotIndividualProbs(hyd_lons, hyd_lats, longitudes, latitudes, lon_limits, lat_limits, toas):
    ## visualize the ToA
    fig, axs = plt.subplots(1, len(hyd_lons), figsize=(25, 5))
    for h1 in range(len(hyd_lons)):
        pcol = axs[h1].pcolormesh(longitudes, latitudes, toas[h1].T, cmap='Blues_r', linewidth=0, rasterized=True, vmin=0, vmax=1)
        
        for h2 in range(len(hyd_lons)):
            axs[h1].scatter(hyd_lons[h2], hyd_lats[h2], c='k', s=15)
            axs[h1].text(hyd_lons[h2], hyd_lats[h2], h2, color='k', fontsize=15)
        
        axs[h1].set_xlim(lon_limits)
        axs[h1].set_ylim(lat_limits)
        #plt.colorbar(pcol)
    
    fig.tight_layout()
    return fig

# def plotProbabilityLandscape(positions, probs):
#     fig, axs = plt.subplots(1, 3, figsize=(21, 5))
#     pcol = axs[1].pcolormesh(longitudes, latitudes, probs.T, cmap='Blues_r', linewidth=0, rasterized=True, vmin=0, vmax=1)

#     for h2 in range(len(positions)):
#         axs[1].scatter(positions[h2, 'Longitude'], positions[h2, 0], c='k', s=15)
#         axs[1].text(positions[h2, 1], positions[h2, 0], h2 + 1, color='k', fontsize=15)

#     axs[1].set_xlim(lon_limits)
#     axs[1].set_ylim(lat_limits)
#     axs[0].axis('off')
#     axs[2].axis('off')
#     fig.colorbar(pcol)
#     return fig

def plotProbabilityLandscapeSingle(lons, lats, probs):
    fig, axs = plt.subplots(1, 1, figsize=(7, 5))
    pcol = axs.pcolormesh(longitudes, latitudes, probs.T, cmap='Blues_r', linewidth=0, rasterized=True, vmin=0, vmax=1)

    for h2 in range(len(lons)):
        axs.scatter(lons[h2], lats[h2], c='k', s=15)
        axs.text(lons[h2], lats[h2], h2 + 1, color='k', fontsize=15)

    axs.set_xlim(lon_limits)
    axs.set_ylim(lat_limits)
    fig.colorbar(pcol)
    return fig



## ---- define all coordinates / parameters ---- ##

st.header('Visualizing the probability of detection across an array ~')

st.text('Sample lat/lon coordinates:')

lats = np.array([41.933717, 41.9466, 41.8413, 41.839717, 41.8949])
lons = np.array([-70.2286, -70.40015, -70.469317, -70.3148, -70.359167])

positions = {'Latitude': lats, 'Longitude': lons}
positions = pd.DataFrame(positions)

num_units = len(positions)
c0, c1, c2 = st.columns((4, 4, 2))
c1.dataframe(positions)


## ---- define all coordinates / parameters ---- ##

# this is the number of longitude and latitude points in our discretized grid
# (the numbers are chosen to get increments of 0.05 degrees in lon/lat)
num_lon = 150
num_lat = 150

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


st.text('Sample array geometry:')
fig_array = plotProbabilityLandscapeSingle(lons, lats, np.ones((num_lon, num_lat)))

print(lons)
print(lats)

# make two columns
c0, c1, c2 = st.columns((3, 4, 3))
c1.pyplot(fig_array)


## ---- detection function ---- ##

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


# plot detection function
c0, c1, c2 = st.columns((3, 4, 3))
fig_det_function, axs_det_function = plt.subplots(1, 1, figsize=(7, 5))
axs_det_function.plot(t, det_function(t))
axs_det_function.set_xlabel('Distance')
axs_det_function.set_ylabel('Probability of Detection')
axs_det_function.set_title('Detection Function')
c1.pyplot(fig_det_function)

probs = {i: det_function(distances[i]) for i in range(num_units)}

# calculate time-difference-of-arrival for each hydrophone pair
st.text('Probability of detection for each unit individually:')
fig_toa = plotIndividualProbs(lons, lats, longitudes, latitudes, lon_limits, lat_limits, probs)
st.pyplot(fig_toa)


## ---- zero units ---- ##

st.text('Probability that a call is not detected at any units:')

no_detection = np.ones((num_lon, num_lat))
for h1 in range(num_units):
    no_detection = no_detection * (1 - probs[h1])

fig_no_detection = plotProbabilityLandscapeSingle(lons, lats, no_detection)
c0, c1, c2 = st.columns((3, 4, 3))
c1.pyplot(fig_no_detection)


## ---- one unit ---- ##

st.text('Probability that a call is detected at exactly one unit:')

# need to do something combinatorial
# probs[0][i, j] * (1 - probs[1][i, j]) * (1 - probs[2][i, j]) * (1 - probs[3][i, j]) * (1 - probs[4][i, j]) + ...

detection_at_one_unit = np.zeros((num_lon, num_lat))
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

fig_det_at_one_unit = plotProbabilityLandscapeSingle(lons, lats, detection_at_one_unit)
c0, c1, c2 = st.columns((3, 4, 3))
c1.pyplot(fig_det_at_one_unit)


## ---- two units ---- ##

st.text('Probability that a call is detected at exactly two units:')

# need to do something combinatorial
# probs[0][i, j] * probs[1][i, j]  * (1 - probs[2][i, j]) * (1 - probs[3][i, j]) * (1 - probs[4][i, j])

detection_at_two_units = np.zeros((num_lon, num_lat))
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

fig_det_at_two_units = plotProbabilityLandscapeSingle(lons, lats, detection_at_two_units)
c0, c1, c2 = st.columns((3, 4, 3))
c1.pyplot(fig_det_at_two_units)



st.text('Probability that a call is detected at three or more units:')

detection_at_three_or_more_units = np.ones((num_lon, num_lat)) - no_detection - detection_at_one_unit - detection_at_two_units

fig_det_at_three_plus_units = plotProbabilityLandscapeSingle(lons, lats, detection_at_three_or_more_units)
c0, c1, c2 = st.columns((3, 4, 3))
c1.pyplot(fig_det_at_three_plus_units)

# LOCALIZATION DEMO

# let user input array positions


# let user arameterize a detection function

# draw probability of detection around each unit in the array

# draw probability of detection at at least 3 units


# now for the other question... 


# next steps...
# add a hazard rate function (two parameters)
# clean up the plots a bit

# next: look at the inverse problem
# given a detection function, set a triangular spacing of units (calculate locations)
# set distance between them and let the user change that
# then, calculate the total probability across space, 
# in essence: what is the probability that a randomly chosen point in this region will be localizeable
# also write this down in equation form

# --- parameters: ---

# spacing of array
# spacing of the region we are measuring
# integration discretization
# the rest is just calculation?
# detection function standard deviation parameter

# then iterate over these parameters and calculate overall 2D integral as a function of array spacing (x-axis) and of SD parameter
# repeat this for small region and for big region