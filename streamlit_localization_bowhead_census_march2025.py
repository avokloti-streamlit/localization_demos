import numpy as np # for most calculations
import pandas as pd # for working with tabular data
from matplotlib import pyplot as plt # for visualization
import streamlit as st
from Localizer import Localizer
import geopandas as gpd
from scipy.interpolate import interpn

st.set_page_config(layout="wide")

## ======================================================================================================================== ##

@st.cache_data
def plotIndividualProbs(hyd_xs, hyd_ys, longitudes, latitudes, toas, limits_xs, limits_ys):
    ## visualize the ToA
    fig, axs = plt.subplots(3, 4, figsize=(16, 12))
    for h1 in range(len(hyd_xs)):
        pcol = axs[h1//4, h1 % 4].pcolormesh(longitudes, latitudes, toas[h1].T, cmap='Blues_r', linewidth=0, rasterized=True, vmin=0, vmax=1)
        
        for h2 in range(len(hyd_xs)):
            axs[h1//4, h1 % 4].scatter(hyd_xs[h2], hyd_ys[h2], c='k', s=15)
            #axs[h1].text(hyd_xs[h2], hyd_ys[h2], h2, color='k', fontsize=15)
        
        axs[h1//4, h1 % 4].set_aspect('equal', 'box')
        axs[h1//4, h1 % 4].set_ylim(limits_ys)
        axs[h1//4, h1 % 4].set_xlim(limits_xs)
    
    fig.tight_layout()
    return fig

@st.cache_data
def plotProbabilityLandscape(unit_xs, unit_ys, probs, study_xs, study_ys, limits_xs, limits_ys):
    fig, axs = plt.subplots(1, 1, figsize=(8, 4))
    if probs is not None:
        pcol = axs.pcolormesh(study_xs, study_ys, probs.T, cmap='magma', linewidth=0, rasterized=True, vmin=0, vmax=1)
    
    axs.scatter(unit_xs[0:7], unit_ys[0:7], c='k', s=15)
    axs.scatter(0, 0, s=60, color='red', marker='*', zorder=8)
    axs.scatter(unit_xs[7:], unit_ys[7:], c='k', marker='x', s=17)

    # draw circles
    t = np.linspace(0, np.pi, 200)
    axs.plot(4 * np.cos(t), 4 * np.sin(t), 'c:')
    axs.plot(4 * np.cos(t) - 8, 4 * np.sin(t), 'c:')
    axs.plot(4 * np.cos(t) + 8, 4 * np.sin(t), 'c:')
    axs.plot(4 * np.cos(t) - 16, 4 * np.sin(t), 'c:')
    axs.plot(4 * np.cos(t) + 16, 4 * np.sin(t), 'c:')
    axs.plot(20 * np.cos(t), 20 * np.sin(t), 'c:')
    
    axs.set_aspect('equal', 'box')
    axs.set_ylim(limits_ys)
    axs.set_xlim(limits_xs)
    axs.set_xlabel('Horizontal Axis (km)')
    axs.set_ylabel('Vertical Axis (km)')
    fig.tight_layout()
    if probs is not None:
        fig.colorbar(pcol)
    return fig

# plot detection function
def plotDetFunction(t, det_function, empirical_sds):
    fig_det_function, axs_det_function = plt.subplots(1, 1, figsize=(7, 5))
    axs_det_function.plot(t, halfNormal(t, empirical_sds['Ice-cover'][80]), 'r-.', label = 'Ice-cover (80dB)')
    axs_det_function.plot(t, halfNormal(t, empirical_sds['Ice-cover'][70]), 'g-.', label = 'Ice-cover (70dB)')
    axs_det_function.plot(t, halfNormal(t, empirical_sds['Ice-cover'][60]), 'b-.', label = 'Ice-cover (60dB)')
    axs_det_function.plot(t, halfNormal(t, empirical_sds['Open-water'][80]), 'r', label = 'Open water (80dB)')
    axs_det_function.plot(t, halfNormal(t, empirical_sds['Open-water'][70]), 'g', label = 'Open water (70dB)')
    axs_det_function.plot(t, halfNormal(t, empirical_sds['Open-water'][60]), 'b', label = 'Open water (60dB)')
    axs_det_function.plot(t, det_function(t), 'k')
    axs_det_function.set_xlabel('Distance (km)')
    axs_det_function.set_ylabel('Probability of Detection')
    axs_det_function.set_title('Detection Function')
    axs_det_function.set_ylim([0, 1])
    axs_det_function.grid(alpha=0.4)
    fig_det_function.legend(loc='center', ncol=2, bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True)
    return fig_det_function


def calculateProbs(_det_function, distances):
    probs = {i: _det_function(distances[i]) for i in range(len(distances))}
    return probs

@st.cache_data
def precalculate(unit_xs, unit_ys, study_xs, study_ys):
    positions = {'X (km)': unit_xs, 'Y (km)': unit_ys}
    positions = pd.DataFrame(positions)

    # get a linear spacing 
    num_units = len(positions)
    num_lon = len(study_xs)
    num_lat = len(study_ys)

    # unwrap the grid
    lon_grid, lat_grid = np.meshgrid(study_xs, study_ys)
    lon_grid = lon_grid.ravel()
    lat_grid = lat_grid.ravel()

    distances = {}

    # calculate time-of-arrival for each hydrophone
    for unit in range(num_units):
        distances[unit] = np.zeros((num_lon, num_lat))
        for i in range(num_lat):
            for j in range(num_lon):
                distances[unit][j, i] = np.sqrt((study_ys[i] - unit_ys[unit])**2 + (study_xs[j] - unit_xs[unit])**2)
                
    return positions, distances

@st.cache_data
def detectionProbabilities(probs):
    num_units = len(probs)
    (num_lon, num_lat) = probs[0].shape

    no_detection = np.ones((num_lon, num_lat))
    detection_at_one_unit = np.zeros((num_lon, num_lat))
    detection_at_two_units = np.zeros((num_lon, num_lat))
    detection_at_three_units = np.zeros((num_lon, num_lat))

    for h1 in range(num_units):
        no_detection = no_detection * (1 - probs[h1])

    for h1 in range(num_units):
        current_product = probs[h1]
        for h2 in range(num_units):
            if (h1 != h2):
                current_product = current_product * (1 - probs[h2])
                
        detection_at_one_unit = detection_at_one_unit + current_product

    for h1 in range(num_units):
        for h2 in range(h1+1, num_units):
            assert (h1 != h2)
            current_product = probs[h1] * probs[h2]
            for h3 in range(num_units):
                if (h3 != h1) and (h3 != h2):
                    current_product = current_product * (1 - probs[h3])
            
            detection_at_two_units = detection_at_two_units + current_product
    
    for h1 in range(num_units):
        for h2 in range(h1+1, num_units):
            for h3 in range(h2+1, num_units):
                assert (h3 != h1) and (h3 != h2) and (h1 != h2)

                current_product = probs[h1] * probs[h2] * probs[h3]
                for h4 in range(num_units):
                    if (h4 != h1) and (h4 != h2) and (h4 != h3):
                        current_product = current_product * (1 - probs[h4])
            
                #print('Final maximum for units %d and %d: %.2f' % (h1, h2, np.max(current_product)))
                detection_at_three_units = detection_at_three_units + current_product
    
    detection_at_three_or_more_units = np.ones((num_lon, num_lat)) - no_detection - detection_at_one_unit - detection_at_two_units
    detection_at_four_or_more_units = np.ones((num_lon, num_lat)) - no_detection - detection_at_one_unit - detection_at_two_units - detection_at_three_units

    return no_detection, detection_at_one_unit, detection_at_two_units, detection_at_three_units, detection_at_three_or_more_units, detection_at_four_or_more_units

def calculateLocalizationErrorWithProbability(localizer, det_function, temporal_error, temporal_sd, num_repeats=10, likelihood_threshold=0.01, minimum_number_of_units=3):
    # save errors here
    errors_rms = np.zeros((localizer.num_xs, localizer.num_ys))
    errors_rms[:] = np.nan
    success_counts = np.zeros((localizer.num_xs, localizer.num_ys))
    nonsuccess_counts = np.zeros((localizer.num_xs, localizer.num_ys))
    unit_counts = np.zeros((localizer.num_xs, localizer.num_ys))
    stats_metric = np.zeros((localizer.num_xs, localizer.num_ys))

    # calculating localization...
    error_progress = st.progress(0, 'Calculating Monte-Carlo simulation...')

    # just use whichever depth
    depth = 0

    for i in range(localizer.num_xs):
        for j in range(localizer.num_ys):
            # store RMS error
            error_sum = 0
            unit_counter = 0

            # repeat calculation ten times
            for r in range(num_repeats):
                # calculate probability of detection
                included_units = []
                for unit in localizer.units:
                    # calculate distance to unit from grid location, and get probability
                    distance_to_unit = localizer.speed_of_sound * localizer.toas_over_depth[depth][unit][i, j]
                    prob = det_function(distance_to_unit)
                    print(prob)

                    # sample with this probability
                    include = np.random.binomial(1, prob)

                    if include:
                        included_units.append(unit)
                
                unit_counter += len(included_units)

                # if the signal is detected on fewer than 3 units, continue
                if len(included_units) < minimum_number_of_units:
                    nonsuccess_counts[i, j] += 1
                    continue

                # first, get the exact measured times
                random_start_time = np.random.rand() * 10
                measured_times = {unit: localizer.toas_over_depth[depth][unit][i, j] + random_start_time + 
                                  np.random.normal(scale = temporal_error) for unit in included_units}

                # calculate pairwise time differences
                measured_tdoas = {}
                for ri in measured_times:
                    for rj in measured_times:
                        measured_tdoas[(ri, rj)] = (measured_times[ri] - measured_times[rj])

                # localize
                _, likelihood_product, max_locations, depth = localizer.localizeAcrossDepths(measured_tdoas, sd_param = temporal_sd)

                #print(len(included_units), np.max(likelihood_product))
                
                # if the location is above threshold, or on the edge of the grid
                if (np.max(likelihood_product) < likelihood_threshold) or (max_locations[0] in limits_xs) or (max_locations[1] in limits_ys):
                    nonsuccess_counts[i, j] += 1
                    continue

                # calculate error and add to RMS sum
                error = (max_locations[0] - localizer.study_xs[i])**2 + (max_locations[1] - localizer.study_ys[j])**2
                #print((localizer.study_xs[i], localizer.study_ys[j]), max_locations, error)

                error_sum += error
                success_counts[i, j] += 1
                stats_metric[i, j] += (localizer.study_ys[j] < 4) == (max_locations[1] < 4)
            
            # save
            errors_rms[i, j] = np.sqrt(error_sum/success_counts[i, j])
            unit_counts[i, j] = unit_counter
            error_progress.progress((i * localizer.num_ys + j)/(localizer.num_xs * localizer.num_ys), 'Calculating Monte-Carlo simulation...')
    
    error_progress.empty()
    return errors_rms, unit_counts/num_repeats, success_counts/num_repeats, nonsuccess_counts/num_repeats, stats_metric/num_repeats

def plotErrorMap(units_xs, units_ys, errors_rms, localizer, limits_xs, limits_ys, vmax, title=None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    pcol = ax.pcolormesh(localizer.study_xs, localizer.study_ys, errors_rms.T, cmap='plasma', linewidth=0, rasterized=True, vmin=0, vmax=vmax)

    for j in localizer.units:
        ax.scatter(units_xs[j], units_ys[j], c='k', s=16)
        ax.text(units_xs[j] + 0.0002, units_ys[j] + 0.0002, j, color='k', fontsize=14, weight='bold')
    
    t = np.linspace(0, np.pi, 200)
    ax.plot(4 * np.cos(t), 4 * np.sin(t), 'c:')
    ax.plot(4 * np.cos(t) - 8, 4 * np.sin(t), 'c:')
    ax.plot(4 * np.cos(t) + 8, 4 * np.sin(t), 'c:')
    ax.plot(4 * np.cos(t) - 16, 4 * np.sin(t), 'c:')
    ax.plot(4 * np.cos(t) + 16, 4 * np.sin(t), 'c:')
    ax.plot(20 * np.cos(t), 20 * np.sin(t), 'c:')
    ax.set_aspect('equal', 'box')
    ax.set_ylim(limits_ys)
    ax.set_xlim(limits_xs)
    ax.set_title(title)
    fig.tight_layout()
    fig.colorbar(pcol)
    return fig

def halfNormal(x, sd):
    return (np.exp(-x**(2) / (2 * sd**2)))

class FloatReader:
    def __init__(self, filename):
        self.f = open(filename, "rb")
    
    def read_floats(self, count : int):
        return np.fromfile(self.f, dtype=np.float32, count=count, sep='')


def rotateArray(x, y, theta):
    theta = theta * np.pi/180
    rotation = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    output = rotation @ np.array([x, y])
    return output[0], output[1]

@st.cache_data
def loadMapData():
    bathymetry = np.loadtxt('bathymetry.csv')
    bathymetry = np.flipud(bathymetry)
    bathymetry[bathymetry < -1000] = np.nan

    xllcorner = 554458.24478533
    yllcorner = 7893428.5169774
    cellsize = 10
    ncols = 5236
    nrows = 4181

    lons = np.linspace(xllcorner, xllcorner + cellsize * (ncols - 1), ncols//3)/1000
    lats = np.linspace(yllcorner, yllcorner + cellsize * (nrows - 1), nrows//3)/1000

    coast = gpd.read_file('individual_files_bb1_coast.shp')
    crs_in_km = '+proj=utm +zone=4 +datum=WGS84 +units=km +no_defs +type=crs'
    coast = coast.to_crs(crs_in_km)

    perch_locations = pd.read_csv('Perch_Locations_Raw.csv')
    perch_locations = perch_locations.drop(14)
    perch_locations_gpd = gpd.points_from_xy(perch_locations['Lon'], perch_locations['Lat'], crs="EPSG:4326")
    perch_locations_metric = perch_locations_gpd.to_crs(coast.crs)

    special_locations_gpd = gpd.points_from_xy([-156.676948, -156.789912], [71.32723, 71.290350], crs="EPSG:4326")
    special_locations_metric = special_locations_gpd.to_crs(coast.crs)
    special_locations_names = ['NARL', 'Utqiagvik']

    return bathymetry, lons, lats, coast, perch_locations_metric, special_locations_metric, special_locations_names


def drawMap(bathymetry, lons, lats, coast, perch_lon, perch_lat, special_locations_metric, special_locations_names, x_array, y_array):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    contours = plt.contourf(lons, lats, np.flipud(bathymetry), levels = np.arange(-140, 9, 10), cmap='Blues', zorder=2)
    
    plt.contour(lons, lats, np.flipud(bathymetry), levels = [-75], cmap='cool', zorder=3)
    plt.colorbar(contours)
    plt.scatter(perch_lon, perch_lat, s=60, color='red', marker='*', zorder=8)

    for i, point in enumerate(special_locations_metric):
        plt.scatter(point.x, point.y, s=20, color='b', marker='D', zorder=10)
        plt.text(point.x, point.y - 1.2, special_locations_names[i], color='b', fontsize=9, zorder=10)
    
    plt.scatter(x_array[0:7], y_array[0:7], s=15, c='k', zorder=10)
    plt.scatter(x_array[7:], y_array[7:], s=17, c='k', zorder=10, marker='x')

    coast.plot(ax=ax, color = 'k', zorder=4)
    ax.set_xlabel('Projected Longitude (km)')
    ax.set_ylabel('Projected Latitude (km)')
    return fig


## ======================================================================================================================== ##

st.title('Bowhead Census - Revisiting Array Design')

empirical_sds = {'Ice-cover': {60: 17, 70: 12, 80: 4.5},
                 'Open-water': {60: 90, 70: 54, 80: 28}}

array_choice = 'Current Configuration'
array_spacing = 6.5
array_offset = 4.0

grid_resolution = 60
grid_limits = 30

x_center = 580.68
y_center = 7921.20
rotation = -28

# prepare map information
bathymetry, lons, lats, coast, perch_locations_metric, special_locations_metric, special_locations_names = loadMapData()

geometries = {'Current Configuration': lambda d, z: (np.array([-(1 + np.sqrt(3)/2) * d, -d, -d, 0, d, d, (1 + np.sqrt(3)/2) * d]), np.array([z + d/2, z, d + z, z, z, d + z, z + d/2]))}
unit_xs = geometries[array_choice](array_spacing, array_offset)[0]
unit_ys = geometries[array_choice](array_spacing, array_offset)[1]

limits_xs = [-grid_limits, grid_limits]
limits_ys = [0, grid_limits]
study_xs = np.linspace(limits_xs[0], limits_xs[1], grid_resolution)
study_ys = np.linspace(limits_ys[0], limits_ys[1], grid_resolution)

with st.sidebar:
    st.subheader('Detection Function Parameters')

    st.markdown('The detection functions below are estimated by reference to *"The influence of sea ice on the detection of bowhead whale calls"* by Jones, Joshua M., et al, published in Scientific Reports (2022).')

    st.markdown('Please use the slider below to set the **standard deviation** of the half-normal detection function indicated by the black curve.')

    t = np.linspace(0, 40.0, 1000)
    sd_parameter = st.slider('SD:', value=10.0, min_value=0.0, max_value = 60.0, step=0.5, format="%.1f")
    det_function = lambda x: halfNormal(x, sd_parameter)

    fig_det_function = plotDetFunction(t, det_function, empirical_sds)
    st.pyplot(fig_det_function)


## ======================================================================================================================== ##

st.subheader('Specify location for the perch:')

c1, c2 = st.columns(2)
c1.write('Longitude (degrees):')
c2.write('Latitude (degrees):')
perch_lon = c1.number_input('Longitude:', min_value=-158.0, max_value=-156.0, value=-156.5930371473, step=0.001, format="%.5f", label_visibility='collapsed')
perch_lat = c2.number_input('Latitude:', min_value=71.0, max_value=72.0, value=71.39566096, step=0.001, format="%.5f", label_visibility='collapsed')

# convert to metric
perch_gpd = gpd.points_from_xy([perch_lon], [perch_lat], crs="EPSG:4326")
perch_metric = perch_gpd.to_crs(coast.crs)

## ======================================================================================================================== ##

st.subheader('Specify location for the new units:')

c1, c2, c3 = st.columns(3)
c2.write('Longitude (degrees):')
c3.write('Latitude (degrees):')

new_lons = []
new_lats = []
for row in range(3):
    c1, c2, c3 = st.columns(3)

    new_unit = c1.toggle('Add additional unit?', value=False, key='new_unit_' + str(row))#, label_visibility='collapsed')
    lon_new = c2.number_input('Longitude:', min_value=-158.0, max_value=-156.0, value=-156.86470, step=0.001, format="%.5f", key='x_new_' + str(row), label_visibility='collapsed', disabled=not new_unit)
    lat_new = c3.number_input('Latitude:', min_value=71.0, max_value=72.0, value=71.46390, step=0.001, format="%.5f", key='y_new_' + str(row), label_visibility='collapsed', disabled=not new_unit)

    if new_unit:
        new_lons.append(lon_new)
        new_lats.append(lat_new)

if len(new_lons) > 0:
    # need to convert from lat/lon to metric units
    new_locations_gpd = gpd.points_from_xy(new_lons, new_lats, crs="EPSG:4326")
    new_locations_metric = new_locations_gpd.to_crs(coast.crs)

    # next, subtract perch and rotate back
    for i in range(len(new_lons)):
        new_loc = [new_locations_metric[i].x, new_locations_metric[i].y]
        new_loc = [new_loc[0] - x_center, new_loc[1] - y_center]
        unit_xs_new, unit_ys_new = rotateArray(new_loc[0], new_loc[1], -rotation)

        # add to list of units
        unit_xs = np.append(unit_xs, unit_xs_new)
        unit_ys = np.append(unit_ys, unit_ys_new)


# translate and rotate array
x_array, y_array = rotateArray(unit_xs, unit_ys, rotation)
x_array += x_center
y_array += y_center


## ======================================================================================================================== ##

st.subheader('Map Visualization:')

st.markdown('This map shows the prospective array overlaid on the [bathymetry](https://arcticdata.io/catalog/view/doi:10.5065/D6QZ2822) \
            of Point Barrow. The orange stars show prior perch locations with the mean location represented by the large red star, and \
            the bright cyan curve indicates the 75m isoline.')


# plot map
fig_map = drawMap(bathymetry, lons, lats, coast, perch_metric[0].x, perch_metric[0].y, special_locations_metric, special_locations_names, x_array, y_array)
st.pyplot(fig_map)


## ======================================================================================================================== ##

st.subheader('Metric Visualization:')

perch_x, perch_y = rotateArray(perch_metric[0].x - x_center, perch_metric[0].y - y_center, -rotation)

# convert perch to basic metric coordinates
unit_xs = [x - perch_x for x in unit_xs]
unit_ys = [y - perch_y for y in unit_ys]

positions, distances = precalculate(unit_xs, unit_ys, study_xs, study_ys)

fig_det_at_four_plus_units = plotProbabilityLandscape(unit_xs, unit_ys, None, study_xs, study_ys, limits_xs, limits_ys)
c0, c1, c2 = st.columns((1, 6, 1))
c1.pyplot(fig_det_at_four_plus_units)

# afterwards, calculate probability landscapes and plot
# positions, distances = precalculate(unit_xs, unit_ys, study_xs, study_ys)

# fig_det_at_four_plus_units = plotProbabilityLandscape(unit_xs, unit_ys, None, study_xs, study_ys, limits_xs, limits_ys)
# c0, c1, c2 = st.columns((1, 6, 1))
# c1.pyplot(fig_det_at_four_plus_units)


## ======================================================================================================================== ##

st.subheader('Coordinates:')

proposed_array_metric = gpd.points_from_xy(x_array, y_array, crs=coast.crs)
proposed_array_epsg = proposed_array_metric.to_crs("EPSG:4326")

num_units = len(unit_xs)
depths = np.zeros(num_units)
proposed_longitudes = np.zeros(num_units)
proposed_latitudes = np.zeros(num_units)

for i in range(num_units):
    proposed_longitudes[i] = proposed_array_epsg[i].x
    proposed_latitudes[i] = proposed_array_epsg[i].y
    try:
        depths[i] = np.round(interpn((lons, lats), np.flipud(bathymetry).T, (x_array[i], y_array[i])), 1)
    except:
        depths[i] = np.nan

c1, c2 = st.columns(2)
array_current = pd.DataFrame({'Longitude': proposed_longitudes[0:7], 'Latitude': proposed_latitudes[0:7], 'Depth': depths[0:7]})
array_new = pd.DataFrame({'Longitude': proposed_longitudes[7:], 'Latitude': proposed_latitudes[7:], 'Depth': depths[7:]})

c1.write('Currently deployed units:')
c1.dataframe(array_current)

c2.write('Prospective new units:')
c2.dataframe(array_new)


## ======================================================================================================================== ##

st.subheader('Probability of Detection Visualization:')

# calculate probabilities
probs = calculateProbs(det_function, distances)

# calculate all detection probabilities
no_detection, detection_at_one_unit, detection_at_two_units, detection_at_three_units, detection_at_three_or_more_units, detection_at_four_or_more_units = detectionProbabilities(probs)

with st.expander('Probability that a call is detected at three or more units:'):
    fig_det_at_three_plus_units = plotProbabilityLandscape(unit_xs, unit_ys, detection_at_three_or_more_units, study_xs, study_ys, limits_xs, limits_ys)
    c0, c1, c2 = st.columns((1, 6, 1))
    c1.pyplot(fig_det_at_three_plus_units)

with st.expander('Probability that a call is detected at four or more units:'):
    fig_det_at_four_plus_units = plotProbabilityLandscape(unit_xs, unit_ys, detection_at_four_or_more_units, study_xs, study_ys, limits_xs, limits_ys)
    c0, c1, c2 = st.columns((1, 6, 1))
    c1.pyplot(fig_det_at_four_plus_units)

# calculate time-difference-of-arrival for each hydrophone pair
with st.expander('Probability of detection for each unit individually:'):
    fig_toa = plotIndividualProbs(unit_xs, unit_ys, study_xs, study_ys, probs, limits_xs, limits_ys)
    st.pyplot(fig_toa)

with st.expander('Probability that a call is not detected at any units:'):
    fig_no_detection = plotProbabilityLandscape(unit_xs, unit_ys, no_detection, study_xs, study_ys, limits_xs, limits_ys)
    c0, c1, c2 = st.columns((1, 6, 1))
    c1.pyplot(fig_no_detection)

with st.expander('Probability that a call is detected at exactly one unit:'):
    fig_det_at_one_unit = plotProbabilityLandscape(unit_xs, unit_ys, detection_at_one_unit, study_xs, study_ys, limits_xs, limits_ys)
    c0, c1, c2 = st.columns((1, 6, 1))
    c1.pyplot(fig_det_at_one_unit)

with st.expander('Probability that a call is detected at exactly two units:'):
    fig_det_at_two_units = plotProbabilityLandscape(unit_xs, unit_ys, detection_at_two_units, study_xs, study_ys, limits_xs, limits_ys)
    c0, c1, c2 = st.columns((1, 6, 1))
    c1.pyplot(fig_det_at_two_units)

with st.expander('Probability that a call is detected at exactly three units:'):
    fig_det_at_three_units = plotProbabilityLandscape(unit_xs, unit_ys, detection_at_three_units, study_xs, study_ys, limits_xs, limits_ys)
    c0, c1, c2 = st.columns((1, 6, 1))
    c1.pyplot(fig_det_at_three_units)


## ======================================================================================================================== ##

st.subheader('Evaluating Localization Performance:')

st.markdown('We apply the likelihood-surface localization algorithm, as described in [*Methods for tracking multiple marine mammals \
            with wide-baseline passive acoustic arrays* (Nosal 2013)](https://pubs.aip.org/asa/jasa/article-abstract/134/3/2383/811358/Methods-for-tracking-multiple-marine-mammals-with).\
            This is a computational time-difference-of-arrival-based algorithm which relies on intersecting smoothed hyperbolic surfaces, \
            and can be flexibly adapted for any array geometry and for nonlinear propagation models. The **temporal measurement error** value \
            below represents the anticipated standard deviation of the actual temporal error present in the data. The **temporal SD parameter** \
            is an algorithmic parameter representing the degree of smoothness (and uncertainty) in the hyperbolic surfaces. Next, the \
            **Monte-Carlo runs** value is the number of repeated samples run per each grid point for subsequent calculations. Lastly, the\
            **minimum units for localizing** parameter determines whether detection at 3 units would be used for localization; increasing\
            this parameter to 4 can be expected to increase localization accuracy and decrease spatial coverage.' )

with st.form("Error parameters"):
    c1, c2, c3, c4 = st.columns(4)
    temporal_error = c1.number_input('Temporal measurement error:', min_value=0.0, max_value=5.0, value=0.01)
    temporal_sd = c2.number_input('Temporal SD parameter:', min_value=0.0, max_value=5.0, value=0.1)
    num_repeats = c3.number_input('Monte-Carlo runs:', min_value=0, max_value=20, step=1, value=5)
    min_units = c4.radio('Minimum units for localizing:', [3, 4])
    submitted = st.form_submit_button("Calculate!")

# next, set a temporal error spatial deviation
if submitted:
    localizer = Localizer(unit_xs, unit_ys, study_xs, study_ys, speed_of_sound = 1.5, source_depths = [0])
    errors_rms, unit_counts, success_counts, nonsuccess_counts, stats_metric = calculateLocalizationErrorWithProbability(localizer,
                                                                                                           det_function,
                                                                                                           temporal_error,
                                                                                                           temporal_sd,
                                                                                                           num_repeats,
                                                                                                           minimum_number_of_units=min_units)

    fig_rms = plotErrorMap(unit_xs, unit_ys, errors_rms, localizer, limits_xs, limits_ys, vmax=2, title='Root-Mean-Square Localization Error (km)')
    c0, c1, c2 = st.columns((1, 6, 1))
    c1.pyplot(fig_rms)

    fig_successes = plotErrorMap(unit_xs, unit_ys, success_counts, localizer, limits_xs, limits_ys, vmax=1, title='Proportion of Feasible Localizations')
    c0, c1, c2 = st.columns((1, 6, 1))
    c1.pyplot(fig_successes)

    fig_units = plotErrorMap(unit_xs, unit_ys, unit_counts, localizer, limits_xs, limits_ys, vmax=8, title='Mean Number of Units Detecting Vocalization')
    c0, c1, c2 = st.columns((1, 6, 1))
    c1.pyplot(fig_units)

    fig_weights = plotErrorMap(unit_xs, unit_ys, stats_metric/success_counts, localizer, limits_xs, limits_ys, vmax=1, title='Of the localizations, what proportion correctly identifies if y < or > 4km?')
    c0, c1, c2 = st.columns((1, 6, 1))
    c1.pyplot(fig_weights)