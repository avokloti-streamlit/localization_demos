import numpy as np # for most calculations
import pandas as pd # for working with tabular data
from matplotlib import pyplot as plt # for visualization
import streamlit as st
from Localizer import Localizer
import geopandas as gpd
from scipy.interpolate import interpn

st.set_page_config(layout="wide")


@st.cache_data
def plotIndividualProbs(hyd_xs, hyd_ys, longitudes, latitudes, toas, limits_xs, limits_ys):
    ## visualize the ToA
    if (len(hyd_xs) > 5):
        num_plots, plot_indices = 5, np.random.choice(np.arange(len(hyd_xs)), 5, replace=False)
    else:
        num_plots, plot_indices = len(hyd_xs), np.arange(len(hyd_xs))
    
    fig, axs = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    for h1 in range(num_plots):
        pcol = axs[h1].pcolormesh(longitudes, latitudes, toas[plot_indices[h1]].T, cmap='Blues_r', linewidth=0, rasterized=True, vmin=0, vmax=1)
        
        for h2 in range(len(hyd_xs)):
            axs[h1].scatter(hyd_xs[h2], hyd_ys[h2], c='k', s=15)
            #axs[h1].text(hyd_xs[h2], hyd_ys[h2], h2, color='k', fontsize=15)
        
        axs[h1].set_aspect('equal', 'box')
        axs[h1].set_ylim(limits_ys)
        axs[h1].set_xlim(limits_xs)
    
    fig.tight_layout()
    return fig

@st.cache_data
def plotProbabilityLandscape(unit_xs, unit_ys, probs, study_xs, study_ys, limits_xs, limits_ys):
    fig, axs = plt.subplots(1, 1, figsize=(8, 4))
    pcol = axs.pcolormesh(study_xs, study_ys, probs.T, cmap='magma', linewidth=0, rasterized=True, vmin=0, vmax=1)

    for h2 in range(len(unit_xs)):
        axs.scatter(unit_xs[h2], unit_ys[h2], c='k', s=15)
        #axs.text(unit_xs[h2], unit_ys[h2], h2 + 1, color='k', fontsize=15)

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
            if (h1 != h2):
                current_product = probs[h1] * probs[h2]
                for h3 in range(num_units):
                    if (h3 != h1) and (h3 != h2):
                        current_product = current_product * (1 - probs[h3])
            
                #print('Final maximum for units %d and %d: %.2f' % (h1, h2, np.max(current_product)))
                detection_at_two_units = detection_at_two_units + current_product
    
    detection_at_three_or_more_units = np.ones((num_lon, num_lat)) - no_detection - detection_at_one_unit - detection_at_two_units

    return no_detection, detection_at_one_unit, detection_at_two_units, detection_at_three_or_more_units


def calculateLocalizationError(localizer, temporal_error, sd_parameter, num_repeats=10):
    # save errors here
    errors_rms = np.zeros((localizer.num_xs, localizer.num_ys))
    error_progress = st.progress(0, 'Calculating Monte-Carlo simulation...')

    # just use whichever depth
    depth = 0

    for i in range(localizer.num_xs):
        for j in range(localizer.num_ys):
            # store RMS error
            #error_list = []
            error_sum = 0

            # repeat calculation ten times
            for r in range(num_repeats):
                # first, get the exact measured times
                random_start_time = np.random.rand() * 10
                measured_times = {unit: localizer.toas_over_depth[depth][unit][i, j] + random_start_time + np.random.normal(scale = temporal_error)
                                for unit in localizer.units}

                # calculate pairwise time differences
                measured_tdoas = {}
                for ri in measured_times:
                    for rj in measured_times:
                        measured_tdoas[(ri, rj)] = (measured_times[ri] - measured_times[rj])

                # localize
                _, _, max_locations, depth = localizer.localizeAcrossDepths(measured_tdoas, sd_param = sd_parameter)

                # calculate error and add to RMS sum
                error = (max_locations[0] - localizer.study_xs[i])**2 + (max_locations[1] - localizer.study_ys[j])**2
                error_sum += error
            
            # save
            errors_rms[i, j] = np.sqrt(error_sum/num_repeats)
            error_progress.progress((i * localizer.num_ys + j)/(localizer.num_xs * localizer.num_ys), 'Calculating Monte-Carlo simulation...')
    
    error_progress.empty()
    return errors_rms

def plotErrorMap(units_xs, units_ys, errors_rms, localizer, limits_xs, limits_ys):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    pcol = ax.pcolormesh(localizer.study_xs, localizer.study_ys, errors_rms.T, cmap='plasma', linewidth=0, rasterized=True, vmin=0, vmax=0.5)

    for j in localizer.units:
        ax.scatter(units_xs[j], units_ys[j], c='w', s=16)
        ax.text(units_xs[j] + 0.0002, units_ys[j] + 0.0002, j, color='w', fontsize=14, weight='bold')
    
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
    fig.tight_layout()
    fig.colorbar(pcol)
    return fig

def halfNormal(x, sd):
    return (np.exp(-x**(2) / (2 * sd**2)))

## ---- NOTES ---- ##

# need cases with and without ice -- different detection functions



## ---- define all coordinates / parameters ---- ##

st.header('Probability of Detection Visualization:')

z = 4

geometries = {'Horizontal Square Grid': lambda d: (np.array([-3/2 * d, -d/2, d/2, 3/2 * d, -3/2 * d, -d/2, d/2, 3/2 * d]), np.array([z, z, z, z, z + d, z + d, z + d, z + d])),
              'Horizontal Triangular Grid': lambda d: (np.array([-d, 0, d, -d/2, d/2, -d, 0, d]), np.array([z, z, z, z + np.sqrt(3)/2 * d, z + np.sqrt(3)/2 * d, z + np.sqrt(3) * d, z + np.sqrt(3) * d, z + np.sqrt(3) * d])),
              'Vertical Square Grid': lambda d: (np.array([-d/2, -d/2, -d/2, -d/2, d/2, d/2, d/2, d/2]), np.array([z, z + d, z + 2 * d, z + 3 * d, z, z + d, z + 2 * d, z + 3 * d])),
              'Vertical Triangular Grid': lambda d: (np.array([-d, -d, -d, d, d, d, 0, 0]), np.array([z, z + d, z + 2 * d, z, z + d, z + 2 * d, z + d/2, z + d * 3/2])),
              'Banner': lambda d: (np.array([-(1 + np.sqrt(3)/2) * d, -d, -d, 0, 0, d, d, (1 + np.sqrt(3)/2) * d]), np.array([z + d/2, z, d + z, z, d + z, z, d + z, z + d/2])),
              'Line': lambda d: (np.array([-7/2 * d, -5/2 * d, -3/2 * d, -d/2, d/2, d * 3/2, d * 5/2, d * 7/2]), 4 * np.ones(8)),
              'Two Squares': lambda d: (np.array([-2 * d, -2 * d, -d, -d, d, d, 2 * d, 2 * d]), np.array([z, z + d, z, z + d, z, z + d, z, z + d])),
              'House': lambda d: (np.array([-d, -d, 0, 0, d, d, -d/2, d/2]), np.array([z, d + z, z, d + z, z, d + z, z + 2 * d, z + 2 * d]))
              #'Plus-sign': lambda d: (np.array([0, 0, 0, 0, 2 * d, d, -d, -2 * d]), np.array([5 * d, 4 * d, 2 * d, d, 3 * d, 3 * d, 3 * d, 3 * d]))
              #'Triangular Grid': lambda d: (np.array([]), np.array([])),
              }

empirical_sds = {'Ice-cover': {60: 17, 70: 12, 80: 4.5},
                 'Open-water': {60: 90, 70: 54, 80: 28}}


with st.sidebar:
    # first, choose and plot detection function
    st.subheader('Detection Function Parameters')

    t = np.linspace(0, 40.0, 1000)
    sd_parameter = st.slider('SD:', value=10.0, min_value=0.0, max_value = 60.0, step=0.5, format="%.1f")
    det_function = lambda x: halfNormal(x, sd_parameter)

    fig_det_function = plotDetFunction(t, det_function, empirical_sds)
    st.pyplot(fig_det_function)
    
    # next, choose array spacing and geometry
    st.subheader('Array Parameters')

    array_choice = st.radio('Choose an array geometry:', list(geometries.keys()))
    array_spacing = st.slider('Array Spacing:', value=5.0, min_value=0.0, max_value = 12.0, step=0.5, format="%.1f")

    unit_xs = geometries[array_choice](array_spacing)[0]
    unit_ys = geometries[array_choice](array_spacing)[1]

    c1, c2 = st.columns(2)
    grid_resolution = c1.number_input('Grid resolution (pixels):', 10, 100, value=60, step=10)
    grid_limits = c2.number_input('Grid limits (km):', 10, 50, value=20, step=1)

    limits_xs = [-grid_limits, grid_limits ] #[1.5 * np.min(unit_xs) - 2, 1.5 * np.max(unit_xs) + 2]
    limits_ys = [0, grid_limits] #[1.5 * np.min(unit_ys) - 2, 1.5 * np.max(unit_ys) + 2]
    study_xs = np.linspace(limits_xs[0], limits_xs[1], grid_resolution)
    study_ys = np.linspace(limits_ys[0], limits_ys[1], grid_resolution)

    positions, distances = precalculate(unit_xs, unit_ys, study_xs, study_ys)

    # st.text('Sample array geometry:')
    # fig_array = plotProbabilityLandscape(unit_xs, unit_ys, np.ones((len(study_xs), len(study_ys))), study_xs, study_ys, limits_xs, limits_ys)
    # st.pyplot(fig_array)
    #st.dataframe(positions)


## ---- define all coordinates / parameters ---- ##

# st.text('Sample array geometry:')
# fig_array = plotProbabilityLandscape(unit_xs, unit_ys, np.ones((len(study_xs), len(study_ys))), study_xs, study_ys, limits_xs, limits_ys)

# # make two columns
# c0, c1, c2 = st.columns((3, 4, 3))
# c1.pyplot(fig_array)


## ---- user input for detection function ---- ##

# max_distance = float(limits_xs[1])

# c0, c1, c2 = st.columns((2, 6, 2))
# function_type = c1.radio('Choose a detection function:', ['Half-normal', 'Hazard-rate'])

# t = np.linspace(0, max_distance, 1000)

# if (function_type == 'Half-normal'):
#     sd_parameter = c1.slider('SD:', value=1.0, min_value=0.0, max_value = max_distance, step=0.01, format="%.3f")
#     det_function = lambda x: (np.exp(-x**(2) / (2 * sd_parameter**2)))
# else:
#     #c1, c2 = st.columns(2)
#     beta_parameter = c1.slider('Exponent (beta):', value=1.0, min_value=0.0, max_value = 10.0, step=0.01, format="%.3f")
#     sd_parameter = c1.slider('SD:', value=1.0, min_value=0.0, max_value = max_distance, step=0.01, format="%.3f")
#     det_function = lambda x: (1 - np.exp(-(x / sd_parameter)**(-beta_parameter)))

# # plot detection function
# fig_det_function = plotDetFunction(t, det_function)
# #c0_det_fig, c1_det_fig, c2_det_fig = st.columns((3, 4, 3))
# c1.pyplot(fig_det_function)


## ---- plots and calculations ---- ##

# calculate probabilities
probs = calculateProbs(det_function, distances)

# calculate all detection probabilities
no_detection, detection_at_one_unit, detection_at_two_units, detection_at_three_or_more_units = detectionProbabilities(probs)

st.text('Probability that a call is detected at three or more units:')
fig_det_at_three_plus_units = plotProbabilityLandscape(unit_xs, unit_ys, detection_at_three_or_more_units, study_xs, study_ys, limits_xs, limits_ys)

c0, c1, c2 = st.columns((2, 6, 2))
c1.pyplot(fig_det_at_three_plus_units)

# calculate time-difference-of-arrival for each hydrophone pair
with st.expander('Probability of detection for each unit individually:'):
    fig_toa = plotIndividualProbs(unit_xs, unit_ys, study_xs, study_ys, probs, limits_xs, limits_ys)
    st.pyplot(fig_toa)

with st.expander('Probability that a call is not detected at any units:'):
    fig_no_detection = plotProbabilityLandscape(unit_xs, unit_ys, no_detection, study_xs, study_ys, limits_xs, limits_ys)
    c0, c1, c2 = st.columns((2, 6, 2))
    c1.pyplot(fig_no_detection)

with st.expander('Probability that a call is detected at exactly one unit:'):
    fig_det_at_one_unit = plotProbabilityLandscape(unit_xs, unit_ys, detection_at_one_unit, study_xs, study_ys, limits_xs, limits_ys)
    c0, c1, c2 = st.columns((2, 6, 2))
    c1.pyplot(fig_det_at_one_unit)

with st.expander('Probability that a call is detected at exactly two units:'):
    fig_det_at_two_units = plotProbabilityLandscape(unit_xs, unit_ys, detection_at_two_units, study_xs, study_ys, limits_xs, limits_ys)
    c0, c1, c2 = st.columns((2, 6, 2))
    c1.pyplot(fig_det_at_two_units)

## ---- map ---- ##

class FloatReader:
    def __init__(self, filename):
        self.f = open(filename, "rb")
    
    def read_floats(self, count : int):
        return np.fromfile(self.f, dtype=np.float32, count=count, sep='')


def rotateArray(x, y, theta, x_mean, y_mean):
    theta = theta * np.pi/180
    rotation = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    output = rotation @ np.array([x, y])
    return output[0] + x_mean, output[1] + y_mean

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

    return bathymetry, lons, lats, coast, perch_locations_metric


def drawMap(bathymetry, lons, lats, coast, perch_locations_metric, x_array, y_array):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    contours = plt.contourf(lons, lats, np.flipud(bathymetry), levels = np.arange(-140, 9, 10), cmap='Blues', zorder=2)
    plt.contour(lons, lats, np.flipud(bathymetry), levels = [-75], cmap='cool', zorder=3)
    # plt.pcolormesh(np.flipud(data[data > 0]), levels = np.arange(0, 50, 10), c='k')
    #plt.pcolormesh(np.flipud(data_ground))
    plt.colorbar(contours)
    plt.scatter(x_mean, y_mean, s=60, color='red', marker='*', zorder=8)

    for i, point in enumerate(perch_locations_metric):
        plt.scatter(point.x, point.y, s=20, color='orange', marker='*', zorder=10)

    for i in np.arange(8):
        plt.scatter(x_array, y_array, s=15, c='k', zorder=10)

    coast.plot(ax=ax, color = 'k', zorder=4)
    ax.set_xlabel('Projected Longitude (km)')
    ax.set_ylabel('Projected Latitude (km)')
    return fig

st.header('Map Visualization:')


# prepare map information
bathymetry, lons, lats, coast, perch_locations_metric = loadMapData()

# mean longitude/latitude
x_mean = np.mean([point.x for point in perch_locations_metric])
y_mean = np.mean([point.y for point in perch_locations_metric])

# user parameters for map
c1, c2, c3 = st.columns(3)
x_center = c1.number_input('Center longitude (km):', min_value=570.0, max_value=590.0, value=x_mean, step=1.0)
y_center = c2.number_input('Center latitude (km):', min_value=7910.0, max_value=7930.0, value=y_mean, step=1.0)
rotation = c3.number_input('Rotation (degrees):', min_value=-90, max_value=90, step=5, value=-35)

# translate and rotate array
x_array, y_array = rotateArray(unit_xs, unit_ys, rotation, x_center, y_center)

# plot map
fig_map = drawMap(bathymetry, lons, lats, coast, perch_locations_metric, x_array, y_array)
st.pyplot(fig_map)


st.header('Coordinates:')

proposed_array_metric = gpd.points_from_xy(x_array, y_array, crs=coast.crs)
proposed_array_epsg = proposed_array_metric.to_crs("EPSG:4326")

depths = np.zeros(8)
proposed_longitudes = np.zeros(8)
proposed_latitudes = np.zeros(8)

for i in range(8):
    proposed_longitudes[i] = proposed_array_epsg[i].x
    proposed_latitudes[i] = proposed_array_epsg[i].y
    try:
        depths[i] = np.round(interpn((lons, lats), np.flipud(bathymetry).T, (x_array[i], y_array[i])), 1)
    except:
        depths[i] = np.nan

array = pd.DataFrame({'Longitude': proposed_longitudes, 'Latitude': proposed_latitudes, 'Depth': depths})
st.dataframe(array)


## ---- spatial error ---- ##

# st.header('Visualizing spatial error across an array:')

# with st.form("Error parameters"):
#     c1, c2, c3 = st.columns(3)
#     temporal_error = c1.number_input('Temporal measurement error:', min_value=0.0, max_value=5.0, value=0.01)
#     sd_parameter = c2.number_input('Temporal SD parameter:', min_value=0.0, max_value=5.0, value=0.1)
#     num_repeats = c3.number_input('Monte-Carlo runs:', min_value=0, max_value=100, step=1, value=5)
#     submitted = st.form_submit_button("Calculate!")

# # next, set a temporal error spatial deviation
# if submitted:
#     localizer = Localizer(unit_xs, unit_ys, study_xs, study_ys, speed_of_sound = 1.5, source_depths = [0])

#     errors_rms = calculateLocalizationError(localizer, temporal_error, sd_parameter, num_repeats)
#     print(np.min(errors_rms), np.max(errors_rms))
#     fig_error = plotErrorMap(unit_xs, unit_ys, errors_rms, localizer, limits_xs, limits_ys)

#     c0, c1, c2 = st.columns((2, 6, 2))
#     c1.pyplot(fig_error)