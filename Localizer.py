import numpy as np
from matplotlib import pyplot as plt

#SOUND_SPEED = 1.500 # previously 1.484

# updated locations across all five hydrophones
# hyd_locations = {'S01': [13.40288, 144.60390, 913.5],
#                 'S02': [13.40306, 144.61397, 636.0],
#                 'S03': [13.39431, 144.61423, 742.9],
#                 'S04': [13.39388, 144.60421, 980.9], 
#                 'S05': [13.39822, 144.60934, 833.2]}

# measured in December, incorporating S02, based on May through July
# measured_travel_times = {'S01': {'S02': 1.266771875,    'S01': 0,           'S03': 1.48095,     'S04': 1.4413375,   'S05': 1.289125},
#                          'S02': {'S01': 1.266771875,    'S03': 1.1120375,   'S02': 0,           'S04': 1.464,       'S05': 1.09400625},
#                          'S03': {'S01': 1.48095,        'S02': 1.1120375,   'S04': 0.74053125,  'S03': 0,           'S05': 0.447775},
#                          'S04': {'S01': 1.4413375,      'S02': 1.464,       'S03': 0.74053125,  'S05': 0.4987375,   'S04': 0},
#                          'S05': {'S01': 1.289125,       'S02': 1.09400625,  'S03': 0.447775,    'S04': 0.4987375,   'S05': 0}}


# guam, num_xs, num_ys, lon_limits, lat_limits, study_xs, study_ys, hyd_locations, theoretical_toas, theoretical_tdoas, hyd_travel_times, hyd_travel_times_with_S01_reflection

class Localizer():
    def __init__(self, units_xs, units_ys, study_xs, study_ys, speed_of_sound, source_depths = [0]):
        self.num_xs = len(study_xs)
        self.num_ys = len(study_ys)

        self.lon_limits = [np.min(study_xs), np.max(study_xs)]
        self.lat_limits = [np.min(study_ys), np.max(study_ys)]

        self.units = np.arange(len(units_xs))
        self.unit_locations = {}
        for unit in range(len(units_xs)):
            self.unit_locations[unit] = [units_xs[unit], units_ys[unit]]
        
        self.speed_of_sound = speed_of_sound

        # get a linear spacing 
        self.study_ys = study_ys
        self.study_xs = study_xs
        self.source_depths = source_depths

        # calculate ToAs and TDoAs
        self.toas_over_depth = {}
        self.tdoas_over_depth = {}
        for depth in source_depths:
            self.toas_over_depth[depth], self.tdoas_over_depth[depth] = self.TOAandTDOA()

    def plotLikelihoodProduct(self, likelihood_product):
        # plot the TDoA's
        max_likelihood = np.max(likelihood_product)
        #likelihood_product = max_likelihood * likelihood_product/np.sum(likelihood_product)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        pcol = ax.pcolormesh(self.study_xs, self.study_ys, likelihood_product.T, cmap='Blues', linewidth=0, rasterized=True, vmax=np.maximum(0.05, np.max(likelihood_product)))
        #self.guam.to_crs(epsg=4326).plot(ax=ax, color='darkgrey')

        for j in self.units:
            ax.scatter(self.unit_locations[j][1], self.unit_locations[j][0], c='k', s=15)
            ax.text(self.unit_locations[j][1], self.unit_locations[j][0], j, color='k', fontsize=15)

        ax.set_title('Overall Likelihood (Max Value = %.2f)' % max_likelihood, fontsize=14)
        ax.set_xlim(self.lon_limits)
        ax.set_ylim(self.lat_limits)
        fig.tight_layout()
        fig.colorbar(pcol)
    
    def plotLikelihoodGrid(self, likelihood_surfaces):
        fig, axs = plt.subplots(len(self.unit_locations), len(self.unit_locations), figsize=(18, 15))
        for h1, hyd1 in enumerate(self.unit_locations):
            for h2, hyd2 in enumerate(self.unit_locations):
                if (hyd1, hyd2) in likelihood_surfaces.keys():
                    axs[h1, h2].pcolormesh(self.study_xs, self.study_ys, likelihood_surfaces[(hyd1, hyd2)].T,
                                                    cmap='Blues', linewidth=0, rasterized=True)

                    #self.guam.to_crs(epsg=4326).plot(ax=axs[h1, h2], color='darkgrey')

                    for i in self.unit_locations:
                        axs[h1, h2].scatter(self.unit_locations[i][1], self.unit_locations[i][0], c='k', s=15)
                        axs[h1, h2].text(self.unit_locations[i][1], self.unit_locations[i][0], i, color='k', fontsize=15)
                    
                    axs[h1, h2].set_title('Feasible Region for %s & %s' % (hyd1, hyd2), fontsize=14)
                    axs[h1, h2].set_xlim(self.lon_limits)
                    axs[h1, h2].set_ylim(self.lat_limits)

        fig.tight_layout()
    
    # calculate time-of-arrival for each hydrophone
    def TOAandTDOA(self):
        toas = {}
        for hyd in self.unit_locations:
            toas[hyd] = np.zeros((self.num_xs, self.num_ys))
            for i in range(self.num_ys):
                for j in range(self.num_xs):
                    toas[hyd][j, i] = np.linalg.norm(np.array([self.study_xs[j], self.study_ys[i]]) - np.array(self.unit_locations[hyd])) / self.speed_of_sound

        # calculate time-difference-of-arrival for each hydrophone pair
        tdoas = {}
        for hyd1 in self.unit_locations:
            for hyd2 in self.unit_locations:
                tdoas[(hyd1, hyd2)] = toas[hyd1] - toas[hyd2]
        
        return toas, tdoas
    
    def localizeTDoAs(self, measured_tdoas, theoretical_tdoas, sd_param):
        # calculate the likelihood surfaces!!
        likelihood_surfaces = {}
        for hyd1 in self.units:
            for hyd2 in self.units:
                if (hyd1 != hyd2) and (hyd1, hyd2) in measured_tdoas.keys():
                    likelihood = np.exp(-1/(2 * sd_param**2) * (theoretical_tdoas[(hyd1, hyd2)] - measured_tdoas[(hyd1, hyd2)])**2)
                    likelihood_surfaces[(hyd1, hyd2)] = np.zeros(likelihood.shape) if (np.max(likelihood) < 1e-8) else (likelihood / np.max(likelihood))

        likelihood_product = np.ones((self.num_xs, self.num_ys)) 
        for pair in likelihood_surfaces:
            likelihood_product = likelihood_product * likelihood_surfaces[pair]

        return likelihood_surfaces, likelihood_product
    
    def localizeAcrossDepths(self, measured_tdoas, sd_param):
        likelihood_surfaces = {}
        likelihood_products = {}
        likelihood_values = {}
        likelihood_values_normalized = {}
        locations = {}

        for depth in self.source_depths:
            likelihood_surfaces[depth], likelihood_products[depth] = self.localizeTDoAs(measured_tdoas, self.tdoas_over_depth[depth], sd_param)
            likelihood_values[depth] = np.max(likelihood_products[depth])
            likelihood_values_normalized[depth] = np.max(likelihood_products[depth])/np.sum(likelihood_products[depth])
            argmax = np.unravel_index(np.argmax(likelihood_products[depth], axis=None), likelihood_products[depth].shape)
            locations[depth] = (self.study_xs[argmax[0]], self.study_ys[argmax[1]])
        
        best_depth = max(likelihood_values, key=likelihood_values.get)
        return likelihood_surfaces[best_depth], likelihood_products[best_depth], locations[best_depth], best_depth

    
def logTransform(spec, scale=10**(-5)):
    return 20 * np.log10(spec + scale * np.ones(spec.shape))

def getReflectedDistance(x1, y1, z1, x2, y2, z2):
    # get just horizontal distance
    d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    # find where the surface reflection happens, approximately
    h1 = d * z1 / (z1 + z2)
    h2 = d * z2 / (z1 + z2)
    
    # calculate overall distance
    dist = np.sqrt(z1**2 + h1**2) + np.sqrt(z2**2 + h2**2)
    return dist

# try calculating distances by mapping into metric space
def relativeXY(lat, lon, ref_lat, ref_lon):
    # x-axis distance is more complex because it scales with latitude
    dx = (lon - ref_lon)/360 * 40000 * np.cos(ref_lat * np.pi/180)
    
    # latitude is just proportionate to the degree difference
    dy = (lat - ref_lat)/360 * 40000
    
    # return coordinates and distance
    return dx, dy, np.sqrt(dx**2 + dy**2)