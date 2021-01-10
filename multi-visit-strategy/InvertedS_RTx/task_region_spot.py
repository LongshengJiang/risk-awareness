# In this script, we define the task for each robot in the human-robot team.
# This script needs to run only once to define the initial prior probability of target PRESENCE in the cells in each
# search region.

# First thing first, let's import some libraries.
import numpy as np
import random as rd


class Region:
    def __init__(self, row=30, col=30):
        # Each cell in the region has its own probability of target presence.
        # Define the probability at each cell randomly.
        self.row = row
        self.col = col
        # Define the basic probability pattern for the region.
        self.prob = np.random.uniform(low=0, high=0.2, size=(row, col))
        # We will add several spot pattern on top of the basic pattern.
        self.add_spot_pattern()
        # Initialize matrix to represent target presence or absence at each cell.
        # presence = 1, absence = 0
        self.tar_presence = np.ones(shape=self.prob.shape, dtype=int) * -1
        # Initialize matrix to represent observations at each cell.
        # observed presence = 1, oberved absence = 0
        self.tar_obsv = np.ones(shape=self.prob.shape, dtype=int) * -1
        # Initialize the total number of targets in this region.
        self.num_tar = 0

    @staticmethod
    def spot_pattern():
        # Initialize the first probability pattern.
        prob_pattern = np.zeros((10, 10))
        prob_pattern[0:2, :] = np.random.uniform(low=0, high=0.2, size=(2, 10))
        prob_pattern[-2:, :] = np.random.uniform(low=0, high=0.2, size=(2, 10))
        prob_pattern[:, 0:2] = np.random.uniform(low=0, high=0.2, size=(10, 2))
        prob_pattern[:, -2:] = np.random.uniform(low=0, high=0.2, size=(10, 2))
        prob_pattern[2:4, 2:8] = np.random.uniform(low=0.1, high=0.4, size=(2, 6))
        prob_pattern[-4:-2, 2:8] = np.random.uniform(low=0.1, high=0.4, size=(2, 6))
        prob_pattern[2:8, 2:4] = np.random.uniform(low=0.1, high=0.4, size=(6, 2))
        prob_pattern[2:8, -4:-2] = np.random.uniform(low=0.1, high=0.4, size=(6, 2))
        prob_pattern[4:6, 4:6] = np.random.uniform(low=0.2, high=0.8, size=(2, 2))
        return prob_pattern

    def add_spot_pattern(self):
        # We will add several spot pattern on top of the basic pattern.
        # Let us randomly select locations for the pattern.
        pattern_locs = set()
        for n in range(5):
            loc_row = rd.randint(0, 2)
            loc_col = rd.randint(0, 2)
            pattern_locs.add((loc_row, loc_col))
        # We now add pattern 1 to the basic prob. pattern.
        for loc in pattern_locs:
            self.prob[(loc[0] * 10):(loc[0] + 1) * 10, (loc[1] * 10):(loc[1] + 1) * 10] = self.spot_pattern()

    def targets(self):
        # The function determines if a cell contains a target. It uses the prior distribution.
        for row_indx in range(self.tar_presence.shape[0]):
            for col_indx in range(self.tar_presence.shape[1]):
                # Here we use a random number generator to assign target to a cell.
                rand_num = np.random.random()
                if rand_num <= self.prob[row_indx, col_indx]:
                    self.tar_presence[row_indx, col_indx] = 1
                else:
                    self.tar_presence[row_indx, col_indx] = 0
        # Check if the target presence has been generated successfully.
        if sum(sum(self.tar_presence)) < 0:
            raise ValueError('Target presence matrix is not correct.')
        # We want to know how many targets in this region.
        self.num_tar = sum(sum(self.tar_presence))


# Let's assume there are 20 robots in the human-robot team. There are 15 independent task regions.
num_regions = 20
# Let's define the size of each region.
num_row = 30
num_col = 30
# Let's initialize a 3-D array for storing the region information of all the robots.
domain_prob = np.zeros(shape=(num_row, num_col, num_regions))
# Let's initialize a 3-D array for target presence information of all the robots.
domain_tar = np.zeros(shape=(num_row, num_col, num_regions), dtype=int)
# Let's initialize 1-D array for the number of targets in each region.
domain_num_tar = np.zeros(shape=num_regions, dtype=int)

# Use a for loop to create a task region for each robot.
for i in range(num_regions):
    region = Region(num_row, num_col)
    region.targets()
    domain_prob[:, :, i] = region.prob
    domain_tar[:, :, i] = region.tar_presence
    domain_num_tar[i] = region.num_tar
# Save the domain prior probability information
np.save('./data/domain_prob', domain_prob)
np.save('./data/domain_tar', domain_tar)
np.save('./data/domain_num_tar', domain_num_tar)
