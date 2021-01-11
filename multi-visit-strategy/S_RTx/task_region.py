# In this script, we define the task for each robot in the human-robot team.
# This script needs to run only once to define the initial prior probability of target PRESENCE in the cells in each
# search region.

# First thing first, let's import some libraries.
import numpy as np
import random as rd


class Region:
    def __init__(self, row=50, col=50):
        # Each cell in the region has its own probability of target presence.
        # Define the probability at each cell randomly.
        self.row = row
        self.col = col
        # Define the prior probability of the domain randomly.
        self.prob = np.random.uniform(low=0, high=0.2, size=(self.row, self.col))
        # Initialize matrix to represent target presence or absence at each cell.
        # presence = 1, absence = 0
        self.tar_presence = np.ones(shape=self.prob.shape, dtype=int) * -1
        # Initialize matrix to represent observations at each cell.
        # observed presence = 1, oberved absence = 0
        self.tar_obsv = np.ones(shape=self.prob.shape, dtype=int) * -1
        # Initialize the total number of targets in this region.
        self.num_tar = 0

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
num_regions = 1
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
