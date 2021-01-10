# This the main script for my TCST0.1 simulation.

# First thing first, let us import some libraries.
# Import standard libs.
import numpy as np
import matplotlib.pyplot as plt
import os

# Import customized libs. Please mark the directory to these files as source root in Pycharm setting.
import dijkstra as dj
import optimal_order as oo


class Robot:
    # We define a class to represent the operations of one robot.
    def __init__(self, robot_id, region_prob):
        # region_prob is the prior probability distribution of target presence at each cell in this robot's region.
        self.region_prob = region_prob
        # We initialize an array to record which cells have been visited.
        self.visited_cell = np.full(region_prob.shape, False)
        # robot_id is the id number of this robot.
        self.id = robot_id
        # We need to specify where initially the robot is.
        self.start = (0, 0)
        # We also need to record the current location of the robot at each time step. Inititially, it is at the start.
        self.current = self.start
        # Create a sensor model for the robot detection.
        self.p_o1_s1 = 0.9  # Probability of true positive.
        self.p_o1_s0 = 0.4  # Probability of false positive.
        self.p_o0_s1 = 1 - self.p_o1_s1
        self.p_o0_s0 = 1 - self.p_o1_s0
        # Let us pack these probabilities into a sensor model
        self.sensor_model = {'p_o1_s1': self.p_o1_s1, 'p_o0_s0': self.p_o0_s0}
        # Initialize an empty matrix to save the observed target presence.
        self.region_obvs = np.ones(shape=region_prob.shape, dtype=int) * -1  # -1 means no observation yet.
        # We also keep a copy of the current observation.
        self.observation = -1
        # Initialize a posterior probability of the observation.
        self.post_prob = -0.5  # This is a nonsense value. I just want to get the type right.
        # Initialize the tuple for communicating with the queuing agent.
        self.robot_tuple = (self.id, self.post_prob)
        # Initialize an empty list for the planned path during path planning.
        self.optimal_path = []
        # Initialize an empty list for the history trajectory.
        self.traj_path = []
        # We need to record if the robot has finished searching its region.
        self.finish_search = False

    def plan(self):
        # This function helps the robot plan its next movement using Dijkstra's algorithm.
        # Because we use Dijkstra' algorithm, we first has to represent current region in a graph. The current cell is
        # the starting point
        graph = dj.Graph(self.region_prob, start=self.current)
        # Get the optimal path with the maximal utility in the graph.
        self.optimal_path = graph.max_util_path()

    def show_path(self):
        # Plot the heatmap of the probabilities of this region.
        # plt.ion()
        fig, ax = plt.subplots()
        # probmap = ax.imshow(self.local_util, cmap='YlOrBr', vmin=-1)
        probmap = ax.imshow(self.region_prob, cmap='hot_r', vmin=0, vmax=1)
        cbar = fig.colorbar(probmap)
        # Label the color bar
        cbar.ax.set_xlabel(r'$p(s=1)$')
        # Overlay with the plot of the trajectory path.
        traj_x = [waypoint[0] for waypoint in self.traj_path] + [self.current[0]]
        traj_y = [waypoint[1] for waypoint in self.traj_path] + [self.current[1]]
        ax.plot(traj_y, traj_x, 'b-o', markersize=2, linewidth=1)  # make sure it is (y, x). plot function treats vertical axis (rows) as y.
        # Overlay with the plot of the optimal path.
        opt_x = [waypoint[0] for waypoint in self.optimal_path]
        opt_y = [waypoint[1] for waypoint in self.optimal_path]
        ax.plot(opt_y, opt_x, 'g--o', markersize=2, linewidth=1)  # make sure it is (y, x). plot function treats vertical axis (rows) as y.
        # Plot a start point.
        ax.plot(self.optimal_path[0][1], self.optimal_path[0][0], 'gs', markersize=4, linewidth=1)
        # Plot an end point.
        ax.plot(self.optimal_path[-1][1], self.optimal_path[-1][0], 'g^', markersize=4, linewidth=1)
        # Save the plot to files.
        picname = './pics/path' + str(len(self.traj_path))
        plt.savefig(picname, format='pdf', bbox_inches='tight')
        plt.close(fig)
        # plt.show()
        # plt.pause(2)

    def move(self):
        # This function helps the robot moves to the next cell.
        # Just update the current cell to the next cell in the planned path. Note, that path includes the current cell.
        # So the next cell is optimal_path[1]. This is done when the optimal_path is at least 2 elements. If it contains
        # only one element, the robot stays at the current location.
        if len(self.optimal_path) > 1:
            # Save the current location to trajectory.
            self.traj_path += [self.current]
            # Move the current location to the next cell in the optimal path.
            self.current = self.optimal_path[1]

    def detection(self, observation):
        # observation is the observation of target at the current cell.
        # Check if the cell has been visited before. Visited means an observation, from the human or the robot, is
        # obtained and the robot left this cell.
        # If this cell has been visited, then we accept our previous observation at this cell by assuming the
        # observation is correct (post_prob = 1).
        if self.visited_cell[self.current]:
            self.post_prob = 1.0
        # If this cell has not been visited before, then we use Bayes' law to get the posterior probability for
        # 1) the detection being correct and 2) the target presence.
        else:
            # We first save the observation.
            self.observation = observation
            # To do so, let us first get the prior probability of the target being present at this cell.
            prior_prob = self.region_prob[self.current]
            # Depending on the observation, the posterior probability of the detection being correct is different.
            # If the observation is target presence (observation = 1),
            if observation == 1:
                # The probability of being correct is p(s=1 | o=1). We use Bayes's law to get it.
                self.post_prob = self.p_o1_s1 * prior_prob / (
                            self.p_o1_s1 * prior_prob + self.p_o1_s0 * (1 - prior_prob))
                # Because we have done an observation, we now can update the probability at this cell.
                # This probability is also p(s=1 | o=1).
                self.region_prob[self.current] = self.post_prob
                # Now the region prob info is updated.
                #
            # If the observation is target absence (observation = 0)
            else:
                # The probability of being correct is p(s=0 | o=0). We use Bayes's law to get it.
                self.post_prob = self.p_o0_s0 * (1 - prior_prob) / (
                            self.p_o0_s0 * (1 - prior_prob) + self.p_o0_s1 * prior_prob)
                # Because we have done an observation, we now can update the probability at this cell.
                # This probability is p(s=1 | o=0) = 1 - p(s=0 | o=0)
                self.region_prob[self.current] = 1.0 - self.post_prob
                # Now the region prob info is updated.
        # Now save the posterior probability with the robot's id as a tuple (robot_id, posterior_prob)
        self.robot_tuple = (self.id, self.post_prob)
        return self.robot_tuple

    def confirm(self):
        # This function confirm the observation before the robot leave the current cell.
        # We will only confirm the observation if the probability of the observation to be correct is greater than 0.9
        # and the cell is not visited.
        if self.post_prob >= 0.9 and not self.visited_cell[self.current]:
            # We first set the current cell as a visited one.
            self.visited_cell[self.current] = True
            # We do not want the robot to re-visit this cell. Hence, despite whether this cell contains a target or not,
            # and despite what observation was obtained, we claim there is no target in this cell.
            # Once we do this, the path planning algorithm will not be motivated to visit this cell.
            self.region_prob[self.current] = 0  # To indicate this cell has been visited, we set the probability to 0.
            # If in the current cell, a target is detected, we add 1 to the number of found targets.
            # We record the observation at the current cell to the region.
            self.region_obvs[self.current] = self.observation
            # Let us empty self.observation to avoid any mistakes.
            self.observation = -1
            # Check if the the robot has visited all the cells in its region.
            if sum(sum(self.visited_cell)) == self.visited_cell.size:
                self.finish_search = True
            else:
                self.finish_search = False

    def hm_detection(self, observation):
        # This function updates the probability of target presence at the current cell after the human helped the robot.
        # We first save the observation.
        self.observation = observation
        # The probability of being correct is 1.
        self.post_prob = 1.0
        # If the observation is target presence (observation = 1),
        if observation == 1:
            # This probability is p(s=1 | o=1).
            self.region_prob[self.current] = 1.0
            # Now the region prob info is updated.
            #
        # If the observation is target absence (observation = 0)
        else:
            # Because we have done an observation, we now can update the probability at this cell.
            # This probability is p(s=1 | o=0) = 1 - p(s=0 | o=0)
            self.region_prob[self.current] = 0.0
            # Now the region prob info is updated.
        # Now update the posterior probability with the robot's id as a tuple (robot_id, posterior_prob)
        self.robot_tuple = (self.id, self.post_prob)
        return self.robot_tuple


class Agent:
    # Agent receives the communication from the robots, in the form of a python list of robot tuples.
    def __init__(self):
        # We note which decision-making model is used.
        self.dm_model = ''
        # Here we initialize an empty list for the ordered waiting line.
        self.wait_line = []
        # We also initialize an empty list for the robots not in the waiting line.
        self.release_robots = []

    def ordering(self, robots, human_cost_bs=0., human_cost_co=-5., wrong_cost=-100.):
        # robots is a list of robot tuples. Each robot tuple is of this form (robot_id, posterior_prob)
        # This function uses regret theory with extension model.
        self.dm_model = 'RTx'
        # Here we use the optimal_order module. The detailed codes are in optimal_order.py within this directory.
        opt_order = oo.OptimalOrder(hum_cost_bs=human_cost_bs, hum_cost_co=human_cost_co, wrong_det_cost=wrong_cost)
        # Do the ordering now.
        opt_order.line_order(robots)
        # Save the results.
        self.wait_line = opt_order.robot_list_wait
        self.release_robots = opt_order.robot_list_release

    def ordering_ev(self, robots, human_cost_bs=0., human_cost_co=-5., wrong_cost=-100.):
        # robots is a list of robot tuples. Each robot tuple is of this form (robot_id, posterior_prob)
        # This function we use Expected Value criterion to order the line.
        self.dm_model = 'EV'
        # Here we use the optimal_order module. The detailed codes are in optimal_order.py within this directory.
        opt_order = oo.OptimalOrder(hum_cost_bs=human_cost_bs, hum_cost_co=human_cost_co, wrong_det_cost=wrong_cost)
        # Do the ordering now.
        opt_order.line_length_ev(robots)
        # Save the results.
        self.wait_line = opt_order.robot_list_wait
        self.release_robots = opt_order.robot_list_release

    def ordering_no_hm(self, robots, human_cost_bs=0., human_cost_co=-5., wrong_cost=-100.):
        # In this function, we suppose there is no human in the team.
        self.dm_model = 'No human'
        self.wait_line = []
        self.release_robots = robots


class Human:
    # The human does the manual detection. Manual detection takes longer time than robot detection.
    def __init__(self, num_history_services):
        # Let us first determine how many times slower human is.
        self.hm_tsk_dur = 20  # human task duration is several times slower than that of the robot detection.
        # Let us define the human base cost
        self.human_cost_bs = 0
        # Initialize the time to go for each detection task.
        self.time2go = 1
        # Create a sensor model for the human detection.
        self.p_o1_s1 = 1  # Probability of observing presence when target is present.
        self.p_o0_s0 = 1  # Probability of observing absence when target is absent.
        self.p_o0_s1 = 1 - self.p_o1_s1
        self.p_o1_s0 = 1 - self.p_o0_s0
        # Let us pack these probabilities into a sensor model
        self.sensor_model = {'p_o1_s1': self.p_o1_s1, 'p_o0_s0': self.p_o0_s0}
        # We need to know the which robot the human is currently helping.
        self.tele_robotid = -1  # Here we initialize nonsense id. We just want to get the type right.
        # We need to record how many times, the human is used.
        self.hm_det_num = num_history_services

    def start_tele(self, robot_id):
        # The function starts the human tele-operation.
        self.tele_robotid = robot_id
        self.time2go = self.hm_tsk_dur

    def progress(self):
        # This function records the reduction of time to go.
        self.time2go -= 1
        return self.time2go


class Environment:
    # The environment knows the ground true of object presence or absence in each cell in the domain.
    def __init__(self):
        # Load the ground truth about the target presence in the domain.
        self.domain_tar = np.load("./data/domain_tar.npy")
        # Load the prior probability about the target presence in the domain.
        self.domain_prob = np.load("./data/domain_prob.npy")
        # Load the total number of targets.
        self.num_tar = np.load("./data/domain_num_tar.npy")
        # Initialize the cost of losing one target
        self.cost_tar = -100

    def get_observation(self, robot_id, sensor_model, cell_loc):
        # This function provides an observation to either a robot or the human.
        # To determine the observation, the environment needs to know the sensor model, the querying robot id (robot_id)
        # and cell location (cell_loc). The region id is the same as the robot_id.
        # The output is the observation, generated randomly.
        # We extract the robot sensor model here.
        p_o1_s1 = sensor_model['p_o1_s1']  # Probability of observing presence when target is present.
        p_o0_s0 = sensor_model['p_o0_s0']  # Probability of observing absence when target is absent.
        # Now we want to know which region and which cell in the region we want to get an observation.
        region_indx = robot_id - 1  # Python is 0-based indexing, while robot_id starts from 1.
        region_tar = self.domain_tar[:, :, region_indx]
        # Here we use a random number to generate the observations.
        rand_num = np.random.random()
        # If the target is present in this cell, we generate the observation using p_o1_s1
        if region_tar[cell_loc] == 1:
            if rand_num <= p_o1_s1:
                observation = 1  # Observe target presence
            else:
                observation = 0
        # If the target is absent in this cell, we generate the observation using p_o0_s0
        else:
            if rand_num <= p_o0_s0:
                observation = 0  # Observe target absence
            else:
                observation = 1
        return observation

    def get_region_prob(self, robot_id):
        # This function provides the prior probabilities in the region of robot_id.
        region_indx = robot_id - 1  # Python is 0-based indexing, while robot_id starts from 1.
        region_prob = self.domain_prob[:, :, region_indx]
        return region_prob


def main():
    # This is the main function.
    # Set the decision-making model here.
    decision_model = 'RTx'  # 'No human', 'EV', or 'RTx'
    # ========================================================================================
    # Spawn the players
    # ========================================================================================
    # We define an environment
    our_env = Environment()
    # We can know the total number of robots in the environment.
    num_all_robot = our_env.num_tar.size
    # We define an ordering agent.
    our_agent = Agent()
    # We define an human who has served 0 robots before.
    our_hm = Human(0)
    our_hm_cost_bs = our_hm.human_cost_bs
    # We define the time limit.
    total_time_step = 18000
    # We define the human cost coefficient.
    our_hm_cost_co = -our_hm.hm_tsk_dur
    # We define num_all_robot different robots with the following for-loop.
    our_rbt_set = []
    for robot_id in range(1, num_all_robot + 1):
        # Retrieve the region prior probabilities.
        region_prob = our_env.get_region_prob(robot_id)
        # Define one robot.
        our_robot = Robot(robot_id, region_prob)
        our_rbt_set = our_rbt_set + [our_robot]
    # ======================================================================================
    # Starting operations of robots
    # ======================================================================================
    # To kick off the search mission, the robots observe the environment.
    # Create a list for storing information used in robot ordering. The format is [(robot_id, probability)]
    robot_tuple_list = [(0, 0.)] * num_all_robot
    # Get the robot tuple from each robot.
    for robot_indx in range(num_all_robot):
        # Get observations from the environment.
        # Which robot?
        robot_id = our_rbt_set[robot_indx].id
        # What is this robot's sensor model?
        sensor_model = our_rbt_set[robot_indx].sensor_model
        # Where is this robot currently?
        current_loc = our_rbt_set[robot_indx].current
        # What is this robot's observation in the environment?
        obsv_1st = our_env.get_observation(robot_id=robot_id, sensor_model=sensor_model, cell_loc=current_loc)
        # Save the robot detection information into the robot tuple list, and updates the probabilities in the domain.
        robot_tuple_list[robot_indx] = our_rbt_set[robot_indx].detection(observation=obsv_1st)
    # =======================================================================================
    # Now we create the main loop
    # =======================================================================================
    # Initialize the current time step.
    time_step = 0
    # We want to know how many targets are missing.
    mis_tar = sum(our_env.num_tar)
    # We want to record how many targets have been found.
    total_found_tar = 0
    # We want to record the waiting line length in each iteration.
    line_len_vec = []
    # This is the main while loop. This loop stops only when time_limit is reached or all targets are found.
    while time_step < total_time_step and total_found_tar < mis_tar:
        # Count up one time step.
        time_step += 1
        # ======================================================================================
        # We check is all the regions have been visited by the corresponding robots.
        finish_searching = [our_rbt_set[rbt_indx].finish_search for rbt_indx in range(num_all_robot)]
        # If all the robots have finished searching their region, we terminate the while loop.
        if all(finish_searching):
            break
        # ======================================================================================
        # The ordering agent orders the robot. When one robot is under tele-operation of the human, this robot should be
        # put aside.
        # If there is one robot under tele-operation, a.k.a, the id of teleoperated robot is a positive integer,
        if our_hm.tele_robotid > 0:
            # The agent only order robots other than the tele-operated robots.
            tele_robot_indx = our_hm.tele_robotid - 1  # Note python uses 0-based indexing.
            # We exclude the teleoperated robot.
            active_candi_robots = robot_tuple_list[:tele_robot_indx] + robot_tuple_list[tele_robot_indx + 1:]
        # If there is no robot under tele-operation, a.k.a, the id is -1.
        else:
            # All the robots are active.
            active_candi_robots = robot_tuple_list[:]
        # We will only consider the robots whose detection correctness probability >= 0.9 to be in the waiting line.
        active_robots = []
        # We will record the robots that are released before the robot ordering.
        released_robots_pre = []
        for active_candi in active_candi_robots:
            if active_candi[1] >= 0.9:
                active_robots += [active_candi]
            else:
                released_robots_pre += [active_candi]
        # +++++++++++++++++++++
        # Check the decision-making model here!
        if decision_model.lower == 'RTx'.lower:
            our_agent.ordering(robots=active_robots, human_cost_bs=our_hm_cost_bs, human_cost_co=our_hm_cost_co,
                               wrong_cost=our_env.cost_tar)
        elif decision_model.lower == 'EV'.lower:
            our_agent.ordering_ev(robots=active_robots, human_cost_bs=our_hm_cost_bs, human_cost_co=our_hm_cost_co,
                                  wrong_cost=our_env.cost_tar)
        else:
            our_agent.ordering_no_hm(robots=active_robots, human_cost_bs=our_hm_cost_bs, human_cost_co=our_hm_cost_co,
                                     wrong_cost=our_env.cost_tar)
        # Save the released robots before and after the ordering into one place.
        our_released_robots = our_agent.release_robots + released_robots_pre
        # We record the length of the waiting line.
        line_len_vec += [len(our_agent.wait_line)]
        # Now our_agent contains a waiting line of robots and a list of robots to be released. Each robot is
        # represented as a tuple (robot_id, post_prob).
        # ======================================================================================
        # The ordering divides the robots into three groups:
        # 1) the robot being tele-operated by the human,
        # 2) the robots released from the waiting line, and
        # 3) the robots in the waiting line.
        #
        # For the robot currently being tele-operated by the human,
        # If the human just finished helping one robot, a.k.a, time2go == 0, then the human starts to help a new robot.
        if our_hm.time2go == 1:
            # If the human was serving one robot, she does know that robot's id, which is greater than 0.
            # Otherwise, what she only has a nonsense id -1. We are about to update the information of the just served
            # robot. Of course, we only do it when there is one.
            if our_hm.tele_robotid > 0:
                # Record the human has finished one more detection.
                our_hm.hm_det_num += 1
                # Get the id of the just served robot.
                just_srv_id = our_hm.tele_robotid
                just_srv_indx = just_srv_id - 1  # Python indexing is 0-based.
                # Get the human observation from the environment.
                current_loc = our_rbt_set[just_srv_indx].current
                hm_observation = our_env.get_observation(robot_id=just_srv_id, sensor_model=our_hm.sensor_model,
                                                         cell_loc=current_loc)
                # Update the information of the just served robot in the list robot tuples.
                robot_tuple_list[just_srv_indx] = our_rbt_set[just_srv_indx].hm_detection(hm_observation)
            # If the line is not empty, the human then helps the first robot in the waiting line.
            if len(our_agent.wait_line) > 0:
                first_wait_tuple = our_agent.wait_line[0]  # Note first_wait is a tuple (robot_id, post_prob)
                # The human starts tele-operation.
                our_hm.start_tele(first_wait_tuple[0])
                # The waiting line updates to exclude the robot that gets human service.
                our_agent.wait_line = our_agent.wait_line[1:]
            # If the line is empty, we re-initialize the human.
            else:
                our_hm = Human(our_hm.hm_det_num)
                # print('The waiting line is empty at time step: {}'.format(time_step))
        # If the human is in the middle of helping one robot, then record the progress in this cycle.
        else:
            our_hm.progress()
        # ===========================================================================================================
        # For the robots that are released by the agent, they need plan and execute their next move. After moving to
        # the new cell, they need to take a new detection of that cell.
        # Make sure there are robots released by the agent.
        if len(our_released_robots) > 0:
            for robot_tuple in our_released_robots:
                # Moving:
                # Get the id of the robot.
                rel_robot_id = robot_tuple[0]
                # Get the index of this robot in the our robot set.
                rel_robot_indx = rel_robot_id - 1  # Note python using 0-based indexing.
                # We only do the following operations for the robots who has not finished searching its region.
                if not our_rbt_set[rel_robot_indx].finish_search:
                    # Confirm this robot is leaving the current cell.
                    our_rbt_set[rel_robot_indx].confirm()
                    # The robot paths its path.
                    our_rbt_set[rel_robot_indx].plan()
                    # # +++++++++++++++++++++++++
                    # # This section is used to plot the path of robot 1.
                    # if rel_robot_id == 1 and time_step % 10 == 0:
                    #     our_rbt_set[rel_robot_indx].show_path()
                    #     print('Print path for robot 1 at time step: {}'.format(time_step))
                    # # +++++++++++++++++++++++++
                    # The robot moves to the next cell now.
                    our_rbt_set[rel_robot_indx].move()
                    # Detecting:
                    # What is this robot's sensor model?
                    sensor_model = our_rbt_set[rel_robot_indx].sensor_model
                    # Where is this robot currently?
                    current_loc = our_rbt_set[rel_robot_indx].current
                    # What is this robot's observation in the environment?
                    new_obsv = our_env.get_observation(robot_id=rel_robot_id, sensor_model=sensor_model,
                                                       cell_loc=current_loc)
                    # Save the robot detection information into the robot tuple list, and updates the probabilities
                    # in the domain.
                    robot_tuple_list[rel_robot_indx] = our_rbt_set[rel_robot_indx].detection(observation=new_obsv)
        # Now the robot being released has done, planning, moving, and detecting.
        #
        # ===========================================================================================================
        # For the robots that are waiting in line, they stays at the current cell and takes a new detection
        # of that cell.
        if len(our_agent.wait_line) > 0:
            for robot_tuple in our_agent.wait_line:
                rel_robot_id = robot_tuple[0]
                # Get the index of this robot in the our robot set.
                rel_robot_indx = rel_robot_id - 1  # Note python using 0-based indexing.
                # We only do the following operations for the robots who has not finished searching its region.
                if not our_rbt_set[rel_robot_indx].finish_search:
                    # Detecting:
                    # What is this robot's sensor model?
                    sensor_model = our_rbt_set[rel_robot_indx].sensor_model
                    # Where is this robot currently?
                    current_loc = our_rbt_set[rel_robot_indx].current
                    # What is this robot's observation in the environment?
                    new_obsv = our_env.get_observation(robot_id=rel_robot_id, sensor_model=sensor_model,
                                                       cell_loc=current_loc)
                    # Save the robot detection information into the robot tuple list, and updates the probabilities
                    # in the domain.
                    robot_tuple_list[rel_robot_indx] = our_rbt_set[rel_robot_indx].detection(observation=new_obsv)

        # ===========================================================================================================
        # We now need to figure out how many targets have been found so far.
        total_found_tar = 0
        for robot_indx in range(num_all_robot):
            # We add the targets found by each robot.
            region_obvs = our_rbt_set[robot_indx].region_obvs
            region_tar_presence = np.where(region_obvs > 0, 1, 0)
            total_found_tar += sum(sum(region_tar_presence))
    # ==================================================================================================================
    # We want to check how many targets are actually found.
    actual_found_tar = 0
    for robot_indx in range(num_all_robot):
        # Let us get the locations where the robot observed a target presence.
        pos_obvs_locs = np.argwhere(our_rbt_set[robot_indx].region_obvs > 0)
        # We now check if at the locations there indeed are targets.
        for pos_loc_row in range(len(pos_obvs_locs)):
            # Get the locations of positive observations.
            pos_loc = pos_obvs_locs[pos_loc_row, :]
            # If at this location there indeed is a target, we get a hit.
            if our_env.domain_tar[pos_loc[0], pos_loc[1], robot_indx] > 0:
                actual_found_tar += 1
    # Detection performance is the number of actually found targets divided by the total number of missing targets.\
    str_1 = 'Decision model: {} \n'.format(our_agent.dm_model)
    str_2 = 'Time limit: {} \n'.format(total_time_step)
    str_3 = 'Human task duration: {} \n'.format(our_hm.hm_tsk_dur)
    str_4 = 'Wrong detection cost: {} \n'.format(our_env.cost_tar)
    str_5 = 'Time limit: {} \n'.format(total_time_step)
    str_6 = 'Sensor model: p_o1_s1 = {}, p_o1_s0 = {} \n'.format(our_robot.p_o1_s1, our_robot.p_o1_s0)
    str_7 = 'Claimed found:{}/{} \n'.format(total_found_tar, mis_tar)
    str_8 = 'Real found:{}/{} \n'.format(actual_found_tar, mis_tar)
    str_9 = 'Human utilization: {} \n'.format(our_hm.hm_det_num)
    str_10 = 'Time step: {} \n'.format(time_step)
    str_11 = 'Average length of waiting line: {} \n'.format(sum(line_len_vec)/len(line_len_vec))
    # Save these strings
    result_strs = [str_1, str_2, str_3, str_4, str_5, str_6, str_7, str_8, str_9, str_10, str_11]
    return result_strs


if __name__ == '__main__':

    # Define a file name to save the results.
    # Delete any existing result file
    filename = 'results/condition65_RTx.txt'
    # if os.path.isfile(filename):
    #     os.remove(filename)
    # Create a new file to store the result.
    with open(filename, 'a') as file:
        # Run the experiment for 20 times and store the data.
        for i in range(19):
            print(i)
            file.write('test {}: \n'.format(i))
            results = main()
            file.writelines(results)
            file.write('\n')
