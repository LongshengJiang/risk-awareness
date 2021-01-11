# This python file optimally queue the robots.


# We define the function to determine the length of the waiting line of robots.
# This function is programmed according to the following pseudo-code.
#
# Input: a set of tuple (robot_id, post_prob) of all the robots.
# 1    lined_tuple <-- Ordered tuples (robot_id, post_prob) with post_prob ascending.
# 2    line_len <-- total amounts of the tuples (robot_id, post_prob)
# 3    while True do
# 4         num_wait <-- 0  // num_wait is the number of waiting robots.
# 5         for post_prob in lined_tuple
# 6             compute net_adv with line_len
# 7
# 8             if net_adv > 0 then
# 9                 num_wait <-- num_wait + 1
# 10        if line_len <= num_wait then
# 11            break
# 12        line_len <-- line_len - 1
# 13
# 14  lined_tuple_trunc <-- the first line_len in lined_tuple
#

# First thing first, let us import some good looking libraries.
import math
import random


class OptimalOrder:
    def __init__(self, hum_cost_bs=0., hum_cost_co=-1., wrong_det_cost=-50.):
        # Initialize two empty lists for the waiting robots and the released robots.
        self.robot_list_wait = []
        self.robot_list_release = []
        # Let prepare the some constants.
        # Set the human cost base
        self.hum_cost_bs = hum_cost_bs
        # Set the human cost coefficient
        self.hum_cost_co = hum_cost_co
        # Set the cost of wrong detection
        self.wrong_cost = wrong_det_cost

    def _line_length(self, robot_list):
        # This function determines the length of the waiting line.
        # robot_list is a list of tuple inf the form of (robot_id, post_prob).
        # We reorder the tuples in robot_list such that post_prob is ascending. That is we order the list by the second
        # element in the tuples.
        robot_list.sort(key=lambda x: x[1], reverse=False)  # This is ascending.
        # Initialize the length of the line to contain all the robots.
        line_len = len(robot_list)
        # Here we will implement step 3 of the pseudo-code.
        while True:
            # Initialize the number of robots who want to wait as 0.
            num_wait = 0
            # For the robots in the robot_list, we want to check its net advantage (net_adv) according to the extended
            # regret theory.
            for robot in robot_list:
                # Start from here, we want to compute the net_adv.
                # Let us retrieve the posterior probability of this robot.
                pr = robot[1]
                # We need to use the q-function and w-function to do this computation. Let me copy them here for
                # convenience.
                #               q = 8.541e-09 * delta_u + 1.305 * sinh(2.388 * delta_u);
                #               w = 0.0575 + 0.8853 * exp(-3.26 * (-math.log(prob)) ^ 5.215);
                # The parameters are from subject YG. Her w-function is inverted-S shaped.
                # Let us first focus on the q-function. We want to get its argument (delta_u) ready.
                delta_u_correct = (self.hum_cost_bs + self.hum_cost_co * line_len) / abs(self.wrong_cost)
                delta_u_wrong = delta_u_correct + 1
                # Let us compute the q-values for the q-functions.
                q_func = lambda delta_u: 8.541e-09 * delta_u + 1.305 * math.sinh(2.388 * delta_u)
                q_value_correct = q_func(delta_u_correct)
                q_value_wrong = q_func(delta_u_wrong)
                # Let us now focus on the w-function
                w_func = lambda prob: 0.0575 + 0.8853 * math.exp(-3.26 * (-math.log(prob)) ** 5.215)
                if pr == 0.0:
                    w_value = 0.0
                elif pr == 1.0:
                    w_value = 1.0
                else:
                    w_value = w_func(pr)
                # With all the components ready, we now can compute the ned_adv according to the extended regret theory.
                net_adv = w_value * q_value_correct + (1 - w_value) * q_value_wrong
                # Well done! we have finished computing the net_adv, the hardiest part of this code.
                # We now work on the next step, step 8 in the pseud-code.
                if net_adv > 0:
                    num_wait += 1
            # We want to check if the number of robots who want to wait is just enough for the capacity of the waiting
            # line.
            if line_len <= num_wait:
                break
            # When the length of the line is longer than the amount of robots who want to wait, we need to cut the line
            # shorter.
            line_len -= 1
        # When we jump out of the while-loop, we are promised to have the line length we desire.
        self.robot_list_wait = robot_list[0:line_len]
        self.robot_list_release = robot_list[line_len:]

    def line_length_ev(self, robot_list):
        # This function determines the length of the waiting line.
        # robot_list is a list of tuple inf the form of (robot_id, post_prob).
        # We reorder the tuples in robot_list such that post_prob is ascending. That is we order the list by the second
        # element in the tuples.
        robot_list.sort(key=lambda x: x[1], reverse=False)  # This is ascending.
        # Initialize the length of the line to contain all the robots.
        line_len = len(robot_list)
        # Here we will implement step 3 of the pseudo-code.
        while True:
            # Initialize the number of robots who want to wait as 0.
            num_wait = 0
            # For the robots in the robot_list, we want to check its net advantage (net_adv) according to the extended
            # regret theory.
            for robot in robot_list:
                # Start from here, we want to compute the net_adv.
                # Let us retrieve the posterior probability of this robot.
                pr = robot[1]
                # We now can compute the ned_adv according to the Expected Value.
                net_adv = (self.hum_cost_bs + self.hum_cost_co * line_len) - (1 - pr) * self.wrong_cost
                # Well done! we have finished computing the net_adv, the hardiest part of this code.
                # We now work on the next step, step 8 in the pseud-code.
                if net_adv > 0:
                    num_wait += 1
            # We want to check if the number of robots who want to wait is just enough for the capacity of the waiting
            # line.
            if line_len <= num_wait:
                break
            # When the length of the line is longer than the amount of robots who want to wait, we need to cut the line
            # shorter.
            line_len -= 1
        # When we jump out of the while-loop, we are promised to have the line length we desire.
        self.robot_list_wait = robot_list[0:line_len]
        self.robot_list_release = robot_list[line_len:]

    def line_order(self, robot_list):
        # This function is used to order the waiting line, which has a known length after running the
        # function line_length().

        # This function is coded according to the following pseudo-code.
        #
        # 1  robot_list_wait <-- calling function _line_length()
        # 2  line_thshd <-- int of wrong_cost/(2 * hum_cost_co)
        # 3  if line_thshd < line_len - 1 then
        # 4      robot_list_reorder <-- reordering probabilities from the line_thshd-th to the line_len-th in descending
        # 5                             fashion.
        # 6  Output robot_list_reorder
        #
        # Let us implement this pseudo-code.
        # We first call the function _line_len()
        self._line_length(robot_list)
        # Let us get the length of the current waiting line.
        line_len = len(self.robot_list_wait)
        # Now we have the robot_list_wait. Next, we compute the threshold position (line_thshd).
        line_thshd = math.floor(self.wrong_cost / (2 * self.hum_cost_co))
        # The robots before and after this threshold will use opposite ordering strategies.
        # We need to check there are at least more than 2 robots after this threshold.
        if line_thshd < line_len - 1:  # There are at least more than 2 robots after this threshold.
            # We separate the original robot_list_wait into 2 parts, divided by the threshold.
            robot_list_wait_asc = self.robot_list_wait[:line_thshd]
            robot_list_wait_dsc = self.robot_list_wait[line_thshd:]  # This is not in descending order yet.
            # We now reorder the trailing part of the robot list.
            robot_list_wait_dsc.sort(key=lambda x: x[1], reverse=True)  # This is descending.
            # Lastly, we need to put these two parts back together. Here "+" is used to merge two python lists.
            self.robot_list_wait = robot_list_wait_asc + robot_list_wait_dsc
        # --------------------------------------------------------------------------------------------------------------
        # The following two lines are just for providing baseline in the experiments. Normally they should be commented.
        # self.robot_list_wait.reverse()  # The waiting line is in the reverse order with respect to the optimal ordering.
        # random.shuffle(self.robot_list_wait)  # The waiting line is randomly shuffled.
        return self


def testing():
    robots = [(1, 0.9001370116252254), (2, 0.9208374320593732), (3, 0.9279141590431638),
              (4, 0.92361518190418204), (5, 0.9984344156937324), (6, 0.9463566440838444),
              (7, 0.9594430208524582), (8, 0.9443470253619494), (9, 0.9941760946617855),
              (10, 0.910898564413953), (11, 0.92671503953632284), (12, 0.97393429000667929),
              (13, 0.9683681785237046), (14, 0.93041881097751014), (15, 0.9707135543788413)]
    opt_queue = OptimalOrder()
    opt_queue.line_order(robots)
    print(opt_queue.robot_list_wait)
    print(opt_queue.robot_list_release)


if __name__ == '__main__':
    testing()
