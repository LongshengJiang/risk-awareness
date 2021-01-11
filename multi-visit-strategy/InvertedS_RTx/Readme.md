### Requirement
- The program is run in python 3.5 
- The RTx decision-making model uses inverted S shaped probability weighting function
- The detection uses multi-visit strategy with the confidence threshold being 0.9.

### Package structure
<pre>
|_  data\
|_  results\
|_  main.py  
|         |_  dijkstra.py  
|         |_  optimal_order.py
|_  task_region.py
|_  task_region_spot.py
</pre>
### Function explanation
- The folder `data\` is used to contain the information of the search domain. The domain is where the human-robot team does the search task. There is a variant `data_1161\` which is a backup folder containing specific domain information. (In this case, the domain contains 1161 target objects.)

- The folder `results\` is used to contain the simulation results.
- The program `main.py` is the main simulation code, and `dijkstra.py` and `optimal_order.py` are the subroutines. `dijkstra.py` is the function of robot path planning for each robot at each moment. `optimal_order.py` is used to order the robots into a waiting line. 
- The program `task_region.py` is used to create the search domain. The probability of target presence is completely generated randomly. Its output is saved in `data\`. Often, we want to test different experimental conditions in the same domain. Hence, this program `task_region.py` is run only one time. 
- The program `task_region_spot.py` is a variant of `task_region.py`. In this variant, the probabilities concentrate on several randominly chosen spots rather than completely random.  

### Change experimental conditions.
Because this is a experimental program, my goal was to get it run instead of making it user friendly. We will need to do some tedious job in order to change experimental conditions. 

In `main.py`, we can make changes to the following places.  
1. `class Robot`: We can change the sensor model in the following code

          # Create a sensor model for the robot detection.
          self.p_o1_s1 = 0.9  # Probability of true positive.
          self.p_o1_s0 = 0.4  # Probability of false positive.

2. `class Human`: We can change the unit human detection duration (detection cost) in the following code

          # Let us first determine how many times slower human is.
          self.hm_tsk_dur = 20  # human task duration is several times slower than that of the robot detection.

3. `class environment`: We can change the wrong detection cost in the following code.
                 
          # Initialize the cost of losing one target
          self.cost_tar = -100

4. `def main()`: We can change decision models in the following code.

          # Set the decision-making model here.
          decision_model = 'RTx'  # choose 'No human', 'EV', or 'RTx'
 
5. `if __name__ == '__main__'`: We need specify a different condition name in the following code

            # Define a file name to save the results.
            # Delete any existing result file
            filename = 'results/condition65_RTx.txt'
            
 ### Change ordering strategies
 We can also use different ordering strategies by changing two line of code in `optimal_order.py`
 
 1. `def line_order(self, robot_list)`: Find the following code
         
           # self.robot_list_wait.reverse()  # The waiting line is in the reverse order with respect to the optimal ordering.
           # random.shuffle(self.robot_list_wait)  # The waiting line is randomly shuffled.
  When commenting out the two lines, the ordering is optimal ordering. When releasing one line, we change to the cooresponding ordering strategy. 
