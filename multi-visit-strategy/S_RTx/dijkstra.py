# Python program for Dijkstra's single
# source maximum utility algorithm. Here our utilities are all negative.

# The following is the pseudo-code.
#  1  function Dijkstra(Graph, source):
#  2
#  3      create vertex set Q
#  4
#  5      for each vertex v in Graph:
#  6          util[v] ← Negative INFINITY
#  7          prev[v] ← UNDEFINED
#  8          stage[v] ← 0
#  9          add v to Q
# 10      util[source] ← 0
# 11      destination ← vertex in Q with max localutil[v]
# 12      while Q is not empty:
# 13          u ← vertex in Q with max util[u]
# 14
# 15          remove u from Q
# 16
# 17          if u == destination:
# 18              break
# 19
# 20          for each neighbor v of u:           // only v that are still in Q
# 21              alt ← util[u] + localutil[v]
# 22
# 23              if alt > util[v]:
# 24                  util[v] ← alt
# 25                  prev[v] ← u
# 26                  stage[v] ← stage[u] + 1
# 27      return u, util[], prev[]


# Import lib.
import sys
import numpy as np
import matplotlib.pyplot as plt


class Graph:
    # Create the graph.
    # Description:
    # To define the graph, we will load a task-region and let each cell in the 2-D region to be a vertex in the graph.
    # We will use an implicit graph in which we only represent the vertices as a 2-D matrix.
    # The edges are implicit according to the following rules.
    # 1.  Any vertex not on the boundaries is connected to the 8 adjacent vertices: top, right-top, right, right-bottom,
    #     bottom, left-bottom, left, left-top.
    # 2.  Any vertex on the boundaries but not on the corners are connected to the 5 adjacent vertices, either on the
    # left or on the right.
    # 3.  Any vertex on the corners is connected to 3 adjacent vertices in a quadrant, either bottom-right, bottom-left,
    #     top-left, or top-right.
    def __init__(self, region_prob, start):
        self.region_prob = region_prob
        # The negative utilities are on the vertices rather than on the edges.
        # The local utility on a vertex is computed according to the following reasoning.
        #       If the next vertex does not have the target we are looking for, then the time spent there is wasted.
        #       utility == time_cost x prob( no target )
        # Note time_cost < 0 and probabilities can be retrieved from region_prob.
        # Define the time_cost as the wasted time.
        time_cost = -1
        # Define the matrix representing local utilities.
        self.local_util = (1 - region_prob) * time_cost
        # Find the row end and column end of the local_util matrix. These two ends will be used to define boundaries
        # and corners.
        # The four corners are (0, 0), (0, col_end-1), (row_end-1, 0), and (row_end-1, col_end-1).
        # The four boundaries are (0,:), (:, col_end-1), (row_end-1, :), and (:, 0).
        self.row_end = region_prob.shape[0]
        self.col_end = region_prob.shape[1]
        # Record the starting vertex of the planning
        self.start = start
        # Set the current vertex to be the start.
        self.current = start
        # We now implement the following piece of pseudo-code, which initialize the tentative cumulative utilities.
        #  5      for each vertex v in Graph:
        #  6          util[v] ← Negative INFINITY
        #  7          prev[v] ← UNDEFINED
        #  8           stage[v] ← 0
        #  9           add v to Q
        # 10      util[source] ← 0
        # But we use a matrix representation instead of a for loop.
        # Let us first initialize the cumulative utilities on each vertex.
        very_negative_number = -100000
        self.cum_util = np.ones(shape=region_prob.shape) * very_negative_number
        # Let us initialize an matrix to store row index of the tentative optimal previous vertex of the current vertex.
        self.prev_row_indx = np.ones(shape=region_prob.shape, dtype=int) * (-1)
        # Let us initialize an matrix to store column index of the tentative optimal previous vertex of the current
        # vertex.
        self.prev_col_indx = np.ones(shape=region_prob.shape, dtype=int) * (-1)
        # Let us initialize the planning stage so far on each vertex as 0.
        self.stage = np.zeros(shape=region_prob.shape, dtype=int)
        # Let the start point has 0 cumulative utility
        self.cum_util[start] = 0
        # Now we need to record whether a vertex is visited or not. This is represented as matrix.
        # If the vertex is NOT visited, its corresponding element is set to 0.
        # If the vertex is visited, its corresponding element is set to a very negative number.
        # Doing so enable us to rule out the visited vertices from vertex selection step in the dijkstra's algorithm,
        # which is based on utility maximization (step 13 in the above pseudo-code). You will see this later.
        # Initially, no vertex is visited, hence we create a matrix full of 0.
        self.visited = np.zeros(shape=region_prob.shape)
        # We also need to record how many vertex has been visited. It is used to keep track if all the vertices are
        # visited.
        self.num_visited = 0
        # Let us say we want to plan some steps ahead. This is the planning horizon.
        self.horizon = 5
        # We create an empty list for the shortest path.
        self.opt_path = []

    def _dijkstra(self):
        # This function deal with the action operations of dijkstra's algorithm. It corresponds to the following piece
        # of the above pseudo-code.
        # 11      destination ← vertex in Q with max localutil[u]
        # 12      while Q is not empty:
        # 13          u ← vertex in Q with min dist[u]
        # 14
        # 15          remove u from Q
        # 16
        # 17          if u == destination:
        # 18              break
        # 19
        # We first want to know the total number of the vertices.
        total_ver = self.local_util.size
        # We now find the destination vertex.
        desti = np.unravel_index(np.argmax(self.local_util), self.local_util.shape)
        # Here we start the big while-loop.
        while self.num_visited < total_ver:
            # We first want to find the index of the vertex with the maximum cumulative utility so far.
            # To do so, we must rule out the visited vertices. We create an auxiliary utility matrix in which the
            # visited vertices have very low cumulative utilities. This way, it will never be selected.
            # The matrix "visited" will do the job.
            aux_util = self.cum_util + self.visited
            # We want to the current vertex to work with. This current vertex is represented by its index (x, y).
            self.current = np.unravel_index(np.argmax(aux_util), aux_util.shape)
            # We now set the current vertex as visited and update the matrix "visit". It corresponds to step 15 in the
            # pseudo-code.
            very_negative_number = -10000
            self.visited[self.current] = very_negative_number
            # Don't forget to update the number of visited vertices.
            self.num_visited += 1
            # We now want to check the planning state to see if it reaches the preset destination. If it reaches,
            # we obtains our goal and break the while loop.
            if self.current == desti:
                break
            # Here come the most critical steps of the dijkstra's algorithm, step 20--25 in the pseudo-code.
            # We want to check every neighbors of the current vertex. The first thing to do is to find these neighbors.
            # We want to check if the current vertex is on the corners, on the boundaries, or in the region.
            # Let us start with check the top boundary.
            if self.current[0] < 1:
                if self.current[1] < 1:
                    # This is the corner (0, 0). It has three neighbors (0, 1), (1, 0), and (1, 1).
                    neighbors = [(0, 1),
                                 (1, 0), (1, 1)]
                elif self.current[1] > self.col_end - 2:
                    # This is the corner (0, col_end -1). It has three neighbors (0, col_end-2), (1, col_end-1), and
                    # (1, col_end-2).
                    neighbors = [(0, self.col_end - 2),
                                 (1, self.col_end - 2), (1, self.col_end - 1)]
                else:
                    # This is on the top boundary. It has five neighbors (0, current_col-1), (0, current_col+1),
                    # (1, current_col-1), (1, current_col), and (1, current_col+1)
                    neighbors = [(0, self.current[1] - 1), (0, self.current[1] + 1),
                                 (1, self.current[1] - 1), (1, self.current[1]), (1, self.current[1] + 1)]
            # Let us look at the bottom boundary
            elif self.current[0] > self.row_end - 2:
                if self.current[1] < 1:
                    # This is the corner (row_end - 1, 0). It has three neighbors (row_end - 1, 1), (row_end - 2, 0),
                    # and (row_end -2, 1).
                    neighbors = [(self.row_end - 2, 0), (self.row_end - 2, 1),
                                 (self.row_end - 1, 1)]
                elif self.current[1] > self.col_end - 2:
                    # This is the corner (row_end-1, col_end -1). It has three neighbors (row_end - 1, col_end - 2),
                    # (row_end - 2, col_end -2), and (row_end-2, col_end - 1)
                    neighbors = [(self.row_end - 2, self.col_end - 2), (self.row_end - 2, self.col_end - 1),
                                 (self.row_end - 1, self.col_end - 2), ]
                else:
                    # This is on the bottom boundary. It has five neighbors (row_end - 1, current_col - 1),
                    # (row_end - 2, current_col - 1), ( row_end - 2, current_col), (row_end - 2, current_col + 1),
                    # and (row_end -1, current_col + 1)
                    neighbors = \
                        [(self.row_end - 2, self.current[1] - 1), (self.row_end - 2, self.current[1]),
                         (self.row_end - 2, self.current[1] + 1),
                         (self.row_end - 1, self.current[1] - 1), (self.row_end - 1, self.current[1] + 1)]
            # Let us look at the other vertices other than on the top and bottom boundaries.
            else:
                # Let us check the left boundary first. Note this is no corner vertices on this boundary.
                if self.current[1] < 1:
                    # This is on the left boundary. It has five neighbors (current_row - 1, 0), (current_row - 1, 1),
                    # (current_row, 1), (current_row + 1, 1), and (current_row + 1, 0).
                    neighbors = [(self.current[0] - 1, 0), (self.current[0] - 1, 1),
                                 (self.current[0], 1),
                                 (self.current[0] + 1, 0), (self.current[0] + 1, 1)]
                elif self.current[1] > self.col_end - 2:
                    # This is on the right boundary. It has five neighbors (current_row - 1, col_end -1),
                    # (current_row - 1, col_end - 2), (current_row, col_end - 2), (current_row + 1, col_end - 2)
                    # and (current_row + 1, col_end - 1).
                    neighbors = [(self.current[0] - 1, self.col_end - 2), (self.current[0] - 1, self.col_end - 1),
                                 (self.current[0], self.col_end - 2),
                                 (self.current[0] + 1, self.col_end - 2), (self.current[0] + 1, self.col_end - 1)]
                else:
                    # These are the vertices not on the boundaries or corners. Each vertex has eight neighbor.
                    # I am tired to list them here in the comments. Please see the following code.
                    neighbors = \
                        [(self.current[0] - 1, self.current[1] - 1), (self.current[0] - 1, self.current[1]),
                         (self.current[0] - 1, self.current[1] + 1),
                         (self.current[0], self.current[1] - 1), (self.current[0], self.current[1] + 1),
                         (self.current[0] + 1, self.current[1] - 1), (self.current[0] + 1, self.current[1]),
                         (self.current[0] + 1, self.current[1] + 1)]
                    # Here we finish defining the neighbors.
            # With the neighbors of the current vertex in hand, let now implement the following important piece
            # of pseudocode:
            # 20          for each neighbor v of u:           // only v that are still in Q
            # 21              alt ← util[u] + localutil(v)
            # 22
            # 23              if alt > util[v]:
            # 24                  util[v] ← alt
            # 25                  prev[v] ← u
            # 26                  stage[v] ← stage[u] + 1
            # We create the for-loop here.
            # We will loop around all the neighbors which are still not visited.
            for neighbor in neighbors:
                # We first check if this neighbor is visited. Recall if a vertex is visited, its corresponding value in
                # the matrix "visited" is a very negative value.
                # We will only update the neighbor that is not visited yet.
                if self.visited[neighbor] > -10:
                    # We compute the alternative cumulative utility for this neighbor.
                    alt_cum_util = self.cum_util[self.current] + self.local_util[neighbor]
                    # Check if the alternative cumulative utility is larger than the current cumulative utility on this
                    # neighbor.
                    if alt_cum_util > self.cum_util[neighbor]:
                        self.cum_util[neighbor] = alt_cum_util
                        # Record the current vertex as the previous vertex of this neighbor in the shortest path.
                        self.prev_row_indx[neighbor] = self.current[0]
                        self.prev_col_indx[neighbor] = self.current[1]
                        # We update the planning stage for this neighbor. Because it moves one step ahead, we add 1.
                        self.stage[neighbor] = self.stage[self.current] + 1

    def max_util_path(self):
        # This function is going to find the shortest path.
        # It is coded according to the following pseudo-code.
        #
        # 1     S ← empty sequence
        # 2     u ← target
        # 3     if prev[u] is defined or u = source:      // Do something only if the vertex is reachable
        # 4         while u is defined:                   //Construct the shortest path with a stack S
        # 5           insert u at the beginning of S      // Push the vertex onto the stack
        # 6           u ← prev[u]                         // Travers from target to source
        #
        # We first the dijkstra's algorithm. This will give us the shortest path implicitly. Our job in this function is
        # to make the path explicit.
        self._dijkstra()
        # We have initialized self.opt_path as an empty python list. We add the target (the current vertex after
        # executing _dijkstra() ). This is step 2 in the pseudo-code.
        this_vrtx = self.current  # this_vrtx is "this vertex".
        # Check if the previous vertex of the current vertex is defined.
        if (self.prev_row_indx[this_vrtx] != -1 and self.prev_col_indx[this_vrtx] != -1) or this_vrtx == self.start:
            while this_vrtx[0] != -1 and this_vrtx[1] != -1:
                # Add this vertex to the optimal path.
                self.opt_path = [this_vrtx] + self.opt_path
                # Update this_vrtx to be the previous vertex
                this_vrtx = (self.prev_row_indx[this_vrtx], self.prev_col_indx[this_vrtx])
        return self.opt_path

    def show_path(self):
        # Plot the heatmap of the probabilities of this region.
        fig, ax = plt.subplots()
        # probmap = ax.imshow(self.local_util, cmap='YlOrBr', vmin=-1)
        probmap = ax.imshow(self.local_util, cmap='gray', vmin=-1.1, vmax=0)
        ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
        cbar = fig.colorbar(probmap)
        # Overlay with the plot of optimal path.
        x = [waypoint[0] for waypoint in self.opt_path]
        y = [waypoint[1] for waypoint in self.opt_path]
        ax.plot(y, x, 'r-o', markersize=4)  # make sure it is (y, x). plot function treats vertical axis (rows) as y.
        # Plot a start point.
        ax.plot(self.opt_path[0][1], self.opt_path[0][0], 'rs', markersize=8)
        # Plot an end point.
        ax.plot(self.opt_path[-1][1], self.opt_path[-1][0], 'r^', markersize=8)
        plt.show()


def main():
    # Load the probability information of the region.
    # Let us first choose a region.
    region_id = 1
    # We now load the domain probability information
    domain_prob = np.load("./data/domain_prob.npy")
    region_prob = domain_prob[:, :, region_id-1]
    # We create a graph for this region and run the dijkstra's algorithm.
    graph = Graph(region_prob, (29, 0))
    result = graph.max_util_path()
    graph.show_path()
    print("anything")


if __name__ == "__main__":
    main()
