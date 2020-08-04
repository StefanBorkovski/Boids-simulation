# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 11:30:17 2020

@author: Stefan Borkovski
"""

from p5 import setup, draw, size, background, run, circle, stroke, Vector
from collections import namedtuple
from operator import attrgetter
import matplotlib.pyplot as plt
import numpy as np
import time

##############################################################################
                          # Defining boids  #
class Boid():

    def __init__(self, x, y, width, height):
        
        """
        Function for boid initialization.
            
        Parameters
        ----------
        x : float
            x - COORDINATE.
        y : float
            y - COORDINATE.
        width : int
            width of the generated space.
        height : int
            height of the generated space.

        Returns
        -------
        None.
        
        """
        self.position = Vector(x, y)
        vec = (np.random.rand(2) - 0.5)*10 # generating random velocity vector
        self.velocity = Vector(*vec)
        self.x = x
        self.y = y

        vec = (np.random.rand(2) - 0.5)/2 # generating random acceleration vector
        self.acceleration = Vector(*vec)
        # initialization of speed and force limits
        self.max_force = 0.3
        self.max_speed = 5
        self.perception = 100

        self.width = width
        self.height = height
    
    def update(self):
        """
        Function for updating boid position and velocity.
        
        Returns
        -------
        None.

        """
        # change in position -> velocity
        self.position += self.velocity
        # change in celocity -> acceleration
        self.velocity += self.acceleration
        
        # if velocity magnitude is higher than the defined limit set the velocity 
        # magnitude to max speed
        if np.linalg.norm(self.velocity) > self.max_speed:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.max_speed
        
        # reset the acceleration
        self.acceleration = Vector(*np.zeros(2))

    def show(self):
        """
        Function for plotting the boid on canvas.

        Returns
        -------
        None.

        """
        stroke(0) # determine the color
        circle((self.position.x, self.position.y), radius=10) # creates a circle with defined radius


    def apply_behaviour(self, boids, boids_in_radius):
        """
        Function for applying the rules defined for simulation of boid effect.

        Parameters
        ----------
        boids : list of boid objects
            GENERATED FLOCK OF BIRDS.
        boids_in_radius : list of boid objects
            BOIDS IN RADIUS OF BOID OF INTEREST.

        Returns
        -------
        None.

        """
        avg_velocity, center_of_mass, avg_vector, total = self.compute_in_radius(boids_in_radius)
        
        alignment = self.align(boids, avg_velocity, total)
        cohesion = self.cohesion(boids, center_of_mass, total)
        separation = self.separation(boids, avg_vector, total)
        
        # acceleration ~ force (mass is equal for every boid)
        self.acceleration += alignment
        self.acceleration += cohesion
        self.acceleration += separation

    def edges(self):   
        """
        Function that keeps the birds/boids in the defined region.

        Returns
        -------
        None.

        """
                #    appearing on other side of the window    #
        
        # if boid gets out of the margins, it shows on the other side of the window
        
        if self.position.x > self.width:
            self.position.x = 0
        elif self.position.x < 0:
            self.position.x = self.width

        if self.position.y > self.height:
            self.position.y = 0
        elif self.position.y < 0:
            self.position.y = self.height
            
                #    bouncing from the window    #
                
        # if self.position.x > self.width or self.position.x < 0:
        #     self.velocity.x = -self.velocity.x
        # if self.position.y > self.height or self.position.y < 0: 
        #     self.velocity.y = -self.velocity.y

    def compute_in_radius(self, boids_in_radius):
        """
        Function for computing average velocity, direction and center of mass
        of boids in radius.

        Parameters
        ----------
        boids_in_radius : list of boid objects
            BOIDS IN RADIUS OF BOID OF INTEREST.

        Returns
        -------
        avg_velocity : Vector
            AVERAGE DIRECTION OF BOIDS IN RADIUS.
        center_of_mass : Vector
            CENTER OF MASS OF BOIDS IN RADIUS.
        avg_vector : Vector
            OPOSITE OF THE AVERAGE DIRECTION OF BOIDS IN RADIUS, SCALED WITH 
            RESPECT TO THE BOIDS DISTANCE.
        total : int
            TOTAL NUMBER OF BOIDS IN RADIUS.

        """
        
        avg_velocity = Vector(*np.zeros(2))
        center_of_mass = Vector(*np.zeros(2))
        avg_vector = Vector(*np.zeros(2))
        total = 0
        for boid in boids_in_radius:
            avg_velocity += boid.velocity # calculating average direction 
            center_of_mass += boid.position # calculating center of mass
            total += 1
            distance = np.linalg.norm(boid.position - self.position)
            
            if self.position != boid.position:
                diff = self.position - boid.position
                diff /= distance # scaling with the distance in order to avoid closer boids with greater force 
                avg_vector += diff # calculating repulsive force vector
                
        return avg_velocity, center_of_mass, avg_vector, total
        
    def align(self, boids, avg_vector, total):
        """
        Function for aligning the boid of interest with the average direction 
        of boids in radius.

        Parameters
        ----------
        boids : list of boid objects
            GENERATED FLOCK OF BIRDS.
        avg_vector : Vector
            AVERAGE DIRECTION OF BOIDS IN RADIUS.
        total : int
            TOTAL NUMBER OF BOIDS IN RADIUS.

        Returns
        -------
        steering : Vector
            VECTOR THAT DRIVES THE BOID TOWARD AVERAGE DIRECTION OF BOIDS IN 
            RADIUS.

        """
        steering = Vector(*np.zeros(2))
        
        if total > 0:
            avg_vector /= total
            avg_vector = Vector(*avg_vector)
            avg_vector = (avg_vector / np.linalg.norm(avg_vector)) * self.max_speed
            steering = avg_vector - self.velocity # calculating force that steers the boid toward average direction

        return steering

    def cohesion(self, boids, center_of_mass, total):
        """
        Function for steering the boid toward the center of mass of boids in 
        radius.

        Parameters
        ----------
        boids : list of boid objects
            GENERATED FLOCK OF BIRDS.
        center_of_mass : Vector
            CENTER OF MASS OF BOIDS IN RADIUS.
        total : int
            TOTAL NUMBER OF BOIDS IN RADIUS.

        Returns
        -------
        steering : Vector
            VECTOR THAT DRIVES THE BOID TOWARD CENTER OF MASS OF BOIDS IN 
            RADIUS.

        """
        
        steering = Vector(*np.zeros(2))
                
        if total > 0:
            center_of_mass /= total
            center_of_mass = Vector(*center_of_mass)
            vec_to_com = center_of_mass - self.position
            if np.linalg.norm(vec_to_com) > 0:
                vec_to_com = (vec_to_com / np.linalg.norm(vec_to_com)) * self.max_speed
            steering = vec_to_com - self.velocity # calculating force that steers the boid toward center of mass
            if np.linalg.norm(steering) > self.max_force:
                steering = (steering /np.linalg.norm(steering)) * self.max_force

        return steering

    def separation(self, boids, avg_vector, total_pom):
        """
        Function for steering the boid away from closer boids in order to avoid 
        colision.

        Parameters
        ----------
        boids : list of boid objects
            GENERATED FLOCK OF BIRDS.
        avg_vector : Vector
            OPOSITE OF THE AVERAGE DIRECTION OF BOIDS IN RADIUS, SCALED WITH 
            RESPECT TO THE BOIDS DISTANCE.
        total_pom : int
            TOTAL NUMBER OF BOIDS IN RADIUS.

        Returns
        -------
        steering : Vector
            VECTOR THAT DRIVES THE BOID AWAY FROM THE BOIDS IN RADIUS.

        """
        
        total = total_pom - 1
        steering = Vector(*np.zeros(2))
        
        if total > 0:
            avg_vector /= total
            avg_vector = Vector(*avg_vector)
            if np.linalg.norm(avg_vector) > 0:
                avg_vector = (avg_vector / np.linalg.norm(avg_vector)) * self.max_speed
            steering = avg_vector - self.velocity # calculating force that steers the boid from neigbours with 
                                                  # respect to neighbour distance
            if np.linalg.norm(steering) > self.max_force:
                steering = (steering /np.linalg.norm(steering)) * self.max_force

        return steering

##############################################################################
                        # Kd-tree construction #
                        
Node = namedtuple('Node', 'location left_child right_child')
 
def kdtree_modified(point_list, depth=0):
    """
    Function for constructing kd-tree with boid objects.

    Parameters
    ----------
    point_list : list of boid objects.
         GENERATED FLOCK OF BIRDS.
    depth : int
         PARAMETER THAT SHOWS THE DEPTH OF THE TREE. The default
         (starting depth) is 0.

    Returns
    -------
    namedtuple
        RETURNS TREE NODE.

    """

    # assumes all points have the same dimension
    try:
        k = len(point_list[0].position) - 1
    except IndexError:
        return None
    # there is exception because at the end the list will get empty
    
    # Select axis based on depth so that axis cycles through all valid values
    axis = depth % k        # changing between 0/1
    
    # Sort point list and choose median as pivot element
    attribute = 'x' if axis == 0 else 'y'
    point_list.sort(key = attrgetter(attribute), reverse = False)
    median = len(point_list) // 2         # choose median
    # '//' divide with integral result
 
    # Create node and construct subtrees
    return Node(
        location=point_list[median],
        left_child=kdtree_modified(point_list[:median], depth + 1),
        right_child=kdtree_modified(point_list[median + 1:], depth + 1)
    )

##############################################################################
                        # Plotting the 2d-tree #
 
# line width for visualization of K-D tree
line_width = [4., 3.5, 3., 2.5, 2., 1.5, 1., .5, 0.3]
 
def plot_tree(tree, min_x, max_x, min_y, max_y, prev_node, branch, depth=0):
    """
    Function for visualization of the kd-tree.

    Parameters
    ----------
    tree : namedtuple
        INPUT TREE TO BE PLOTED.
    min_x : int
        STARTING POINT OF DEFINED REGION ON x - AXIS.
    max_x : int
        ENDING POINT OF DEFINED REGION ON x - AXIS.
    min_y : int
        STARTING POINT OF DEFINED REGION ON y - AXIS.
    max_y : int
        ENDING POINT OF DEFINED REGION ON y - AXIS.
    prev_node : NAMEDTUPLE
        PARENT'S NODE OF CURRENT NODE.
    branch : NAMEDTUPLE
        TRUE IF LEFT, FLASE IF RIGHT.
    depth : int
         PARAMETER THAT SHOWS THE DEPTH OF THE TREE. The default
         (starting depth) is 0.

    Returns
    -------
    namedtuple
    RETURNS TREE NODE.

    """
 
    cur_node = tree.location         # current tree's node
    left_branch = tree.left_child    # its left branch
    right_branch = tree.right_child  # its right branch
 
    # set line's width depending on tree's depth
    if depth > len(line_width)-1:
        ln_width = line_width[len(line_width)-1]
    else:
        ln_width = line_width[depth]
 
    k = len(cur_node.position) - 1 # k = 2
    axis = depth % k
 
    # draw a vertical splitting line
    if axis == 0:
 
        if branch is not None and prev_node is not None:
 
            if branch:
                max_y = prev_node[1]
            else:
                min_y = prev_node[1]
 
        plt.plot([cur_node.position[0],cur_node.position[0]], [min_y,max_y], linestyle='-', color='red', linewidth=ln_width)
 
    # draw a horizontal splitting line
    elif axis == 1:
 
        if branch is not None and prev_node is not None:
 
            if branch:
                max_x = prev_node[0]
            else:
                min_x = prev_node[0]
 
        plt.plot([min_x,max_x], [cur_node.position[1],cur_node.position[1]], linestyle='-', color='blue', linewidth=ln_width)
 
    # draw the current node
    plt.plot(cur_node.position[0], cur_node.position[1], 'ko')
 
    # draw left and right branches of the current node
    if left_branch is not None:
        plot_tree(left_branch, min_x, max_x, min_y, max_y, cur_node.position, True, depth+1)
 
    if right_branch is not None:
        plot_tree(right_branch, min_x, max_x, min_y, max_y, cur_node.position, False, depth+1)

##############################################################################
            # Modified nearest neighbors in defined radius #
  
nearest_nn = None           # nearest neighbor (NN)
distance_nn = float('inf')  # distance from NN to target
radius = 10000
in_range = []                
  
def nearest_neighbor_search_radius_modified(tree, target_point, hr, distance, nearest=None, depth=0):
    """
    Function for finding the boids in radius of the boid of interest.

    Parameters
    ----------
    tree : namedtuple
        INPUT TREE TO BE PLOTED.
    target_point : Boid
        BOID OF INTEREST.
    hr : tuple
        SPLITTING HYPERPLANE.
    distance : float
        MINIMAL DISTANCE.
    nearest : Boid
        NEAREST BOID FOUND. The default is None.
    depth : int
         PARAMETER THAT SHOWS THE DEPTH OF THE TREE. The default
         (starting depth) is 0.

    Returns
    -------
    in_range : LIST
        LIST OF BOIDS IN RADIUS OF THE BOID OF INTEREST.

    """
    
    global nearest_nn
    global distance_nn
 
    if tree is None:
        return 
    # at the end the whole tree is pruned - None
    
    k = len(target_point.position) - 1 # k = 2
 
    cur_node = tree.location         # current tree's node
    left_branch = tree.left_child    # its left branch
    right_branch = tree.right_child  # its right branch
 
    nearer_kd = further_kd = None
    nearer_hr = further_hr = None
    left_hr = right_hr = None
 
    # Select axis based on depth so that axis cycles through all valid values
    axis_pom = depth % k
    axis = 'x' if axis_pom == 0 else 'y'
    
    # hr = [(min_val-delta, max_val+delta), (max_val+delta, min_val-delta)]  # initial splitting plane
    #    = [(-2, 22), (22, -2)]
    
    # split the hyperplane depending on the axis
    if axis == 'x':
        left_hr = [hr[0], (cur_node.position[0], hr[1][1])]
        right_hr = [(cur_node.position[0],hr[0][1]), hr[1]]
 
    if axis == 'y':
        left_hr = [(hr[0][0], cur_node.position[1]), hr[1]]
        right_hr = [hr[0], (hr[1][0], cur_node.position[1])]
 
    # check which hyperplane the target point belongs to
        # if the target_point is on the left/bottom side
    if target_point.position[axis_pom] <= cur_node.position[axis_pom]:
        nearer_kd = left_branch # closer sub-tree is the left/bottom_branch
        further_kd = right_branch # further sub-tree is the right/top_branch
        nearer_hr = left_hr # closer hyperplane is the left/bottom_hyperplane
        further_hr = right_hr # futher hyperplane is the right/top_hyperplane
        
        # if the target_point is on the right/top side
    if target_point.position[axis_pom] > cur_node.position[axis_pom]:
        nearer_kd = right_branch
        further_kd = left_branch
        nearer_hr = right_hr
        further_hr = left_hr
 
    # check whether the current node is closer
    # print("curr node", cur_node)      #test
    # print("targ node", target_point)
    dist = (cur_node.position[0] - target_point.position[0])**2 + (cur_node.position[1] - target_point.position[1])**2
 
    if dist < distance:
        nearest = cur_node
        distance = dist

    if dist < radius: # and all([i != j for i, j in zip(cur_node, target_point)]):
        in_range.append(cur_node)
        
    # go deeper in the tree, pass the sub-tree and hyperplane in which the target_point bellow,
    # pass current best distance and closest node, increase the depth 
    nearest_neighbor_search_radius_modified(nearer_kd, target_point, nearer_hr, distance, nearest, depth+1)
 
    # once we reached the leaf node we check whether whether we found closer points inside the hypersphere
    if distance < distance_nn:
        nearest_nn = nearest
        distance_nn = distance
 
    # a nearer point (px,py) could only be in further_kd (further_hr) -> explore it
    px = compute_closest_coordinate(target_point.position[0], further_hr[0][0], further_hr[1][0])
    py = compute_closest_coordinate(target_point.position[1], further_hr[1][1], further_hr[0][1])
 
    # check whether it is closer than the current nearest neighbor => whether a hypersphere crosses the hyperplane
    dist = (px - target_point.position[0])**2 + (py - target_point.position[1])**2
 
    # explore the further kd-tree / hyperplane if necessary
    if radius > distance_nn: 
        check_dist = radius
    else:
        check_dist = distance_nn
        
    if dist < check_dist:
        nearest_neighbor_search_radius_modified(further_kd, target_point, further_hr, distance, nearest, depth+1)
    
    return in_range

##############################################################################
                     # Compute closes coordinate #

def compute_closest_coordinate(value, range_min, range_max):
    """
    Function for computing closest coordinate for the neighboring hyperplane.

    Parameters
    ----------
    value : float
        COORDINATE VALUE (x or y) OF THE TARGET POINT.
    range_min : float
        MINIMAL COORDINATE (x or y) OF THE NEGHBORING HYPERPLANE.
    range_max : float
        MAXIMAL COORDINATE (x or y) OF THE NEGHBORING HYPERPLANE..

    Returns
    -------
    v : float
        x or y coordinate.

    """
    
    v = None
 
    if range_min < value < range_max:
        v = value
 
    elif value <= range_min:
        v = range_min
 
    elif value >= range_max:
        v = range_max
 
    return v

##############################################################################
                       # Defining the canvas #
                        
def setup():
    """
    Function that define the size of the canvas.

    Returns
    -------
    None.

    """
    #this happens just once
    size(width, height) #instead of create_canvas
    
time_vec = []

##############################################################################
                  # Drawing the boids for simulation #
                  
def draw():
    """
    Function that is executing in a loop when the program is started. Here all 
    functions are called in order to generate list of boids, plot the boids on
    canvas, create a tree from boids, find neighbours, make calculations, 
    update every boid positions and plot the boids with updated positions. 

    Returns
    -------
    None.

    """
    global flock

    start = time.time()
    background(30, 30, 47)
    
    tree = kdtree_modified(flock)
        
    delta = 10 # extension of the range
    min_val = 0
    max_val_x = width
    max_val_y = height

                # Ploting the tree structure between iterations #
                
    # plot_tree(tree, min_val-delta, max_val_x+delta, min_val-delta, max_val_y+delta, None, None)
    # plt.title('K-D Tree')
    # plt.show()
    
                   # applying flocking rules to every boid #            
    
    for boid in flock:
 
        global in_range
        in_range = []
        
        hr = [(min_val-delta, max_val_x+delta), (max_val_y+delta, min_val-delta)]  # initial splitting plane 
        max_dist = float('inf')
        
        boids_in_range = nearest_neighbor_search_radius_modified(tree, boid, hr, max_dist)

        boid.edges()
        boid.apply_behaviour(flock, boids_in_range)
        boid.update()
        boid.show()
        
    end = time.time()
    time_vec.append(end-start)    
    avg_time = np.sum(time_vec)/len(time_vec)
    print("Average simulation-time needed to this point: ", avg_time)

##############################################################################
                          # Initialization #
                          
width = 1400
height = 700
number_of_boids = 60

bird_number = input("Pass the number of birds: ")

if bird_number != None:
    flock = [Boid(*(np.random.rand(1)*width, np.random.rand(1)*height), width, height) for _ in range(int(bird_number))] 
    run()



