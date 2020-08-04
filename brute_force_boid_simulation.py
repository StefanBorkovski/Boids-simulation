# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 13:50:27 2020

@author: Stefan Borkovski
"""

from p5 import setup, draw, size, background, run, Vector, stroke, circle
import numpy as np
import time

class Boid():

    def __init__(self, x, y, width, height):
        self.position = Vector(x, y)
        vec = (np.random.rand(2) - 0.5)*10
        self.velocity = Vector(*vec)
        self.x = x
        self.y = y
        
        vec = (np.random.rand(2) - 0.5)/2
        self.acceleration = Vector(*vec)
        self.max_force = 0.3
        self.max_speed = 5
        self.perception = 100

        self.width = width
        self.height = height

    def fprint(self):
        print("Position: ", self.position)
        print("Velocity: ", self.velocity)
        print("Acceleration ", self.acceleration)

    def update(self):
        self.position += self.velocity
        self.velocity += self.acceleration
        #limit
        if np.linalg.norm(self.velocity) > self.max_speed:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.max_speed

        self.acceleration = Vector(*np.zeros(2))

    def show(self):
        stroke(0)

        circle((self.position.x, self.position.y), radius=10)


    def apply_behaviour(self, boids):
        alignment = self.align(boids)
        cohesion = self.cohesion(boids)
        separation = self.separation(boids)

        self.acceleration += alignment
        self.acceleration += cohesion
        self.acceleration += separation

    def edges(self):   
        
                ####    appearing on other side of the window    ####
        if self.position.x > self.width:
            self.position.x = 0
        elif self.position.x < 0:
            self.position.x = self.width

        if self.position.y > self.height:
            self.position.y = 0
        elif self.position.y < 0:
            self.position.y = self.height
            
                ####    bouncing from the window    ####
        # if self.position.x > self.width or self.position.x < 0:
        #     self.velocity.x = -self.velocity.x
        # if self.position.y > self.height or self.position.y < 0: 
        #     self.velocity.y = -self.velocity.y

    def align(self, boids):
        steering = Vector(*np.zeros(2))
        total = 0
        avg_vector = Vector(*np.zeros(2))
        for boid in boids:
            if np.linalg.norm(boid.position - self.position) < self.perception:
                avg_vector += boid.velocity
                total += 1
        if total > 0:
            avg_vector /= total
            avg_vector = Vector(*avg_vector)
            avg_vector = (avg_vector / np.linalg.norm(avg_vector)) * self.max_speed
            steering = avg_vector - self.velocity

        return steering

    def cohesion(self, boids):
        steering = Vector(*np.zeros(2))
        total = 0
        center_of_mass = Vector(*np.zeros(2))
        for boid in boids:
            if np.linalg.norm(boid.position - self.position) < self.perception:
                center_of_mass += boid.position
                total += 1
        if total > 0:
            center_of_mass /= total
            center_of_mass = Vector(*center_of_mass)
            vec_to_com = center_of_mass - self.position
            if np.linalg.norm(vec_to_com) > 0:
                vec_to_com = (vec_to_com / np.linalg.norm(vec_to_com)) * self.max_speed
            steering = vec_to_com - self.velocity
            if np.linalg.norm(steering)> self.max_force:
                steering = (steering /np.linalg.norm(steering)) * self.max_force

        return steering

    def separation(self, boids):
        steering = Vector(*np.zeros(2))
        total = 0
        avg_vector = Vector(*np.zeros(2))
        for boid in boids:
            distance = np.linalg.norm(boid.position - self.position)
            if self.position != boid.position and distance < self.perception:
                diff = self.position - boid.position
                diff /= distance
                avg_vector += diff
                total += 1
        if total > 0:
            avg_vector /= total
            avg_vector = Vector(*avg_vector)
            if np.linalg.norm(avg_vector) > 0:
                avg_vector = (avg_vector / np.linalg.norm(avg_vector)) * self.max_speed
            steering = avg_vector - self.velocity
            if np.linalg.norm(steering) > self.max_force:
                steering = (steering /np.linalg.norm(steering)) * self.max_force

        return steering

##############################################################################
                         # Initialization #

width = 1400
height = 700

def setup():
    #this happens just once
    size(width, height) #instead of create_canvas

# vispy.use(app='Glfw')
time_vec = []
def draw():
    start = time.time()
    global flock

    background(30, 30, 47)

    for boid in flock:

        boid.edges()
        boid.apply_behaviour(flock)
        boid.update()
        boid.show()

    end = time.time()
    time_vec.append(end-start)
    print("Average simulation-time needed to this point: ", np.sum(time_vec)/len(time_vec) )

bird_number = input("Pass the number of birds: ")

if bird_number != None:
    flock = [Boid(*(np.random.rand(1)*width, np.random.rand(1)*height), width, height) for _ in range(int(bird_number))] 
    run()

