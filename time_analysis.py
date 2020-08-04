# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:10:09 2020

@author: Stefan Borkovski
"""

##############################################################################
              # Time analysis with increasing number of birds #
            
import matplotlib.pyplot as plt
number_of_birds = (20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300)
req_time_modified = (0.033, 0.071, 0.11, 0.16, 0.22, 0.26, 0.33, 0.4, 0.46, 0.54, 0.6, 0.69, 0.72, 0.82, 0.89)
req_time_bruteforce = (0.078, 0.26, 0.57, 0.97, 1.5, 2, 2.8, 3.6, 4.65, 5.93, 7.2, 8.56, 9.8, 11.2, 12.8)
plt.plot(number_of_birds, req_time_modified)
plt.plot(number_of_birds, req_time_bruteforce)
plt.title("Time Complexity")
plt.xlabel('Number of birds')
plt.ylabel('Calculation time required per step [s]')
plt.legend(['implementation with kd-tree','brute force implementation'])
plt.show()