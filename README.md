## What is boids-simulation?

As with most artificial life simulations, Boids is an example of emergent behavior; that is, the complexity of Boids arises from the interaction of individual agents (the boids, in this case) adhering to a set of simple rules. The rules applied in the simplest Boids world are as follows:

* separation: steer to avoid crowding local flockmates
* alignment: steer towards the average heading of local flockmates
* cohesion: steer to move towards the average position (center of mass) of local flockmates

More complex rules can be added, such as obstacle avoidance and goal-seeking.

## About this repository

This repository tries to reimplement/optimize [this](https://github.com/roholazandie/boids) brute-force simulation approach which has quadratic time complexity, by using [k-d tree](https://en.wikipedia.org/wiki/K-d_tree) to achieve logarithmic time complexity. Also comparison between two approaches is presented. 

## Installation requirements

 Run:
 ```
 pip install numpy
 ```
  ```
 pip install matplotlib
 ```
  ```
 pip install p5
 ```
 
For installing glfw library, download pre-compiled binaries corresponding to your architecture, from [here](https://www.glfw.org/download.html).
Extract the folder anywhere you like, i would prefer C drive. In extracted folder there is a folder called lib-mingw-w64/w32, add that to the PATH variable. 

If there is a problem with OpenGL try [this](https://gist.github.com/rb-dahlb/26f316c5b6089807a139fc44ee69f0d1) solution.
