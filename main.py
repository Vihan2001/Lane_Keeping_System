import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#Equations of Motion
# x' = v * cos(theta)
# y' = v * sin(theta)
#theta' = v * tan(u) / L

# (x, y) describe the position of vehicle on the plane
# theta is the orientation of the car e.g N or E
# v is the velocity of car
# L is the length from axle to axle
# U is the steering angle

# State Space Representation
#      [x]
#  z = [y]
#      [theta]
#z' = f(t, z) #system only solves two arguments time and z

v = 5 #5m\s
L = 2.3 #2.3m
u = np.deg2rad(-2) #radians and counterclockwise with positive co-efficient due to frame of reference/clockwise would be -

def system_dynamics(t, z): #putting eq of motion into a function of the state z

    theta = z[2] #2th element of z array
    return [v * np.cos(theta),  #put the eq of motion
            v* np.sin(theta),
            v * np.tan(u)/L]

#Simulation
#IVP consist's of differential eq z' = f(z) and given initial condition z(0)

t_final = 3 #to what time(s) will the system be simulated
z_intial = [0, 0.3, np.deg2rad(5)] #intial state z(0), what is x,y & theta at time=0

solution = solve_ivp(system_dynamics,
                     [0, t_final],
                     z_intial,
                     t_eval = np.linspace(0, t_final, 250)) #Pass Arguments and use t_eval to increase resolution by including more points
#print("time =",solution.t) # t sol is array of time intervals from to t_final so it shows list of t values
#print("x = ", solution.y[0]) # y is z, the way python calls the solution, y[0] is x
#print("y = ", solution.y[1]) #y[1] is y
#print("theta = ", solution.y[2]) #y[2] is pose or theta angle

t = solution.t
x = solution.y[0] #x value from 1st element of array
plt.plot(t, x)
plt.xlabel('time (s)') #labels for axis
plt.ylabel('x coordinate (m)') #labels for axis
plt.grid()
plt.show()

t = solution.t
y = solution.y[1] #x value from 2st element of array
plt.plot(t, y)
plt.xlabel('time (s)') #labels for axis
plt.ylabel('y coordinate (m)') #labels for axis
plt.grid()
plt.show()

t = solution.t
theta = solution.y[2] #theta value from 3nd element of array
plt.plot(t, theta)
plt.xlabel('time (s')
plt.ylabel('theta (radians)')
plt.grid()
plt.show()

#<--- OPEN LOOP SYSTEM with no controller