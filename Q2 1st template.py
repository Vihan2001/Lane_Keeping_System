import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class Car: #Class of a car
    #aspects of a car
    def __init__(self, length =2.3, velocity= 5, x=0, y=0, theta= np.deg2rad(0)): #default values

        self.length = length #length of car provided by user
        self.velocity = velocity #velocity of car provided by user
        self.x = x #x coordinate of car  provided by user
        self.y = y #y coordinate of car  provided by user
        self.theta = theta #Theta angle provided by user but steering angle isnt a proper property of car

    def move(self, steering_angle, dt): #new function called move, add steering angle for a time dt
        z_initial = [self.x, self.y, self.theta ]

        def system_dynamics(t, z): #Function used to simulate where the car will go with constant steering action for time dt
            theta = z[2]
            return [self.velocity * np.cos(theta),
                    self.velocity * np.sin(theta),
                    self.velocity * np.tan(steering_angle) / self.length] #pass the 3 arguments for simulation

        solution = solve_ivp(system_dynamics,
                             [0, dt],
                             z_initial) #pass arugments for solution
        self.x = solution.y[0][-1] #new x coordinate of car, find from a matrix, first column is inital state and last column is the final state
        self.y = solution.y[1][-1]
        self.theta = solution.y[2][-1]

class PIDController: #Adding a controller element
    def __init__(self, kp, ki, kd, ts): # user has to tell kp, ki, kd
        self.kp = kp
        self.ki = ki * ts
        self.kd = kd / ts
        self.ts = ts
        self.previous_error = None #doesnt exist
        self.sum_errors = 0 #integral from 0 till time=0 of error

    def control(self, y, set_point=0): #controller controls and tells how far from error, compute's u
        error = set_point -y
        kp = self.kp #P controller
        u = kp * error

        # Add D component(mode) to converge
        if self.previous_error is not None:
            kd = self.kd
            error_difference = error - self.previous_error #we dont have a previous error
            u += kd * error_difference
        self.previous_error = error #update previous with error

        #Add I component(mode) to eliminate off-set
        u += self.ki * self.sum_errors
        self.previous_error = error
        self.sum_errors += error

        return u

dt= 0.025 #used for sampling rate (s)
audi = Car(length =2.3, y=0.3, theta= np.deg2rad(0)) #example of car and call its length
p_controller =PIDController(kp= 0.3, ki=0.01, kd= 0.4, ts= dt)


#print(audi.y) #before applying steering angle
#audi.move(steering_angle= 0.1, dt=0.01) #to move audi, state steering angle for a time
#print(audi.y) #y coordinate after applying steering angle
#print(audi.x)#x coordinate after applying steering angle

#Simulation

y_cache = [audi.y]
t_cache = [0] #intial time is 0s
u_distb = np.deg2rad(1) #constant distrubance applied to car
for idx_t in range(2000): #looping the car moving 1000 times
    u = p_controller.control(y=audi.y) #call the PID argument
    audi.move(u + u_distb, dt) #car goes straight + disturbance
    y_cache += [audi.y] #new value added to cache/empty list
    t_cache += [(idx_t+1)*dt]
    #make range to 1000 and multiply by dt
    #print("X coordinate is", audi.x) #printing audi at every point at the x-axis
    #print("Y coordinate is", audi.y) #Audi at y-axis


u_cache = [p_controller.control(y=audi.y)]
t_cache = [0]
u_distb = np.deg2rad(1)
for idx_t in range(2000):
    u = p_controller.control(y=audi.y)
    audi.move(u + u_distb, dt)
    u_cache = np.append(u_cache, audi.y)
    t_cache += [(idx_t + 1) * dt]



plt.plot(t_cache, u_cache)
plt.xlabel("Time (s)")
plt.ylabel("Steering Angle (radians)")
plt.grid()
plt.show()
