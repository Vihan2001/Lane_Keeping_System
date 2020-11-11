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
audi = Car(length =2.3, y=1, theta= np.deg2rad(0)) #example of car and call its length
p_controller =PIDController(kp= 0.001, ki=0, kd= 0, ts= dt)

#print(audi.y) #before applying steering angle
#audi.move(steering_angle= 0.1, dt=0.01) #to move audi, state steering angle for a time
#print(audi.y) #y coordinate after applying steering angle
#print(audi.x)#x coordinate after applying steering angle


# Simulation

# kp = 0.1, Kd = 0.001
audi = Car(length=2.3, y=0.3)
p_controller1 = PIDController(kp=0.1, ki=0, kd=0.001, ts=dt)
y_cache = [audi.y]
t_cache = [0]
u_dist = np.deg2rad(1)
for idx_t in range(2000):
    u = p_controller.control(y=audi.y)
    audi.move(u + u_dist, dt)
    y_cache += [audi.y]
    t_cache += [(idx_t+1)*dt]

# kp = 0.1, Kd = 0.01
audi = Car(length=2.3, y=0.3)
p_controller2 = PIDController(kp=0.1, ki=0, kd=0.01, ts=dt)
y_cache2 = [audi.y]
for idx_t in range(2000):
    u = p_controller2.control(y=audi.y)
    audi.move(u + u_dist, dt)
    y_cache2 += [audi.y]

# kp = 0.1, Kd = 0.05
audi = Car(length=2.3, y=0.3)
p_controller3 = PIDController(kp=0.1, ki=0, kd=0.05, ts=dt)
y_cache3 = [audi.y]
for idx_t in range(2000):
    u = p_controller3.control(y=audi.y)
    audi.move(u + u_dist, dt)
    y_cache3 += [audi.y]

# kp = 0.1, Kd = 0.5
audi = Car(length=2.3, y=0.3)
p_controller4 = PIDController(kp=0.1, ki=0, kd=0.5, ts=dt)
y_cache4 = [audi.y]
for idx_t in range(2000):
    u = p_controller3.control(y=audi.y)
    audi.move(u + u_dist, dt)
    y_cache4 += [audi.y]


#Plotting
plt.plot(t_cache,y_cache,'r',label='kd = 0.001')
plt.plot(t_cache,y_cache2,'g',label='kd = 0.01')
plt.plot(t_cache,y_cache3,'k',label='kd = 0.05')
plt.plot(t_cache,y_cache4,'y',label='kd = 0.5')
plt.xlabel('Time (s)')
plt.ylabel('y-plane (m)')
plt.grid()
plt.title('PD controller of varying Kd values, but fixed Kp = 0.1')
plt.legend(loc='upper right')
plt.show()