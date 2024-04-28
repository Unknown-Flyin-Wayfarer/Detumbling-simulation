########## 0.5mT magnetic field PLOTTING ###############


import matplotlib.pyplot as plt
import numpy as np

# Sample data for time and magnetic field (replace with your actual data)
# time = [2100, 1700, 1400, 1100, 800, 600, 400, 300, 250, 200, 170, 150]
# init_angular_velocity = [0.035784, 0.0565784, 0.085784, 0.097784, 0.100084, 0.115784, 0.35784, 0.75784, 1.15784, 1.65784, 2.25784, 2.65784]


time = [2100, 1700, 1400, 1100, 800, 600, 400, 300, 250]
init_angular_velocity = [0.035784, 0.0565784, 0.085784, 0.097784, 0.100084, 0.115784, 0.35784, 0.75784, 1.15784]

# Plotting the data
plt.plot(time, init_angular_velocity, marker='o', linestyle='-')

# Adding labels and title
plt.xlabel('Detumbling Time (in sec)')
plt.ylabel('Initial Angular velocity (in rad)')
plt.title('0.5mT External Magnetic Field. Angular velocity vs Detumbling time')
plt.legend(['0.5mT'])
#plt.yticks(np.arange(0.01, 2.9, 0.005))
#plt.xticks(np.arange(100, 2300, 50))
# Displaying the plot
plt.grid(False)
plt.legend(['0.5mT'])
plt.show()



########## 0.75mT magnetic field PLOTTING ###############


import matplotlib.pyplot as plt
import numpy as np

# Sample data for time and magnetic field (replace with your actual data)
# time = [2100, 1700, 1400, 1100, 800, 600, 400, 300, 250, 200, 170, 150]
# init_angular_velocity = [0.035784, 0.0565784, 0.085784, 0.097784, 0.100084, 0.115784, 0.35784, 0.75784, 1.15784, 1.65784, 2.25784, 2.65784]


time = [1800, 1500, 1200, 900, 650, 400, 250, 200, 175]
init_angular_velocity = [0.035784, 0.0565784, 0.085784, 0.097784, 0.100084, 0.115784, 0.35784, 0.75784, 1.15784]

# Plotting the data
plt.plot(time, init_angular_velocity, marker='o', linestyle='-')

# Adding labels and title
plt.xlabel('Detumbling Time (in sec)')
plt.ylabel('Initial Angular velocity (in rad)')
plt.title('0.75mT External Magnetic Field. Angular velocity vs Detumbling time')
plt.legend(['0.75mT'])
#plt.yticks(np.arange(0.01, 2.9, 0.005))
#plt.xticks(np.arange(100, 2300, 50))
# Displaying the plot
plt.grid(False)
plt.show()


########## 1mT magnetic field PLOTTING ###############


import matplotlib.pyplot as plt
import numpy as np

# Sample data for time and magnetic field (replace with your actual data)
# time = [2100, 1700, 1400, 1100, 800, 600, 400, 300, 250, 200, 170, 150]
# init_angular_velocity = [0.035784, 0.0565784, 0.085784, 0.097784, 0.100084, 0.115784, 0.35784, 0.75784, 1.15784, 1.65784, 2.25784, 2.65784]


time = [1500, 1300, 900, 650, 300, 200, 150, 124, 115]
init_angular_velocity = [0.035784, 0.0565784, 0.085784, 0.097784, 0.100084, 0.115784, 0.35784, 0.75784, 1.15784]

# Plotting the data
plt.plot(time, init_angular_velocity, marker='o', linestyle='-')

# Adding labels and title
plt.xlabel('Detumbling Time (in sec)')
plt.ylabel('Initial Angular velocity (in rad)')
plt.title('1mT External Magnetic Field. Angular velocity vs Detumbling time')
#plt.yticks(np.arange(0.01, 2.9, 0.005))
#plt.xticks(np.arange(100, 2300, 50))
# Displaying the plot
plt.grid(False)
plt.legend(['1mT'])
plt.show()


########## 1.25mT magnetic field PLOTTING ###############


import matplotlib.pyplot as plt
import numpy as np

# Sample data for time and magnetic field (replace with your actual data)
# time = [2100, 1700, 1400, 1100, 800, 600, 400, 300, 250, 200, 170, 150]
# init_angular_velocity = [0.035784, 0.0565784, 0.085784, 0.097784, 0.100084, 0.115784, 0.35784, 0.75784, 1.15784, 1.65784, 2.25784, 2.65784]


time = [1300, 1100, 800, 600, 270, 180, 140, 120, 100]
init_angular_velocity = [0.035784, 0.0565784, 0.085784, 0.097784, 0.100084, 0.115784, 0.35784, 0.75784, 1.15784]

# Plotting the data
plt.plot(time, init_angular_velocity, marker='o', linestyle='-')

# Adding labels and title
plt.xlabel('Detumbling Time (in sec)')
plt.ylabel('Initial Angular velocity (in rad)')
plt.title('1.25mT External Magnetic Field. Angular velocity vs Detumbling time')
#plt.title('Angular velocity vs Detumbling time for different External Magnetic field')
#plt.yticks(np.arange(0.01, 2.9, 0.005))
#plt.xticks(np.arange(100, 2300, 50))
# Displaying the plot
plt.grid(False)
plt.legend(['1.25mT'])
#plt.legend(['0.5mT','0.75mT','1mT','1.25mT'])
plt.show()