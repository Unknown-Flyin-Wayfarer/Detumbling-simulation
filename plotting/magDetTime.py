########## HARDWARE RESULT PLOTTING ###############


import matplotlib.pyplot as plt
import numpy as np

# Sample data for time and magnetic field (replace with your actual data)
time = [5.5, 5.1, 4.7, 4.5, 4.45, 4.4, 4.3, 4.223, 4.198123, 4.123, 3.9, 3.812, 3.564]
magnetic_field = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.2]

# Plotting the data
plt.plot(time, magnetic_field, marker='o', linestyle='-')

# Adding labels and title
plt.xlabel('Detumbling Time (in minutes)')
plt.ylabel('External Magnetic Field (mT)')
plt.title('External Magnetic Field vs Detumbling time')
plt.yticks(np.arange(0.1, 1.4, 0.1))
plt.xticks(np.arange(3.3, 6, 0.2))
# Displaying the plot
plt.grid(False)
plt.show()



########## SIMULATION RESULT PLOTTING ###############

'''import matplotlib.pyplot as plt
import numpy as np

# Sample data for time and magnetic field (replace with your actual data)
time = [5.5, 5.2, 4.7, 3.3, 2.08, 1.69, 1.25, 1.1]
magnetic_field = [0.3, 0.5, 0.65, 0.8, 1.0, 1.15, 1.3, 1.5]

# Plotting the data
plt.plot(time, magnetic_field, marker='o', linestyle='-')

# Adding labels and title
plt.xlabel('Detumbling Time (in minutes)')
plt.ylabel('External Magnetic Field (mT)')
plt.title('External Magnetic Field vs Detumbling time')
plt.yticks(np.arange(0.1, 1.7, 0.1))
plt.xticks(np.arange(1, 5.7, 0.2))
# Displaying the plot
plt.grid(False)
plt.show()'''