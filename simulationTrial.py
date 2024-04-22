import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyteapot import pybox
#from pyteapot.pyteapot import pybox
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R
import math
import sympy as sp

inertia_matrix = np.array([[0.0015, 0.0, 0.0], 
                           [0.0, 0.0015, 0.0], 
                           [0.0, 0.0, 0.0015]])  # Replace with your inertia matrix
# Replace with the external magnetic field components
ex_b = np.array([0.7E-3, 0.7E-3, 0.7E-3]) #np.array([1E-3, 0, 0])
dt = 0.1  # Time step
K = 100
m = 0.02
prev_mag = [0, 0, 0.0]
omega = np.array([0.45784, 0.21877,0.1548456])  # Initial angular velocity
MaxIter = 3000
progress = tqdm(total=MaxIter)
bderivative = []
torque = []
tx= []
ty= []
tz= []
orientation_array = []
q_now = np.array([1, 0, 0, 0])


visualise = 0

simbox = pybox()

if(visualise):
    #simbox = pybox()
    simbox.update(q_now)


def update_orientation(angular_velocity, dt) -> np.ndarray:
    global q_now
    p, q, r = angular_velocity
    w = np.array([[0, -p, -q, -r],
                  [p, 0, r, -q],
                  [q, -r, 0, p],
                  [r, q, -p, 0]])
    q_now = np.matmul(expm(0.5 * w * dt), q_now.T)
    if(visualise):
        simbox.update(q_now)
    return q_now


def rotate_vector_by_quaternion(B, quaternion):
    # Convert the vector to a pure quaternion
    B_quat = np.concatenate(([0], B))

    rotated_quat = quat_mul(quat_mul(quaternion, B_quat), quaternion.conj())
    # Extract the rotated vector from the result quaternion
    rotated_vector = rotated_quat[1:]
    return np.array(rotated_vector)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])


def reduce_spin_to_zero(inertia_matrix, dt):
    global prev_mag
    global m
    global K
    global omega
    global orientation_array
    global ex_b
    global q_now
    angular_velocities = []
    iteration = 0
    print(str(q_now.T))
    while True: 
        iteration += 1
        # print(iteration)
        rotated_magnetic_field = rotate_vector_by_quaternion(
            ex_b, q_now)

        field = rotated_magnetic_field/np.linalg.norm(rotated_magnetic_field)

        bdot = list((field[i]-prev_mag[i])/dt for i in range(3))

        # Bdot control
        magnetic_moment = K*np.cross(omega, rotated_magnetic_field)

        # bang bang only
        #magnetic_moment = np.array(list((-m if (math.copysign(1,bdot[i]) == 1) else m) for i in range(3)))

        bderivative.append(bdot[0])

        # #Bang Bang + Bdot control
        # if w<0.1:
        #     magnetic_moment = K*np.cross(angular_velocity,bdot)
        # else:
        #     magnetic_moment = list(m if bdot[i]<0 else -m for i in range(3))
        # print(magnetic_moment)
        p, q, r = omega.flatten()

        # print(str([L,M,N]))
        
        LMN = np.cross(magnetic_moment, rotated_magnetic_field)

        #tx, ty, tz = LMN.flatten()

        torque.append(LMN)
        #print(str(LMN))
        tx.append(LMN[0])
        ty.append(LMN[0])
        tz.append(LMN[0])

        Spqr = np.array([[0, -r, q],
                        [r, 0, -p],
                        [-q, p, 0]])  # Skew matrix of pqr

        a = np.matmul(inertia_matrix, omega)
        b = np.matmul(Spqr, a)
        pqrdot = np.matmul(np.linalg.inv(inertia_matrix), (LMN-b))

        p += pqrdot[0] * dt
        q += pqrdot[1] * dt
        r += pqrdot[2] * dt
        omega = np.array([p, q, r])

        # w = np.linalg.norm([p,q,r])
        # print(w)
    
        orientation_array.append(update_orientation(omega, dt).flatten().copy())
        angular_velocities.append(omega.flatten().copy())
        prev_mag = field
        # if w <= 0.01:
        #     break
        if iteration > MaxIter:
            break
        progress.update(1)

    print("Number of iterations:", iteration)
    progress.close()
    return np.array(angular_velocities)


angular_velocities = reduce_spin_to_zero(
    inertia_matrix, dt)

if(visualise):
    simbox.end()

# Plot angular velocity of each axis
time = np.arange(0, len(angular_velocities) * dt, dt)
wx, wy, wz = angular_velocities.T
q0, q1, q2,q3 = np.array(orientation_array).T

fig1 = plt.figure()
plt.plot(time, wx-0.2, label='wx')
plt.plot(time, wy-0.2, label='wy')
plt.plot(time, wz-0.2, label='wz')
plt.xlabel('Time')
plt.ylabel('Angular Velocity')
plt.yticks(np.arange(-0.25, 0.25, 0.05))
plt.legend()

fig2 = plt.figure()
plt.plot(time, bderivative)
plt.xlabel('Time')
plt.ylabel('Bderivative X')

fig3 = plt.figure()
plt.plot(time, q0, label='q0')
plt.plot(time, q1, label='q1')
plt.plot(time, q2, label='q2')
plt.plot(time, q3, label='q3')
plt.xlabel('Time')
plt.ylabel('quaternions')
plt.legend()

'''fig4 = plt.figure()
plt.plot(time, tx, label='tx')
plt.plot(time, ty, label='ty')
plt.plot(time, tz, label='tz')
plt.xlabel('Time')
plt.ylabel('Torque')
plt.legend()'''

fig5 = plt.figure()
plt.plot(time, torque)
plt.xlabel('Time')
plt.ylabel('Torque')
plt.legend()



plt.show()