import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
def forwardKinematics(phi, theta0, theta1, theta2, l0, l1, l2):
	j1_x = l0*np.cos(phi)*np.cos(theta0)
	j1_y = l0*np.sin(phi)*np.cos(theta0)
	j1_z = l0*np.sin(theta0)
	
	j2_x = j1_x + l1*np.cos(phi)*np.cos(theta0 + theta1)
	j2_y = j1_y + l1*np.sin(phi)*np.cos(theta0 + theta1)
	j2_z = j1_z + l1*np.sin(theta0 + theta1)
	
	e_x = j2_x + l2*np.cos(phi)*np.cos(theta0 + theta1 - theta2)
	e_y = j2_y + l2*np.sin(phi)*np.cos(theta0 + theta1 - theta2)
	e_z = j2_z + l2*np.sin(theta0 + theta1 - theta2)
	return j1_x, j1_y, j1_z, j2_x, j2_y, j2_z, e_x, e_y, e_z

def jacobian(phi,theta0,theta1,theta2,l0,l1,l2):

	J = np.zeros((3,4))
    #partial derivative order: phi, theta0, theta1, theta2
	#x derivatives
	J[0,0] = -l0*np.sin(phi)*np.cos(theta0) - l1*np.sin(phi)*np.cos(theta0 + theta1) - l2*np.sin(phi)*np.cos(theta0 + theta1 - theta2)
	J[0,1] = -l0*np.cos(phi)*np.sin(theta0) - l1*np.cos(phi)*np.sin(theta0 + theta1) - l2*np.cos(phi)*np.sin(theta0 + theta1 - theta2)
	J[0,2] = -l1*np.cos(phi)*np.sin(theta0 + theta1) - l2*np.cos(phi)*np.sin(theta0 + theta1 - theta2)
	J[0,3] = l2*np.cos(phi)*np.sin(theta0 + theta1 - theta2)
	#y derivatives
	J[1,0] = l0*np.cos(phi)*np.cos(theta0) + l1*np.cos(phi)*np.cos(theta0 + theta1) + l2*np.cos(phi)*np.cos(theta0 + theta1 - theta2)
	J[1,1] = -l0*np.sin(phi)*np.sin(theta0) - l1*np.sin(phi)*np.sin(theta0 + theta1) - l2*np.sin(phi)*np.sin(theta0 + theta1 - theta2)
	J[1,2] = -l1*np.sin(phi)*np.sin(theta0 + theta1) - l2*np.sin(phi)*np.sin(theta0 + theta1 - theta2)
	J[1,3] = l2*np.sin(phi)*np.sin(theta0 + theta1 - theta2)
	#z derivatives
	J[2,0] = 0
	J[2,1] = l0*np.cos(theta0) + l1*np.cos(theta0 + theta1) + l2*np.cos(theta0 + theta1 - theta2)
	J[2,2] = l1*np.cos(theta0 + theta1) + l2*np.cos(theta0 + theta1 - theta2)
	J[2,3] = -l2*np.cos(theta0 + theta1 - theta2)
	
	return J

	'''
	This function is supposed to implement inverse kinematics for a robot arm
	with 3 links constrained to move in 2-D. The comments will walk you through
	the algorithm for the Jacobian Method for inverse kinematics.

	INPUTS:
	l0, l1, l2: lengths of the robot links
	x_e_target, y_e_target: Desired final position of the end effector 

	OUTPUTS:
	theta0_target, theta1_target, theta2_target: Joint angles of the robot that
	take the end effector to [x_e_target,y_e_target]
	'''
def inverseKinematics(l0,l1,l2,x_e_target,y_e_target,z_e_target):


    # Initialize the thetas to some value
	phi = 0
	theta0 = 0
	theta1 = 0
	theta2 = 0
    
	# Obtain end effector position x_e, y_e for current thetas: 
	j1_x, j1_y, j1_z, j2_x, j2_y, j2_z, e_x, e_y, e_z = forwardKinematics(phi, theta0, theta1, theta2, l0, l1, l2)
	alpha = 0.01
	dist = 1000
	i=0
	
	#making colormaps for easy visualization
	colors = plt.cm.hsv(np.linspace(0, 1, 200))
	custom_cmap = ListedColormap(colors)

	while dist > 0.1: # Replace the '1' with a condition that checks if your estimated [x_e,y_e] is close to [x_e_target,y_e_target]
    
		#calculate the jacobian
		J = jacobian(phi,theta0,theta1,theta2,l0,l1,l2)
        
        # Calculate the pseudo-inverse of the jacobian (HINT: numpy pinv())     
		J_inv = np.linalg.pinv(J)

        # Update the values of the thetas by a small step
		de_x = x_e_target - e_x
		de_y = y_e_target - e_y
		de_z = z_e_target - e_z
		phi += alpha*(J_inv[0,0]*de_x + J_inv[0,1]*de_y + J_inv[0,2]*de_z)
		theta0 += alpha*(J_inv[1,0]*de_x + J_inv[1,1]*de_y + J_inv[1,2]*de_z)
		theta1 += alpha*(J_inv[2,0]*de_x + J_inv[2,1]*de_y + J_inv[2,2]*de_z)
		theta2 += alpha*(J_inv[3,0]*de_x + J_inv[3,1]*de_y + J_inv[3,2]*de_z)
		

		# Obtain end effector position x_e, y_e for the updated thetas:
		j1_x, j1_y, j1_z, j2_x, j2_y, j2_z, e_x, e_y, e_z = forwardKinematics(phi,theta0, theta1, theta2, l0, l1, l2)
    
		
		dist = np.sqrt((x_e_target - e_x)**2 + (y_e_target - e_y)**2 + (z_e_target - e_z)**2)

        # If you would like to visualize the iterations, draw the robot using drawRobot. 

        
        # Plot the final robot pose
		# Plot the end effector position through the iterations
		drawRobot3D(j1_x, j1_y, j1_z, j2_x, j2_y, j2_z, e_x, e_y, e_z, l0, l1, l2)
		ax.cla()
	drawRobot3D(j1_x, j1_y, j1_z, j2_x, j2_y, j2_z, e_x, e_y, e_z, l0, l1, l2)
	plt.show()
	return phi, theta0, theta1, theta2
    
    

def drawRobot3D(x_1,y_1,z_1,x_2,y_2,z_2,x_e,y_e,z_e, l0, l1, l2):
	x_0, y_0, z_0 = 0, 0, 0 
	ax.set_xlim([-np.sum([l0,l1,l2]), np.sum([l0,l1,l2])])
	ax.set_ylim([-np.sum([l0,l1,l2]), np.sum([l0,l1,l2])])
	ax.set_zlim([-np.sum([l0,l1,l2]), np.sum([l0,l1,l2])])
	ax.plot3D([x_0, x_1, x_2, x_e], [y_0, y_1, y_2, y_e], [z_0, z_1, z_2, z_e], lw=4.5)
	ax.scatter3D([x_0, x_1, x_2, x_e], [y_0, y_1, y_2, y_e], [z_0, z_1, z_2, z_e], color='r')
	plt.pause(0.1)
	
phi = 0
theta0 = 0
theta1 = 0
theta2 = 0
l0 = 1
l1 = 1
l2 = 1
x_e_target = 2
y_e_target = 1
z_e_target = 2

ax = plt.axes(projection='3d')
inverseKinematics(l0,l1,l2,x_e_target,y_e_target,z_e_target)