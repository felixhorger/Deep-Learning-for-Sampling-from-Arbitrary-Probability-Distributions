
"""--- Inverse transform sampling for two dimensions, example: bimodal gaussian at (1.5, 1.5) and (-1.5, -1.5) ---"""



# Imports
import numpy as np
from scipy.special import erf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# pdfs rho(x,y) alpha(x), beta_x(x,y)

# Aim density rho
def rho(x,y):
	return (np.exp(-((x-1.5)**2 + (y-1.5)**2)/2.0) + np.exp(-((x+1.5)**2 + (y+1.5)**2)/2.0))/(2.0*np.pi*2.0)

# alpha(x) = int rho(x,y) dy over |R  == Probability of drawing x from rho
def alpha(x):
	return (np.exp(-(x-1.5)**2/2.0) + np.exp(-(x+1.5)**2/2.0))/(np.sqrt(2.0*np.pi)*2.0)

# beta_x(y) = rho(x,y) / alpha(x)  == Probability of drawing y from rho, if x was drawn
def beta_x(x,y):
	return rho(x,y)/alpha(x)



# cdf of alpha, beta_x

# integral alpha(x') dx' from -inf to x, currently analytically
def integral_alpha(x):
	return (2.0 + erf((x-1.5)/np.sqrt(2.0)) + erf((x+1.5)/np.sqrt(2.0)))/4.0
	# Numerically:
	# args: (x, x_min=-10.0, dx=1.0/1024.0):
	"""if (x < x_min):
		return 0.0
	x_axis = np.linspace(x_min, x, np.ceil((x-x_min)/dx))
	return np.sum(alpha(x_axis)*dx)"""

# integral beta_x(y') dy' from -inf to y, currently analytically
def integral_beta_x(x, y):
	return (1.0 + erf((y-1.5)/np.sqrt(2.0)))/(2.0*(1.0 + np.exp(-2.0*1.5*x))) + (1.0 + erf((y+1.5)/np.sqrt(2.0)))/(2.0*(1.0 + np.exp(2.0*1.5*x)))
	# Numerically:
	# args: (x, y, y_min=-20.0, dx=1.0/1024.0):
	"""if (y < y_min):
		return 0.0
	y_axis = np.linspace(y_min, y, np.ceil((x-x_min)/dx))
	return np.sum(beta(y_axis)*dx)"""



# Inverse cdfs of alpha, beta_x

def inverse_integral_alpha(u, x_start=0.0, epsilon=1e-08):
	x_min = x_start
	x_max = x_start
	step = 1.0
	# Find lower bound
	while (integral_alpha(x_min) > u):
		x_min -= step	
		if (x_min < -1e10):
			print "Failed to find lower boundary" 
	if (integral_alpha(x_min) == u):
			return x_min
 	# Find upper bound
	while (integral_alpha(x_max) < u):
		x_max += step
		if (x_max > 1e10):
			print "Failed to find upper boundary"
	if (integral_alpha(x_max) == u):
			return x_max
	# Bisection
	x = (x_max + x_min)/2.0
	while (np.abs(integral_alpha(x) - u) > epsilon):
		if (integral_alpha(x) < u):
			x_min = x
		elif (integral_alpha(x) > u):
			x_max = x	
		x = (x_max + x_min)/2.0
	return x


def inverse_integral_beta_x(x, v, y_start=0.0, epsilon=1e-08):
	y_min = y_start
	y_max = y_start
	step = 1.0
	# Find lower bound
	while (integral_beta_x(x, y_min) > v):
		y_min -= step	
		if (y_min < -1e10):
			print "Failed to find lower boundary" 
	if (integral_beta_x(x, y_min) == v):
			return y_min
 	# Find upper bound
	while (integral_beta_x(x, y_max) < v):
		y_max += step 
		if (y_max > 1e10):
			print "Failed to find upper boundary"
	if (integral_beta_x(x, y_max) == v):
			return y_max
	# Bisection
	y = (y_max + y_min)/2.0
	while (np.abs(integral_beta_x(x, y) - v) > epsilon):
		if (integral_beta_x(x, y) < v):
			y_min = y
		elif (integral_beta_x(x, y) > v):
			y_max = y	
		y = (y_max + y_min)/2.0
	return y



# Drawing samples from uniform, transform to samples from rho
N = 10000
x = np.empty(N)
y = np.empty(N)
for i in range(0, N):
	(u, v) = np.random.uniform(0.0, 1.0, 2)
	x[i] = inverse_integral_alpha(u)
	y[i] = inverse_integral_beta_x(x[i], v)



# Calculate histgram
hist = np.histogram2d(x, y, bins=20)
grid = np.meshgrid((hist[1][:hist[1].size-1] + hist[1][1:])/2.0, hist[2][1:])



# Plot
fig = plt.figure(1)
ax = fig.add_subplot(111, projection="3d")
ax.plot_wireframe(grid[0], grid[1], hist[0], color='b')
plt.show()


quit()

"""--- END ---"""

# Generalized inversion function:
"""
def invert(f, y, x_start=0.0, epsilon=1e-08):
	x_min = x_start
	x_max = x_start
	step = 1.0
	while (f(x_min) > y):
		x_min -= step	
		if (x_min < -1e10):
			print "Failed to find lower boundary" 
	if (f(x_min) == y):
			return x_min
 
	while (f(x_max) < y):
		x_max += step 
		if (x_max > 1e10):
			print "Failed to find upper boundary"
	if (f(x_max) == y):
			return x_max

	# Bisection
	x = (x_max + x_min)/2.0
	while (np.abs(f(x) - y) > epsilon):
		if (f(x) < y):
			x_min = x
		elif (f(x) > y):
			x_max = x	
		x = (x_max + x_min)/2.0
		
	return x
"""
