"""

Sample Mapper

A Multilayer-Perceptron is trained to map random samples of the 2D uniform PDF over
[-1, 1] x [-1, 1] to the same amount of random samples of an arbitrary 2D PDF.

"""





""" / / / / / IMPORTS / / / / / """


# Import os
import os


# Import numpy
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Import keras
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Reshape
import keras.backend as K
from keras.callbacks import TerminateOnNaN as K_TerminateOnNaN
from tensorflow import reset_default_graph as tf_reset_default_graph





""" / / / / / TRAINING AND TESTING DATA / / / / / """


"""

Name definition:

					 ___________
					|			|
	input vector ->	|	Model	| -> output vector
					|___________|


	One in-/output vector consists of elements_per_vector elements.
	In a batch are vectors_per_batch vectors.

"""


# Dimensions
elements_per_vector = 500
vectors_per_batch = 500


# Generator for training data
"""
	Generates a batch of input vectors and the corresponding desired output of
	the model. For this model, the desired output can not be defined in terms
	of a fixed output vector for each input vector. Therefore an empty array is
	used as a placeholder. The input vectors are produced by drawing samples from
	the uniform 2D PDF over [-1, 1] x [-1, 1].
"""
def uniform_gen():
	while True:
		yield (np.random.uniform(low=-1.0, high=1.0, size=(vectors_per_batch, elements_per_vector, 2)), np.empty((vectors_per_batch, elements_per_vector, 2)))


# Generator for testing data
"""
	Does the same as the generator above, but it will be used to evaluate the model
	after training. Also, N_testvectors can be chosen higher than vectors_per_batch.
"""
N_testvectors = vectors_per_batch
def test_gen():
	while True:
		yield np.random.uniform(low=-1.0, high=1.0, size=(N_testvectors, elements_per_vector, 2))





""" / / / / / PDF APPROXIMATION / / / / / """


# 2D Kernel density estimation (KDE) using a 2D gaussian kernel function.
"""
	This produces a single value of the KDE of a vector y = [[y_11, y_12], [y_21, y_22], ...] at position y' = [y_1, y_2], with delta_y[i] = y[i] - y'.
	The width of the kernel function can be defined by h.
"""
def kde_gauss2D(delta_y, h):
	return np.sum( np.exp( -0.5*np.sum(delta_y**2, axis=1)/(h**2) ), 0)/ (2.0*np.pi*(h**2)*float(delta_y.shape[0]))





""" / / / / / DENSITY / / / / / """


# 2D Gaussian PDF with zero mean and sigma = 1, last input dim has to be 2
"""
Gnuplot formula:
rho(x,y) = exp(-0.5*(x**2+y**2))/(2.0*pi)
"""
def gaussian2D(y):
	if (type(y) == np.ndarray):
		return np.exp(-(np.sum(y**2, axis=-1))/2.0)/(2.0*np.pi)
	else:
		return K.exp(-K.sum(y**2, axis=1)/2.0) / (2.0*np.pi)


# 2D bimodal gaussian PDF with peaks at (1.5, 1.5) and (-1.5, -1.5) with sigma = 1
"""
Gnuplot formula:
rho(x,y) = (exp(-0.5*((x-1.5)**2+(y-1.5)**2)) + exp(-0.5*((x+1.5)**2+(y+1.5)**2)))/(2.0*2.0*pi)
"""
def bimodal2D(y):
	if (type(y) == np.ndarray):
		return (np.exp(-np.sum((y - np.array([1.5, 1.5]))**2, axis=-1)/2.0) + np.exp(-np.sum((y + np.array([1.5, 1.5]))**2, axis=-1)/2.0)) / (2.0*np.pi*2.0)
	else:
		return (K.exp(-K.sum((y - K.variable([1.5, 1.5]))**2, axis=1)/2.0) + K.exp(-K.sum((y + K.variable([1.5, 1.5]))**2, axis=1)/2.0)) / (2.0*np.pi*2.0)

# Probability density function to be used in the loss function, set here for simplicity
density = bimodal2D





""" / / / / / LOSS FUNCTION AND METRICS / / / / / """


"""

	Explanation of the loss fct:



	y_pred =	| y_pred(1,1)					| y_pred(1,2)					|					| y_pred(1,elements_per_vector)					|   first output vector
			 	| y_pred(2,1) 					| y_pred(2,2)					|					|												|   second output vector
				|								|								|					|												|
				|  ...							|  ...							|		...			|	...											|    ...
				|								|								|					|												|
				| y_pred(vectors_per_batch,1)	|  								|					| y_pred(vectors_per_batch,elements_per_vector)	|   last output vector


				first element of output vectors  second element of output vectors		...			  last element of output vectors

	where y_pred(i,k) = [y_pred(i,k)_1, y_pred(i,k)_2]

	Penalties:
	"vectors":	KDE of each output vector is compared to density.
	"elements":	KDE of each element of the output vectors is compared to density.
	"pot":		Since the KDE and density are only compared in [y_min, y_max] x [y_min, y_max], there has to be a penalty for points outside of this interval, a linear potential well.



	The KDE and density will be compared using the Mean Squared Error (MSE) or the Jensen-Shannon-Divergence (JSD). These include integrals, which have to approximated
	numerically by transforming them into a sum over the cartesian product of y1_axis and y2_axis (grid). This grid is the same for both the output vectors and the
	vector elements, for simplicity. The axis are linspaces from y_min to y_max with elements separated by step.

"""


# Interval, where the pdfs are compared
y_min = -5.0
y_max = 5.0


# y1 and y2 axis step sizes
step = 0.5
# Width parameter h of KDE is chosen to be step, compromise between training time and resolution

# y1 and y2 axis for the comparison of the KDE to rho
y1_axis = np.linspace(y_min, y_max, int((y_max-y_min)/step)+1)
y2_axis = np.linspace(y_min, y_max, int((y_max-y_min)/step)+1)
y1_axis_size = y1_axis.size
y2_axis_size = y2_axis.size


# Weighting factors omitted for the 2D case, for simplicity


""" Penalties """

"""

	Explanation of ``array calculus'':

	Since in tensorflow, it is not possible to directly access array elements by indexing,
	all calculations have to be done with whole arrays using broadcasting (see e.g. numpy reference).
	The proceed will be explained for the calculation of the KDE of the output vectors:
	In order to calculate the KDE of every output vector, the difference between every output values
	(two dimensional!) and every pair of elements in y1_axis and y2_axis has to be calculated.

	Denoting the elements in y1_axis as y1_j with j in {1, 2, ..., y1_axis_size} and the elements of
	y2_axis accordingly. Then the y_grid is given by (cartesian product of y1_axis and y2_axis)

	y_grid =

	= 	| y1_1, y2_1 |   	|
		| y1_2, y2_1 |		V	size in this direction is y1_axis_size*y2_axis_size
		| y1_3, y2_1 |
			...
		| y1_1, y2_2 |
			...

	The repeated y grid is calculated to:

		y_grid_repeated_vectors =

		=	| y1_1, y2_1 | y1_1, y2_1 |	...
			| y1_2, y2_1 | y1_2, y2_1 | ...			|
			| y1_3, y2_1 | ...						V	size in this direction is y_grid_size
			  	...

  			->
  			size in this direction is elements_per_vector

  	Denoting the elements of the output matrix y_pred as y_ik_m with m in {1,2}

  	K.tile(y_pred, [1, y_grid_size, 1]) =

  	=	| y_11_1, y_11_2 | y_12_1, y_12_2 | ... | y_11_1, y_11_2 | ...		|
  		| y_21_1, y_21_2 | y_22_1, y_22_2 | ... | y_21_1, y_21_2 | ...		V	size in this direction is vectors_per_batch


  		->
  		size in this direction is elements_per_vector*y_grid_size


	Then,
	K.reshape( K.tile(y_pred, [1, y_grid_size, 1]), [vectors_per_batch, y_grid_size, elements_per_vector, 2] )
	simply splits the second axis of the array K.tile(y_pred, [1, y_grid_size, 1]) with size elements_per_vector*y_grid_size
	into two axis with sizes y_grid_size and elements_per_vector. Unfortunately this array is four dimensional and it can be
	displayed without confusion on this 2D plane...
	Now, y_grid_repeated_vectors can directly be subtracted.

	The KDE is calculated by squaring delta_y and summing of the very last dimension (this correspond to index m).
	Then the exponential can be applied (and other factors), summing again over the last axis gives the KDE at the
	any position listed in y_grid.

"""


# Calculate the size of the carteasian product of y1_axis and y2_axis, helper variable
y_grid_size = y1_axis_size*y2_axis_size


def penalty_vectors(y_true, y_pred):

	error = K.variable(0.0)

	# Calculate the y grid
	y_grid = K.transpose(K.stack([ K.tile(K.constant(y1_axis), y2_axis_size), K.flatten( K.tile( K.reshape(K.constant(y2_axis), [y2_axis_size, 1]), [1, y1_axis_size]))]))
	y_grid_repeated_vectors = K.reshape(K.repeat_elements(y_grid, elements_per_vector, axis=0) , [y_grid_size, elements_per_vector, 2])

	# Calculate the KDE of output vectors
	delta_y = K.reshape( K.tile(y_pred, [1, y_grid_size, 1]), [vectors_per_batch, y_grid_size, elements_per_vector, 2] ) - y_grid_repeated_vectors
	tmp = K.exp(-0.5*K.sum(delta_y**2, axis=3)/((step/2.0)**2))
	kde_vectors = K.sum(tmp, axis=2) / (2.0*np.pi*(step/2.0)**2*float(elements_per_vector))

	# MSE
	#error = K.sum(K.sum((kde_vectors - density(y_grid))**2, axis=1), axis=0)/(float(vectors_per_batch))

	# JSD
	error = K.sum(K.sum(kde_vectors*K.log(K.maximum(kde_vectors, 1e-08)/K.maximum(density(y_grid), 1e-08)) + density(y_grid)*K.log(K.maximum(density(y_grid), 1e-08)/K.maximum(kde_vectors, 1e-08)), axis=1), axis=0)/float(vectors_per_batch)
	return error
	# y_true unused


def penalty_elements(y_true, y_pred):

	error = K.variable(0.0)

	# Calculate the y grid
	y_grid = K.transpose(K.stack([ K.tile(K.constant(y1_axis), y2_axis_size), K.flatten( K.tile( K.reshape(K.constant(y2_axis), [y2_axis_size, 1]), [1, y1_axis_size]))]))
	y_grid_repeated_elements = K.reshape(K.repeat_elements(y_grid, vectors_per_batch, axis=0) , [y_grid_size, vectors_per_batch, 2])


	# Calculation of the KDE of output vector elements
	delta_y = K.reshape( K.tile( K.permute_dimensions(y_pred, [1,0,2]), [1, y_grid_size, 1]), [elements_per_vector, y_grid_size, vectors_per_batch, 2] ) - y_grid_repeated_elements
	tmp = K.exp(-0.5*K.sum(delta_y**2, axis=3)/((step/2.0)**2))
	kde_elements = K.sum(tmp, axis=2) / (2.0*np.pi*(step/2.0)**2*float(vectors_per_batch))

	# MSE
	#error = K.sum(K.sum((kde_elements - density(y_grid))**2, axis=1), axis=0)/(float(elements_per_vector))

	# JSD
	error = K.sum(K.sum(kde_elements*K.log(K.maximum(kde_elements, 1e-08)/K.maximum(density(y_grid), 1e-08)) + density(y_grid)*K.log(K.maximum(density(y_grid), 1e-08)/K.maximum(kde_elements, 1e-08)), axis=1), axis=0)/float(elements_per_vector)
	return error
	# y_true unused


def penalty_pot(y_true, y_pred):
	return 1.0*K.sum(K.sum(K.sum(K.maximum(K.abs(y_pred - (y_max + y_min)/2.0) - (y_max - y_min)/2.0, 0.0), axis=2), axis=1), axis=0)
	# y_true unused


def maximum(y_true, y_pred):
	# Maximum absolute output value, problem: is averaged over batch before displayed during training
	return K.max(K.abs(y_pred))


# Loss function
"""
	This simply sums the penalties listed above.
"""
def loss(y_true, y_pred):
	# y_true unused
	return penalty_vectors(y_true, y_pred) + penalty_elements(y_true, y_pred) + penalty_pot(y_true, y_pred)





""" / / / / / RUN_MODEL / / / / / """


def run_model(load, N_layers, N_units, opt, epochs, N_testvectors, x_train, x_test, save_results, path, plot):

	"""
		This method can be used to perform one training on the model given all necessary parameters, save the results,
		or load previous ones. Also, plots of the results can be produced.

		args:

		* load = (True, False), whether to load weights and model from argument path
		* N_layers = unsigned int > 0
		* N_units = unsigned int > 0, amount units in a hidden layer, in- and output layer have elements_per_vector units
		* opt = keras optimizer object, attention: create a new optimizer object for every run, otherwise it will not work!
		* epochs = unsigned int > 0
		* N_testvectors = unsigned int > 0, how many vectors are to be expected in x_test
		* x_train = generator yielding array[vectors_per_batch][elements_per_vector][2], training vectors
		* x_test = generator yielding array[N_testvectors][elements_per_vector][2], testing vectors
		* save_results = (True, False), whether to save the results or not
		* path = string to where the results should be saved to or where the weights and model should be loaded from
		* plot = (True, False), whether to plot or not

	"""


	# Build model
	if (load == True):
		with open(path+"/model.json", 'r') as json_file:
			model_json = json_file.read()
			json_file.close()
		model = model_from_json(model_json)
		model.load_weights(path+"/weights")
		# Needed if training should go on after loading, not implemented yet:
		#if (epochs > 0):
		#	model.compile(loss=loss, optimizer=opt, metrics=[penalty_vectors, penalty_elements, penalty_pot, maximum])
		# model fit generator...
	else:
		# Define Model layers, units and activation
		model = Sequential()
		layers = [Reshape((2*elements_per_vector,),input_shape=(elements_per_vector, 2))]
		N = 1
		for i in range(1, N_layers):
			layers.append(Dense(2*elements_per_vector, activation="elu"))
		layers.append(Dense(2*elements_per_vector, activation=None))
		layers.append(Reshape((elements_per_vector, 2)))
		for layer in layers:
			model.add(layer)
		# Train the model
		model.compile(loss=loss, optimizer=opt, metrics=[penalty_vectors, penalty_elements, penalty_pot, maximum])
		terminator = K_TerminateOnNaN()
		history = model.fit_generator(generator=x_train(), steps_per_epoch=1, epochs=epochs, workers=1, use_multiprocessing=False, callbacks=[terminator], verbose=1)


	# Let the model map the test-input vectors from x_test to output vectors into y_pred
	y_pred = model.predict_generator(generator=x_test(), steps=1, max_queue_size=1, workers=1, use_multiprocessing=False, verbose=0)
	y_pred_t = np.transpose(y_pred, axes=(1,0,2))


	# Calculate the KDE of output vectors
	kde_vectors = np.empty((N_testvectors, y1_axis_size, y2_axis_size))
	for s in range(0, N_testvectors): # Over all vectors
		for i in range(0, y1_axis_size): # Over all intervals of length step in [-y_min,y_max] along the y1 axis
			for j in range(0, y2_axis_size): # Over all intervals of length step in [-y_min,y_max] along the y2 axis
				kde_vectors[s][i][j] = kde_gauss2D(y_pred[s] - [y1_axis[i], y2_axis[j]], step/2.0)
	kde_vectors_mean = np.mean(kde_vectors, axis=0)
	kde_vectors_std = np.std(kde_vectors, axis=0)


	# Calculate the KDE of each output vector element across all output vectors
	kde_elements = np.empty((elements_per_vector, y1_axis_size, y2_axis_size))
	for l in range(0, elements_per_vector): # Over all elements in a vector
		for i in range(0, y1_axis_size): # Over all intervals of length step in [-y_min,y_max] along the y1 axis
			for j in range(0, y2_axis_size): # Over all intervals of length step in [-y_min,y_max] along the y2 axis
				kde_elements[l][i][j] = kde_gauss2D(y_pred_t[l] - [y1_axis[i], y2_axis[j]], step/2.0)
	kde_elements_mean = np.mean(kde_elements, axis=0) # is equal to kde_vectors_mean! fuer egal welches parzen window!
	kde_elements_std = np.std(kde_elements, axis=0) # is not equal to kde_vectors_std!


	# Expected probability density rho
	rho = np.empty((y1_axis_size, y2_axis_size))
	for i in range(0, y1_axis_size): # Over all intervals of length step in [-y_min,y_max] along the y1 axis
		for j in range(0, y2_axis_size): # Over all intervals of length step in [-y_min,y_max] along the y2 axis
			rho[i][j] = density(np.array([y1_axis[i], y2_axis[j]]))


	# Save results
	if (save_results == True):
		# Weights were loaded, where to save results?
		if (load == True):
			raw_in = raw_input("Insert save path: ")
			if (raw_in != ""):
				path = raw_in

		# Make last directory if it does not exist
		if (not os.path.isdir(path)):
			os.mkdir(path)

		# Save the weights again to path
		model_json = model.to_json()
		with open(path+"/model.json", "w") as json_file:
			json_file.write(model_json)
			json_file.close()
		model.save_weights(path+"/weights", overwrite=True)

		# Losses
		if ((load == False) and (epochs > 0)):
			with open(path+"/penalty_epoch", "w") as f:
				f.write("#penalty_vectors\tpenalty_elements\tpenalty_pot\tmaximum\n")
				for i in range(0, len(history.history["penalty_vectors"])):
					f.write(str(history.history["penalty_vectors"][i]) + "\t" +
							str(history.history["penalty_elements"][i]) + "\t" +
							str(history.history["penalty_pot"][i]) + "\t" +
							str(history.history["maximum"][i]) + "\n")
				f.close()
		else:
			print "History not saved, only possible if load == False and epochs > 0"

		# KDE of output vectors
		with open(path+"/density_vectors", "w") as f:
			f.write("#y1\ty2\tmean\tstd\tdensity of vector i ...\n")
			for i in range(0, y1_axis_size): # Over all elements of the y1 axis
				for j in range(0, y2_axis_size): # Over all elements of the y2 axis
					# Mean and std. deviation
					f.write(str(y1_axis[i]) + "\t" + str(y2_axis[j]) + "\t" +
							str(kde_vectors_mean[i][j]) + "\t" +
							str(kde_vectors_std[i][j]))
					# Individual vectors
					for s in range(0, N_testvectors): # Over all output vectors
						f.write("\t" + str(kde_vectors[s][i][j]))
					f.write("\n")
				f.write("\n") # Newline for gnuplot's pm3d
			f.close()

		# KDE of output vectors elements across all output vectors
		with open(path+"/density_elements", "w") as f:
			f.write("#y1\ty2\tmean\tstd\tdensity of element i ...\n")
			for i in range(0, y1_axis_size): # Over all elements of the y1 axis
				for j in range(0, y2_axis_size): # Over all elements of the y2 axis
					# Mean and std. dev.
					f.write(str(y1_axis[i]) + "\t" + str(y2_axis[j]) + "\t" +
							str(kde_elements_mean[i][j]) + "\t" +
							str(kde_elements_std[i][j]))
					# Individual vectors
					for l in range(0, elements_per_vector): # Over all elements in a output vector
						f.write("\t" + str(kde_elements[l][i][j]))
					f.write("\n")
				f.write("\n") # Newline for gnuplot's pm3d
			f.close()


	# Weight histogram saving ommited for clarity


	# Plot
	if (plot == True):
		# History of losses
		if ((load == False) and (epochs > 0)):
			plt.figure(0)
			plt.title("Losses")
			plt.xlabel("training step")
			plt.ylabel("loss")
			plt.plot(history.history["penalty_vectors"], "b", label="penalty_vectors")
			plt.plot(history.history["penalty_elements"], "g", label="penalty_elements")
			plt.plot(history.history["penalty_pot"], "r", label="penalty_pot")
			plt.legend(loc='best', ncol=1)

		# y grid
		y1_grid, y2_grid = np.meshgrid(y1_axis, y2_axis)

		# Mean of KDEs of output vectors
		plt.figure(1)
		plt.title("Mean KDE of output vectors")
		plt.xlabel("y_1")
		plt.ylabel("y_2")
		plt.imshow(kde_vectors_mean, interpolation="none")
		plt.colorbar()

		# Std. dev. of KDE of output vectors
		plt.figure(2)
		plt.title("Std. dev. of output vectors")
		plt.xlabel("y_1")
		plt.ylabel("y_2")
		plt.imshow(kde_vectors_std, interpolation="none")
		plt.colorbar()

		# Mean of KDEs of output vector elements
		plt.figure(3)
		plt.title("Mean KDE of output vector elements")
		plt.xlabel("y_1")
		plt.ylabel("y_2")
		plt.imshow(kde_elements_mean, interpolation="none")
		plt.colorbar()

		# Std. dev. of KDE of output vector elements
		plt.figure(4)
		plt.title("Std. dev. of output vector elements")
		plt.xlabel("y_1")
		plt.ylabel("y_2")
		plt.imshow(kde_elements_std, interpolation="none")
		plt.colorbar()

		# Example KDE of output vectors
		plt.figure(5)
		plt.title("Example KDE of output vectors")
		plt.xlabel("y_1")
		plt.ylabel("y_2")
		plt.imshow(kde_vectors[0], interpolation="none")
		plt.colorbar()

		# Example KDE of output vector elements
		plt.figure(6)
		plt.title("Example KDE of output vector elements")
		plt.xlabel("y_1")
		plt.ylabel("y_2")
		plt.imshow(kde_elements[0], interpolation="none")
		plt.colorbar()

		plt.show()


	# Weight histogram plots ommited for clarity


	# Clean up memory
	K.clear_session()
	tf_reset_default_graph()

	# Return prediction == output vectors for x_test
	return y_pred
#





""" / / / / / TESTS / / / / / """
"""
	The model for sampling from a 2D PDF with dependent variables is trained, using a
	symmetric bimodal Gaussian PDF.

"""


path = "Tests"
y_pred_2d = run_model(	load			= False,
						N_layers		= 20,
						N_units			= 2*elements_per_vector,
						opt				= Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001),
						epochs			= 10000,
						N_testvectors	= N_testvectors,
						x_train			= uniform_gen,
						x_test			= test_gen,
						save_results	= True,
						path			= path,
						plot			= False
					)

# Save prediction
with open(path+"/prediction", "w") as f:
	f.write("#vector_ik_1\t vector_ik_2 ... \n")
	for k in range(0, elements_per_vector):
		for i in range(0, vectors_per_batch):
			f.write(str(y_pred_2d[i][k][0]) + "\t" + str(y_pred_2d[i][k][1]) + "\t")
		f.write("\n")
	f.close()


""" / / / / / END OF FILE / / / / / """