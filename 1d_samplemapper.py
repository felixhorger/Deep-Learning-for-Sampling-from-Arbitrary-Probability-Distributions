"""

1D Sample Mapper

A multilayer perceptron is trained to map random samples of the uniform PDF over [-1, 1]
to the same amount of random samples of an arbitrary PDF.

"""





""" / / / / / IMPORTS / / / / / """


# Import os
import os


# Import numpy and matplotlib
import numpy as np
from matplotlib import pyplot as plt


# Import keras
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam, SGD
from keras.layers import Dense
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
	the uniform PDF over [-1, 1].
"""
def uniform_gen():
	while True:
		yield (np.random.uniform(low=-1.0, high=1.0, size=(vectors_per_batch, elements_per_vector)), np.empty((vectors_per_batch, elements_per_vector)))

# Generator for testing data
"""
	Does the same as the generator above, but it will be used to evaluate the model
	after training. Also, N_testvectors can be chosen higher than vectors_per_batch.
"""
N_testvectors = vectors_per_batch
def uniform_test_gen():
	while True:
		yield np.random.uniform(low=-1.0, high=1.0, size=(N_testvectors, elements_per_vector))





""" / / / / / PDF APPROXIMATION / / / / / """


# Kernel density estimation (KDE) using a Gaussian kernel function.
"""
	This produces a single value of the KDE of a vector y at position y', with delta_y[i] = y[i] - y'.
	The width of the kernel function can be defined by h.
"""
def kde_gauss(delta_y, h):
	return np.sum( np.exp(-0.5*((delta_y) / (h) )**2), 0)/ (np.sqrt(2.0*np.pi)*(h)*float(delta_y.size))





""" / / / / / TARGETS / / / / / """


# Gaussian PDF with zero mean and sigma = 1
"""
	Gnuplot formula:
	rho(y) = exp(-(y**2)/2.0) / (sqrt(2.0*pi))
"""
def gaussian(y):
	return np.exp(-np.power(y, 2)/2.0)/np.sqrt(2.0*np.pi)


# Bimodal Gaussian PDF at y = -3 and y = 3 with sigma = 1
"""
	Gnuplot formula:
	rho(y) = exp(-((y - 3.0)** 2)/2.0) / (2.0*sqrt(2.0*pi)) + exp(-((y + 3.0)** 2)/2.0) / (2.0*sqrt(2.0*pi))
"""
def bimodal(y):
	if (type(y) == np.ndarray):
		return (np.exp(-np.power(y - 3.0, 2)/2.0) + np.exp(-np.power(y + 3.0, 2)/2.0)) / (2.0*np.sqrt(2.0*np.pi))
	else:
		return (K.exp(-(y - 3.0)**2/2.0) + K.exp(-(y + 3.0)**2/2.0)) / (2.0*np.sqrt(2.0*np.pi))


# Asymmetric bimodal Gaussian PDF with peaks at y = 1 and y = -3 and with sigma = 1
"""
	Gnuplot formula:
	rho(y) = exp(-((y - 1.0)** 2)/2.0) / (2.0*sqrt(2.0*pi)) + exp(-((y + 3.0)** 2)/2.0) / (2.0*sqrt(2.0*pi))
"""
def asym_bimodal(y):
	if (type(y) == np.ndarray):
		return (np.exp(-np.power(y - 1.0, 2)/2.0) + np.exp(-np.power(y + 3.0, 2)/2.0)) / (2.0*np.sqrt(2.0*np.pi))
	else:
		return (K.exp(-(y - 1.0)**2/2.0) + K.exp(-(y + 3.0)**2/2.0)) / (2.0*np.sqrt(2.0*np.pi))


# Asymmetric bimodal Gaussian PDF with peaks at y = 1 (sigma = 0.5) and y = -3 (sigma = 1.0)
"""
	Gnuplot formula:
	rho(y) = exp(-(((y - 1.0)/0.5)** 2)/2.0) / (2.0*0.5*sqrt(2.0*pi)) + exp(-((y + 3.0)** 2)/2.0) / (2.0*sqrt(2.0*pi))
"""
def asymvar_bimodal(y):
	if (type(y) == np.ndarray):
		return np.exp(-np.power((y - 1.0)/0.5, 2)/2.0) / (2.0*0.5*np.sqrt(2.0*np.pi)) + np.exp(-np.power(y + 3.0, 2)/2.0) / (2.0*np.sqrt(2.0*np.pi))
	else:
		return K.exp(-((y - 1.0)/0.5)**2/2.0) / (2.0*0.5*np.sqrt(2.0*np.pi)) + K.exp(-(y + 3.0)**2/2.0) / (2.0*np.sqrt(2.0*np.pi))


# Triangular PDF from -1 to 1
"""
	Gnuplot formula:
	max(a, b) = a > b ? a : b
	rho(y) = max(0.0, -abs(y) + 1.0)
"""
def triangular(y):
	if (type(y) == np.ndarray):
		return np.maximum(0.0, -np.abs(y)+1.0)
	else:
		return K.maximum(0.0, -K.abs(y) + 1.0)


# Target PDF to be used in the loss function, set here for simplicity
target = asymvar_bimodal






""" / / / / / LOSS FUNCTION AND METRICS / / / / / """


"""

	Explanation of the loss fct:



	y_pred =	| y_pred(1,1) 					| y_pred(1,2)					|					| y_pred(1,elements_per_vector)					|   first output vector
			 	| y_pred(2,1) 					| y_pred(2,2)					|					|												|   second output vector
				|								|								|					|												|
				|  ...							|  ...							|		...			|	...											|    ...
				|								|								|					|												|
				| y_pred(vectors_per_batch,1)	|  								|					| y_pred(vectors_per_batch,elements_per_vector)	|   last output vector


				first element of output vectors  second element of output vectors		...			  last element of output vectors


	Penalties:
	"vectors":	KDE of each output vector is compared to target.
	"elements":	KDE of each element of the output vectors is compared to target.
	"pot":		Since the KDE and target are only compared in [y_min, y_max], there has to be a penalty for points outside of this interval, a linear potential well.



	The KDE and target will be compared using the Mean Squared Error (MSE) or the Jensen-Shannon-Divergence (JSD). These include integrals, which have to approximated
	numerically by transforming them into a sum over the set y_axis_vectors for the vectors and y_axis_elements for the vector elements. These are linspaces from y_min
	to y_max with elements separated by step_vectors or step_elements, respectively.

"""


# Interval, where the PDFs are compared
y_min = -10.0
y_max = 10.0


# y axis step sizes
step_vectors = 0.1
step_elements = 0.1
# Width parameter h of KDE is chosen to be 2*step, see sampling theorem f_max*2


# y axis for the comparison of the KDE to rho
y_axis_vectors = np.linspace(y_min, y_max, int((y_max-y_min)/step_vectors)+1)
y_axis_elements = np.linspace(y_min, y_max, int((y_max-y_min)/step_elements)+1)
y_axis_vectors_size = y_axis_vectors.size
y_axis_elements_size = y_axis_elements.size


# Weighting factor w for penalty_vectors, penalty_elements
w_vectors = 1.0
w_elements = 1.0


""" Penalties """

"""

	Explanation of ``array calculus'':

	Since in tensorflow, it is not possible to directly access array elements by indexing,
	all calculations have to be done with whole arrays using broadcasting (see e.g. numpy reference).
	The proceed will be explained for the calculation of the KDE of the output vectors:
	In order to calculate the KDE of every output vector, the difference between every output value
	and every element in y_axis_vectors has to be calculated. Therefore, both y_axis_vectors and the
	output matrix y_pred have to be copied multiple times. The y_axis_vectors is simply tiled,
	vectors_per_batch times. The resulting array is 1D and has a length of y_axis_vectors_size*vectors_per_batch.
	Denoting the elements in y_axis_vectors as y_j with j in {1, 2, ..., y_axis_vectors_size}:

	y_axis_vectors_tiled =

	= | y_1 | y_2 | y_3 | ... | y_{y_axis_vectors_size} | y_1 | y_2 | ... |

		->
		size in this direction is y_axis_vectors_size*vectors_per_batch

	Then, denoting the elements in the output matrix y_pred as y_ik, where i indexes through the single output vectors:

	K.tile(y_pred, [1, y_axis_vectors_size]) =

	=	| y_11 | y_12 | y_13 | ... | y_{elements_per_vector} | y_11 | y_12 | ...		|
		| y_21 | y_22 | ....															V	size in this direction is vectors_per_batch
		   ...

		->
		size in this direction is elements_per_vector*y_axis_vectors_size

	Then,

	K.reshape(K.tile(y_pred, [1, y_axis_vectors_size]), [y_axis_vectors_size*vectors_per_batch, elements_per_vector]) =

	= 	| y_11 | y_12 | ...
		| y_11 | y_12 | ...		 |
		  ...					 V size in this direction is y_axis_vectors_size*vectors_per_batch
		| y_21 | y_22 | ...
		  ...

		->
		size in this direction is elements_per_vector

	Then,

	K.transpose(K.reshape(K.tile(y_pred, [1, y_axis_vectors_size]), [y_axis_vectors_size*vectors_per_batch, elements_per_vector])) =

	= 	| y_11 | y_11 | y_11 | ... | y_21 | y_21 | ...			|
		| y_12 | y_12 | ...  									V  size in this direction is elements_per_vector
		| y_13 | ...


		->
		size in this direction is y_axis_vectors_size*vectors_per_batch

	Then, the difference to y_axis_vectors_tiled can be calculated, since the last dimension's size of the above
	array (direction ->) equals the last dimension's size of y_axis_vectors_tiled.

	In order to obtain an array like the below, the result of the difference has to be transposed again:

	delta_y =

	= 	| y_11 - y_1 | y_12 - y_1 | y_13 - y_1 | ... | y_1{elements_per_vector} - y_1 |
		| y_11 - y_2 | y_12 - y_2 | y_13 - y_2 | ... | y_1{elements_per_vector} - y_2 |		|
			...																				V	size in this direction is y_axis_vectors_size*vectors_per_batch
		| y_21 - y_2 | y_22 - y_2 | y_23 - y_2 | ... | y_2{elements_per_vector} - y_2 |

		->
		size in this direction is elements_per_vector

	Now, the KDE of any output vector at y_j may be calculated by appling the kernel function (here Gaussian) to a row of delta_y
	and summing along the -> direction of the above array.
	If this is done for all rows, the resulting array has the value of the KDE of the i-th output vector at y_j at the index
	[i*vectors_per_batch + j] (array is one dimensional, attention: indices start at 0 in the code, in the explanation they start with one!).

	This can directly be compared to target(y_axis_vectors_tiled).

"""


def penalty_vectors(y_true, y_pred):

	error = K.variable(0.0)

	# Tiled y_axis
	y_axis_vectors_tiled = K.tile(K.constant(y_axis_vectors), [vectors_per_batch])

	# Calculating the KDE of the output vectors
	delta_y = K.transpose(K.transpose(K.reshape(K.tile(y_pred, [1, y_axis_vectors_size]), [y_axis_vectors_size*vectors_per_batch, elements_per_vector])) - y_axis_vectors_tiled)
	kde_vectors = K.sum( K.exp(-0.5*((delta_y) / (step_vectors*2.0) )**2), axis=1)/ (np.sqrt(2.0*np.pi)*(step_vectors*2.0)*float(elements_per_vector))

	# Normalization factor step_vectors omitted in the following

	# MSE
	#error = error + K.sum((kde_vectors - target(y_axis_vectors_tiled))**2)/(float(vectors_per_batch))

	# JSD
	error = error + K.sum(	(kde_vectors*K.log(K.maximum(kde_vectors, 1e-08)/K.maximum(target(y_axis_vectors_tiled), 1e-08)) +
							target(y_axis_vectors_tiled)*K.log(K.maximum(target(y_axis_vectors_tiled), 1e-08)/K.maximum(kde_vectors, 1e-08)))
						)/float(vectors_per_batch)

	# Chi^2 distance, actually included in Jensen-Shannon-Div.!
	# error = error + K.sum((kde_vectors - target(y_axis_vectors_tiled))**2/target(y_axis_vectors_tiled))/(float(vectors_per_batch))

	return w_vectors*error
	# y_true unused


def penalty_elements(y_true, y_pred):

	error = K.variable(0.0)

	# Tiled y_axis
	y_axis_elements_tiled = K.tile(K.variable(y_axis_elements), [elements_per_vector])

	# Calculating the KDE of each output vector element across all output vectors
	y_pred = K.transpose(y_pred)
	delta_y = K.transpose(K.transpose(K.reshape(K.tile(y_pred, [1, y_axis_elements_size]), [y_axis_elements_size*elements_per_vector, vectors_per_batch])) - y_axis_elements_tiled)
	kde_elements = K.sum( K.exp(-0.5*((delta_y) / (step_elements*2.0) )**2), axis=1)/ (np.sqrt(2.0*np.pi)*(step_elements*2.0)*float(vectors_per_batch))

	# Normalization factor step_elements omitted in the following

	# MSE
	#error = error + K.sum((kde_elements - target(y_axis_elements_tiled))**2)/(float(elements_per_vector))

	# JSD
	error = error + K.sum(	(kde_elements*K.log(K.maximum(kde_elements, 1e-08)/K.maximum(target(y_axis_elements_tiled), 1e-08)) +
							target(y_axis_elements_tiled)*K.log(K.maximum(target(y_axis_elements_tiled), 1e-08)/K.maximum(kde_elements, 1e-08)))
						)/float(elements_per_vector)

	# Chi^2 distance, actually included in Jensen-Shannon-Div.!
	# error = K.sum((kde_elements - target(y_axis_elements_tiled))**2/target(y_axis_elements_tiled))/(float(elements_per_vector))

	return w_elements*error
	# y_true unused


def penalty_pot(y_true, y_pred):
	return 1.0*K.sum(K.sum(K.maximum(K.abs(y_pred - (y_max + y_min)/2.0) - (y_max - y_min)/2.0, 0.0)))
	# y_true unused


def maximum(y_true, y_pred):
	# Maximum absolute output value, problem: is averaged over batch
	return K.max(K.abs(y_pred))



# Loss function
"""
	This simply sums the penalties listed above.
"""
def loss(y_true, y_pred):
	return penalty_vectors(y_true, y_pred) + penalty_elements(y_true, y_pred) + penalty_pot(y_true, y_pred)





""" / / / / / RUN_MODEL / / / / / """


def run_model(load, N_layers, N_units, opt, epochs, N_testvectors, x_train, x_test, save_results, save_weights_hist, path, plot):

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
		* x_train = generator yielding array[vectors_per_batch][elements_per_vector], training vectors
		* x_test = generator yielding array[N_testvectors][elements_per_vector], testing vectors
		* save_results = (True, False), whether to save the results or not
		* save_weights_hist = (True, False), whether to save the histogram of weights or not
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
		layers = [Dense(N_units, input_shape=(elements_per_vector,), activation="elu")]
		for i in range(1, N_layers-1):
			layers.append(Dense(N_units, activation="elu"))
		layers.append(Dense(elements_per_vector, activation=None))
		for layer in layers:
			model.add(layer)
		# Train the model
		model.compile(loss=loss, optimizer=opt, metrics=[penalty_vectors, penalty_elements, penalty_pot, maximum])
		terminator = K_TerminateOnNaN()
		history = model.fit_generator(generator=x_train(), steps_per_epoch=1, epochs=epochs, workers=1, use_multiprocessing=False, callbacks=[terminator], verbose=1)


	# Let the model map the test-input vectors from x_test to output vectors into y_pred
	y_pred = model.predict_generator(generator=x_test(), steps=1, max_queue_size=1, workers=1, use_multiprocessing=False, verbose=0)
	y_pred_t = np.transpose(y_pred)


	# Calculate the KDE of output vectors
	kde_vectors = np.empty((N_testvectors, y_axis_vectors_size))
	for s in range(0, N_testvectors): # Over all vectors
		for i in range(0, y_axis_vectors_size): # Over all elements in y_axis_vectors
			kde_vectors[s][i] = kde_gauss(y_pred[s] - y_axis_vectors[i], 2.0*step_vectors)
	kde_vectors_mean = np.mean(kde_vectors, axis=0)
	kde_vectors_std = np.std(kde_vectors, axis=0)


	# Calculate the KDE of each output vector element across all output vectors
	kde_elements = np.empty((elements_per_vector, y_axis_elements_size))
	for l in range(0, elements_per_vector): # Over all elements in a vector
		for i in range(0, y_axis_elements_size): # Over all elements in y_axis_elements
			kde_elements[l][i] = kde_gauss(y_pred_t[l] - y_axis_elements[i], 2.0*step_elements)
	kde_elements_mean = np.mean(kde_elements, axis=0) # is equal to kde_vectors_mean! To see this write the formula of the KDE and calculate the mean
	kde_elements_std = np.std(kde_elements, axis=0) # is not equal to kde_vectors_std!


	# Expected probability target rho, has to be done twice, if the axis for vectors/elements are not equal
	rho_vectors = target(y_axis_vectors)
	rho_elements = target(y_axis_elements)


	# Save results
	if (save_results == True):
		# Weights were loaded, where to save results?
		if (load == True):
			raw_in = raw_input("Insert save path: ")
			if (raw_in != ""):
				path = raw_in

		# Make last directory in path if it does not exist
		if (not os.path.isdir(path)):
			os.mkdir(path)

		# Save the weights again to path
		model_json = model.to_json()
		with open(path+"/model.json", "w") as json_file:
			json_file.write(model_json)
			json_file.close()
		model.save_weights(path+"/weights", overwrite=True)

		# History of losses
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
			f.write("#y\tmean\tstd\tdensity of vector i ...\n")
			for i in range(0, y_axis_vectors_size): # Over elements in y_axis_vectors
				f.write(str(y_axis_vectors[i]) + "\t" +
						str(kde_vectors_mean[i]) + "\t" +
						str(kde_vectors_std[i]))
				for s in range(0, N_testvectors): # Over all output vectors
					f.write("\t" + str(kde_vectors[s][i]))
				f.write("\n")
			f.close()

		# KDE of output vectors elements across all output vectors
		with open(path+"/density_elements", "w") as f:
			f.write("#y\tmean\tstd\tdensity of element i ...\n")
			for i in range(0, y_axis_elements_size): # Over elements in y_axis_elements
				f.write(str(y_axis_elements[i]) + "\t" +
						str(kde_elements_mean[i]) + "\t" +
						str(kde_elements_std[i]))
				for l in range(0, elements_per_vector): # Over all output vector elements
					f.write("\t" + str(kde_elements[l][i]))
				f.write("\n")
			f.close()


	# Weights histogram
	if (save_weights_hist == True):
		""" Calculate the histogram of weights """
		weights = model.get_weights()
		# For each unit
		# The amount of bins is calculated using the square-root rule
		weights_histunit = np.empty((N_layers, N_units, int(np.ceil(np.sqrt(N_units))))) # [layer, unit, bin]
		weights_histunitaxis = np.empty((N_layers, N_units, int(np.ceil(np.sqrt(N_units)))+1)) # [layer, unit, bin], one element more, see numpy reference
		for i in range(0, N_layers): # Over all layers
			for j in range(0, N_units): # Over all units
				(weights_histunit[i][j], weights_histunitaxis[i][j]) = np.histogram(weights[2*i][j], int(np.ceil(np.sqrt(N_units))))

		# For each layer
		weights_histlayer = np.empty((N_layers, N_units)) # [layer, bin]
		weights_histlayeraxis = np.empty((N_layers, N_units+1)) # [layer, bin]
		bias_histlayer = np.empty((N_layers, int(np.ceil(np.sqrt(N_units))))) # [layer, bin]
		bias_histlayeraxis = np.empty((N_layers, int(np.ceil(np.sqrt(N_units)))+1)) # [layer, bin]
		for i in range(0, N_layers): # Over all layers
			tmp_histogram = np.histogram(weights[2*i], N_units)
			(weights_histlayer[i], weights_histlayeraxis[i]) = tmp_histogram
			(bias_histlayer[i], bias_histlayeraxis[i]) = np.histogram(weights[2*i+1], int(np.ceil(np.sqrt(N_units))))

		""" Save """
		# Overall histogram
		weights_histall = np.histogram(weights[::2], int(np.ceil(np.sqrt(N_units**2*N_layers))))
		bias_histall = np.histogram(weights[1::2], int(np.ceil(np.sqrt(N_units*N_layers))))

		# Weights histogram for each layer
		for l in range(0, N_layers): # Over all layers
			with open(path+"/weights_histlayer"+str(l), "w") as f:
				f.write("#y\tweights_histlayer\n")
				for i in range(0, weights_histlayeraxis[l].size-1): # Over elements in weights_histlayeraxis[l] (axis of histogram, or "bins")
					f.write(str((weights_histlayeraxis[l][i] + weights_histlayeraxis[l][i+1])/2.0) + "\t" +	str(weights_histlayer[l][i]) + "\n")
				f.close()
			with open(path+"/bias_histlayer"+str(l), "w") as f:
				f.write("#y\tbias_histlayer\n")
				for i in range(0, bias_histlayeraxis[l].size-1): # Over elements in bias_histlayeraxis[l] (axis of histogram, or "bins")
					f.write(str((bias_histlayeraxis[l][i] + bias_histlayeraxis[l][i+1])/2.0) + "\t" +	str(bias_histlayer[l][i]) + "\n")
				f.close()

		# Overall weights histogram
		with open(path+"/weights_histall", "w") as f:
			f.write("#y\tweights_histall\n")
			for i in range(0, weights_histall[1].size-1): # Over elements in weights_histall[1] (axis of histogram, or "bins")
				f.write(str((weights_histall[1][i] + weights_histall[1][i+1])/2.0) + "\t" +	str(weights_histall[0][i]) + "\n")
			f.close()
		with open(path+"/bias_histall", "w") as f:
			f.write("#y\tbias_histall\n")
			for i in range(0, bias_histall[1].size-1): # Over elements in bias_histall[1] (axis of histogram, or "bins")
				f.write(str((bias_histall[1][i] + bias_histall[1][i+1])/2.0) + "\t" +	str(bias_histall[0][i]) + "\n")
			f.close()


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

		# Mean of KDE of output vectors
		plt.figure(1)
		plt.title("Mean KDE of output vectors")
		plt.xlabel("y")
		plt.ylabel("PDF")
		plt.plot(y_axis_vectors, rho_vectors, "b", label="target PDF")
		plt.plot(y_axis_vectors, kde_vectors_mean, "r+", label="mean KDE")
		plt.errorbar(y_axis_vectors, kde_vectors_mean, kde_vectors_std, ls="", c="r")
		plt.legend(loc='best', ncol=1)

		# Mean of KDE of output vector elements
		plt.figure(2)
		plt.title("Mean KDE of output vector elements")
		plt.xlabel("y")
		plt.ylabel("PDF")
		plt.plot(y_axis_elements, rho_elements, "b", label="target PDF")
		plt.plot(y_axis_elements, kde_elements_mean, "r+", label="mean KDE")
		plt.errorbar(y_axis_elements, kde_elements_mean, kde_elements_std, ls="", c="r")
		plt.legend(loc='best', ncol=1)

		# Two examples of KDEs of output vectors
		plt.figure(3)
		plt.xlabel("y")
		plt.ylabel("PDF")
		plt.title("Example KDEs of output vectors")
		plt.plot(y_axis_vectors, rho_vectors, "b", label="target PDF")
		plt.plot(y_axis_vectors, kde_vectors[0], "r+", label="Example KDE 1")
		plt.plot(y_axis_vectors, kde_vectors[1], "gx", label="Example KDE 2")
		plt.legend(loc='best', ncol=1)

		# Two examples of KDEs of output vector elements
		plt.figure(4)
		plt.xlabel("y")
		plt.ylabel("PDF")
		plt.title("Example KDEs of output vector elements")
		plt.plot(y_axis_elements, rho_elements, "b", label="target PDF")
		plt.plot(y_axis_elements, kde_elements[0], "r+", label="Example KDE 1")
		plt.plot(y_axis_elements, kde_elements[2], "gx", label="Example KDE 2")
		plt.legend(loc='best', ncol=1)

		# Two examples of output vectors
		plt.figure(5)
		plt.xlabel("element")
		plt.ylabel("y")
		plt.title("Examples of output vectors")
		plt.plot(y_pred[0], "bo", label="Example 1")
		plt.plot(y_pred[1], "r+", label="Example 2")
		plt.legend(loc='best', ncol=1)

		# Two examples of output vector elements
		plt.figure(6)
		plt.xlabel("element")
		plt.ylabel("y")
		plt.title("Examples of output vector elements")
		plt.plot(y_pred_t[0], "bo", label="Example 1")
		plt.plot(y_pred_t[1], "r+", label="Example 2")
		plt.legend(loc='best', ncol=1)

		plt.show()


	# Clean up memory
	K.clear_session()
	tf_reset_default_graph()

	# Return prediction == output vectors for x_test
	return y_pred
#





""" / / / / / TESTS / / / / / """


"""
	In this section, all performed tests are listed, before starting anything, please pay attention that
	the directory tree already exists, the function run_model can only create the last directory given in
	the path argument, if it does not exist. If any test shall be started, simply put a True in the
	corresponding if condition. Some tests involve loading weights from previous runs. 	If these weights
	do not exist yet, please set the load argument of run_model to False and perform a new training,
	the number of epochs will be given.

"""



"""--- Mean Squared Error vs. Jensen-Shannon-Divergence ---"""
"""
	In order to perform this test, comment the MSE/JSD part in the penalty functions and
	uncomment the other. Secondly, set the measure variable below to the proper value.
	For example if the tests for the MSE are to be performed, comment the JSD in the penalty
	functions and uncomment the MSE part, finally set the measure variable below to "MSE".

"""
if (False):
	# Set target
	target = asymvar_bimodal

	# Used measure
	measure = "MSE"

	if (measure == "MSE"):
		# MSE
		run_model(	load				= False,
					N_layers			= 32,
					N_units				= elements_per_vector,
					opt					= Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.004),
					epochs				= 1000,
					N_testvectors		= N_testvectors,
					x_train				= uniform_gen,
					x_test				= uniform_test_gen,
					save_results		= True,
					save_weights_hist	= False,
					path				= "Tests/Measure/MSE",
					plot				= False
				)

		# MSE with longer training
		run_model(	load				= False,
					N_layers			= 32,
					N_units				= elements_per_vector,
					opt					= Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.004),
					epochs				= 10000,
					N_testvectors		= N_testvectors,
					x_train				= uniform_gen,
					x_test				= uniform_test_gen,
					save_results		= True,
					save_weights_hist	= False,
					path				= "Tests/Measure/MSE_10000",
					plot				= False
				)

	elif (measure == "JSD"):
		# JSD
		run_model(	load				= False,
					N_layers			= 32,
					N_units				= elements_per_vector,
					opt					= Adam(lr= 0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.004),
					epochs				= 1000,
					N_testvectors		= N_testvectors,
					x_train				= uniform_gen,
					x_test				= uniform_test_gen,
					save_results		= True,
					save_weights_hist	= False,
					path				= "Tests/Measure/JSD",
					plot				= False
				)





"""--- Elements loss ---"""
"""
	This test concerns the influence of penalty_elements() on the resulting output values.

"""
if (False):
	# Set target
	target = asymvar_bimodal

	# Normal run
	path = "Tests/Elementsloss/with"
	y_pred = run_model(	load				= False,
						N_layers			= 32,
						N_units				= elements_per_vector,
						opt					= Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.004),
						epochs				= 1000,
						N_testvectors		= N_testvectors,
						x_train				= uniform_gen,
						x_test				= uniform_test_gen,
						save_results		= True,
						save_weights_hist	= False,
						path				= path,
						plot				= False
					)

	# Save prediction
	with open(path+"/prediction", "w") as f:
		f.write("#vector_i ...\n")
		for i in range(0, elements_per_vector): # over all elements
			for j in range(0, vectors_per_batch): # over all vectors
				f.write(str(y_pred[j][i]) + "\t")
			f.write("\n")
		f.close()


	# Removing element loss from loss function
	def loss(y_true, y_pred):
		return penalty_vectors(y_true, y_pred) + penalty_pot(y_true, y_pred)

	# Running without element loss
	path = "Tests/Elementsloss/without"
	y_pred = run_model(	load				= False,
						N_layers			= 32,
						N_units				= elements_per_vector,
						opt					= Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.004),
						epochs				= 1000,
						N_testvectors		= N_testvectors,
						x_train				= uniform_gen,
						x_test				= uniform_test_gen,
						save_results		= True,
						save_weights_hist	= False,
						path				= path,
						plot				= False
					)

	# Save prediction
	with open(path+"/prediction", "w") as f:
		f.write("#vector_i ...\n")
		for i in range(0, elements_per_vector): # over all elements
			for j in range(0, vectors_per_batch): # over all vectors
				f.write(str(y_pred[j][i]) + "\t")
			f.write("\n")
		f.close()

	# Add element loss to loss function again
	def loss(y_true, y_pred):
		return penalty_vectors(y_true, y_pred) + penalty_elements(y_true, y_pred) + penalty_pot(y_true, y_pred)





"""--- Optimizer ---"""
"""
	Two optimizers (SGD, Adam) are tested for different combinations of parameters.

"""
if (False):
	# Set target
	target = asymvar_bimodal

	# SGD
	lr_list = [0.0001, 0.0002, 0.0003, 0.0004]
	decay_list = [0.0]
	momentum_list = [0.0, 0.1, 0.2, 0.3, 0.4]
	for lr in lr_list:
		for decay in decay_list:
			for momentum in momentum_list:
				run_model(	load				= False,
							N_layers			= 32,
							N_units				= elements_per_vector,
							opt					= SGD(lr=lr, momentum=momentum, decay=decay, nesterov=False),
							epochs				= 1000,
							N_testvectors		= N_testvectors,
							x_train				= uniform_gen,
							x_test				= uniform_test_gen,
							save_results		= True,
							save_weights_hist	= False,
							path				= "Tests/Optimizers/SGD/lr"+str(lr)+"_decay"+str(decay)+"_momentum"+str(momentum),
							plot				= False
						)

	# Adam
	lr_list = [0.00005, 0.00010, 0.00015, 0.00020]
	decay_list = [0.0, 0.001, 0.002, 0.003, 0.004]
	for lr in lr_list:
		for decay in decay_list:
			run_model(	load				= False,
						N_layers			= 32,
						N_units				= elements_per_vector,
						opt					= Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay),
						epochs				= 1000,
						N_testvectors		= N_testvectors,
						x_train				= uniform_gen,
						x_test				= uniform_test_gen,
						save_results		= True,
						save_weights_hist	= False,
						path				= "Tests/Optimizers/Adam/lr"+str(lr)+"_decay"+str(decay),
						plot				= False
					)





"""--- Network dimensions ---"""
"""
	The optimal amount of layers and units is determined.

"""
if (False):
	# Set target
	target = asymvar_bimodal

	# Testing N_layers
	N_layers_list = np.arange(5, 61, 5)
	opt_param_list = [{"lr": 0.00002, "decay": 0.0002}, {"lr": 0.0002, "decay": 0.002}, {"lr": 0.002, "decay": 0.02}]
	for N_layers in N_layers_list:
		for opt_param in opt_param_list:
			for i in range(1,5):
				run_model(	load				= False,
							N_layers			= N_layers,
							N_units				= elements_per_vector,
							opt					= Adam(lr=opt_param["lr"], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=opt_param["decay"]),
							epochs				= 1000,
							N_testvectors		= N_testvectors,
							x_train				= uniform_gen,
							x_test				= uniform_test_gen,
							save_results		= True,
							save_weights_hist	= False,
							path				= "Tests/Sizing/Layers/lr"+str(opt_param["lr"])+"_decay"+str(opt_param["decay"])+"/N_layers"+str(N_layers)+"_run"+str(i),
							plot				= False
						)

	# Testing N_units
	N_units_list = np.arange(100, 1050, 100)
	opt_param_list = [	{"lr": 0.00002, "decay": 0.0002}, {"lr": 0.0002, "decay": 0.002}, {"lr": 0.002, "decay": 0.02}]
	for opt_param in opt_param_list:
		for N_units in N_units_list:
			for i in range(1,5):
				run_model(	load				= False,
							N_layers			= 32,
							N_units				= N_units,
							opt					= Adam(lr=opt_param["lr"], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=opt_param["decay"]),
							epochs				= 1000,
							N_testvectors		= N_testvectors,
							x_train				= uniform_gen,
							x_test				= uniform_test_gen,
							save_results		= True,
							save_weights_hist	= False,
							path				= "Tests/Sizing/Units/lr"+str(opt_param["lr"])+"_decay"+str(opt_param["decay"])+"/N_units"+str(N_units)+"_run"+str(i),
							plot				= False
						)





"""--- Input density ---"""
"""
	Testing different input PDFs on three output PDFs

"""
if (False):

	""" Testing uniform [0, 1] """
	# Shows that this input PDF leads to worse results
	# Training data generator
	def uniform01_gen():
		while True:
			yield (np.random.uniform(low=0.0, high=1.0, size=(vectors_per_batch, elements_per_vector)), np.empty((vectors_per_batch, elements_per_vector)))

	# Generator for testing data
	def uniform01_test_gen():
		while True:
			yield np.random.uniform(low=0.0, high=1.0, size=(N_testvectors, elements_per_vector))

	# Set target
	target = asymvar_bimodal

	# Run with uniform PDF [0, 1] as input
	run_model(	load				= False,
				N_layers			= 10,
				N_units				= elements_per_vector,
				opt					= Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.005),
				epochs				= 1000,
				N_testvectors		= N_testvectors,
				x_train				= uniform01_gen,
				x_test				= uniform01_test_gen,
				save_results		= True,
				save_weights_hist	= False,
				path				= "Tests/Input/uniform01_in",
				plot				= False
			)


	""" Define target PDFs rho """
	target_list = [{"name": "triangular", "func": triangular}, {"name": "bimodal", "func": bimodal}, {"name": "asymvar_bimodal", "func": asymvar_bimodal}]


	""" Testing uniform [-1, 1] """

	# Run for all target PDFs with the uniform PDF over [-1, 1] as input PDF
	for target_PDF in target_list:
		target = target_PDF["func"]
		for i in range(1, 5):
			run_model(	load				= False,
						N_layers			= 10,
						N_units				= elements_per_vector,
						opt					= Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.005),
						epochs				= 1000,
						N_testvectors		= N_testvectors,
						x_train				= uniform_gen,
						x_test				= uniform_test_gen,
						save_results		= True,
						save_weights_hist	= False,
						path				= "Tests/Input/"+target_PDF["name"]+"_target/uniform_run"+str(i),
						plot				= False
					)


	""" Testing Gaussian with zero mean and std dev 1.0 """
	# Training data generator
	def gaussian_gen():
		while True:
			yield (np.random.normal(loc=0.0, scale=1.0, size=(vectors_per_batch,elements_per_vector)), np.empty((vectors_per_batch,elements_per_vector)))

	# Generator for testing data
	def gaussian_test_gen():
		while True:
			yield np.random.normal(loc=0.0, scale=1.0, size=(N_testvectors,elements_per_vector))

	# Run for all target PDFs with the Gaussian input PDF
	for target_PDF in target_list:
		target = target_PDF["func"]
		for i in range(1, 5):
			run_model(	load				= False,
						N_layers			= 10,
						N_units				= elements_per_vector,
						opt					= Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.005),
						epochs				= 1000,
						N_testvectors		= N_testvectors,
						x_train				= gaussian_gen,
						x_test				= gaussian_test_gen,
						save_results		= True,
						save_weights_hist	= False,
						path				= "Tests/Input/"+target_PDF["name"]+"_target/gaussian_run"+str(i),
						plot				= False
					)


	""" Testing exponential with mean 1, mirrored at the y-axis to produce zero mean """
	# Training data generator
	def exp_gen():
		while True:
			yield ((2.0*np.random.randint(2, size=(vectors_per_batch,elements_per_vector)) - 1.0)*np.random.exponential(scale=1.0, size=(vectors_per_batch,elements_per_vector)),
					np.empty((vectors_per_batch,elements_per_vector)))

	# Generator for testing data
	def exp_test_gen():
		while True:
			yield (2.0*np.random.randint(2, size=(vectors_per_batch,elements_per_vector)) - 1.0)*np.random.exponential(scale=1.0, size=(vectors_per_batch,elements_per_vector))

	# Run for all target PDFs with the exponential input PDF
	for target in target_list:
		target = target_PDF["func"]
		for i in range(1, 5):
			run_model(	load				= False,
						N_layers			= 10,
						N_units				= elements_per_vector,
						opt					= Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.005),
						epochs				= 1000,
						N_testvectors		= N_testvectors,
						x_train				= exp_gen,
						x_test				= exp_test_gen,
						save_results		= True,
						save_weights_hist	= False,
						path				= "Tests/Input/"+target_PDF["name"]+"_target/exp_run"+str(i),
						plot				= False
					)





"""--- Results ---"""
"""
	This test shows the performance of the model.

"""
if (False):
	# Run for asymvar_bimodal
	target = asymvar_bimodal
	path = "Tests/Results/asymvar_bimodal"
	y_pred = run_model(	load				= False,
						N_layers			= 32,
						N_units				= elements_per_vector,
						opt					= Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.002),
						epochs				= 10000,
						N_testvectors		= N_testvectors,
						x_train				= uniform_gen,
						x_test				= uniform_test_gen,
						save_results		= True,
						save_weights_hist	= False,
						path				= path,
						plot				= False
					)

	# Save prediction
	with open(path+"/prediction", "w") as f:
		f.write("#vector_i ...\n")
		for i in range(0, elements_per_vector): # over all elements
			for j in range(0, vectors_per_batch): # over all vectors
				f.write(str(y_pred[j][i]) + "\t")
			f.write("\n")
		f.close()

	# Run for triangular
	target = triangular
	path = "Tests/Results/triangular"
	y_pred = run_model(	load				= False,
						N_layers			= 32,
						N_units				= elements_per_vector,
						opt					= Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.002),
						epochs				= 10000,
						N_testvectors		= N_testvectors,
						x_train				= uniform_gen,
						x_test				= uniform_test_gen,
						save_results		= True,
						save_weights_hist	= False,
						path				= path,
						plot				= False
					)

	# Save prediction
	with open(path+"/prediction", "w") as f:
		f.write("#vector_i ...\n")
		for i in range(0, elements_per_vector): # over all elements
			for j in range(0, vectors_per_batch): # over all vectors
				f.write(str(y_pred[j][i]) + "\t")
			f.write("\n")
		f.close()

	# Run for triangular using MSE
	# Uncomment MSE and comment JSD in loss-function!
	target = triangular
	path = "Tests/Results/triangularMSE"
	y_pred = run_model(	load				= False,
						N_layers			= 32,
						N_units				= elements_per_vector,
						opt					= Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.004),
						epochs				= 10000,
						N_testvectors		= N_testvectors,
						x_train				= uniform_gen,
						x_test				= uniform_test_gen,
						save_results		= True,
						save_weights_hist	= False,
						path				= path,
						plot				= False
					)

	# Save prediction
	with open(path+"/prediction", "w") as f:
		f.write("#vector_i ...\n")
		for i in range(0, elements_per_vector): # over all elements
			for j in range(0, vectors_per_batch): # over all vectors
				f.write(str(y_pred[j][i]) + "\t")
			f.write("\n")
		f.close()





"""--- Different Input than in training ---"""
"""
	In this test, it is showed that a trained model is also able to map any input drawn from a
	PDF with zero mean to the target PDF.

"""
if (False):
	# Set target
	target = asymvar_bimodal

	# Generates input vectors drawn from a bimodal asymmetric Gaussian with zero mean
	def diffin_test_gen():
		while True:
			tmp = np.empty((2, N_testvectors, elements_per_vector))
			tmp[0] = np.random.normal(-3.0, 0.5, size=(N_testvectors, elements_per_vector))
			tmp[1] = np.random.normal(3.0, 1.0, size=(N_testvectors, elements_per_vector))
			tmp = np.reshape(np.transpose(tmp, [1, 2, 0]), [N_testvectors*elements_per_vector, 2])
			test_vectors = tmp[np.arange(0, N_testvectors*elements_per_vector), np.random.randint(0, 2, size=(N_testvectors*elements_per_vector))]
			test_vectors = np.reshape(test_vectors, [N_testvectors, elements_per_vector])
			yield test_vectors

	# Run
	path = "Tests/Differentinput"
	diffin_y_pred = run_model(	load				= True,
								N_layers			= 32,
								N_units				= elements_per_vector,
								opt					= Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.002),
								epochs				= 0,
								N_testvectors		= N_testvectors,
								x_train				= uniform_gen,
								x_test				= diffin_test_gen,
								save_results		= True,
								save_weights_hist	= False,
								path				= path,
								plot				= False
							)

	# Save prediction
	with open(path+"/prediction", "w") as f:
		f.write("#element_i ...\n")
		for i in range(0, N_testvectors): # over all predicted vectors
			for j in range(0, elements_per_vector): # over all vector elements
				f.write(str(diffin_y_pred[i][j]) + "\t")
			f.write("\n")
		f.close()





"""--- Weights ---"""
"""
	This test examines the histograms of weights. IMPORTANT: The weights for the models
	used in this test are loaded from files. If these files do not exist, set the load
	argument of run_model() to False wherever it is set to True and the epochs argument
	to 10000. Then start the test and reset the arguments to the previous state afterwards.

"""
if (False):
	# Set target
	target = asymvar_bimodal

	# Weights of a trained network with 32 layers and 500 units each
	run_model(	load				= True,
				N_layers			= 32,
				N_units				= elements_per_vector,
				opt					= Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.002),
				epochs				= 0,
				N_testvectors		= N_testvectors,
				x_train				= uniform_gen,
				x_test				= uniform_test_gen,
				save_results		= True,
				save_weights_hist	= True,
				path				= "Tests/Weights/trained",
				plot				= False
			)


	# Weights of an untrained network of equal dimension as above
	run_model(	load				= False,
				N_layers			= 32,
				N_units				= elements_per_vector,
				opt					= Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.002),
				epochs				= 0,
				N_testvectors		= N_testvectors,
				x_train				= uniform_gen,
				x_test				= uniform_test_gen,
				save_results		= True,
				save_weights_hist	= True,
				path				= "Tests/Weights/glorotuniform",
				plot				= False
			)


	# Weights of an trained network with 10 layers and 500 units each
	run_model(	load				= True,
				N_layers			= 10,
				N_units				= elements_per_vector,
				opt					= Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.02),
				epochs				= 0,
				N_testvectors		= N_testvectors,
				x_train				= uniform_gen,
				x_test				= uniform_test_gen,
				save_results		= True,
				save_weights_hist	= True,
				path				= "Tests/Weights/10layers",
				plot				= False
			)





"""--- Varying a fixed amount of elements only ---"""
"""
	This test investigates the model viewed as a function. This is done
	by varying a fixed amount of input values and set the others to zero.
	Also, a test is perform where only the first two input values are varied
	and the first output value is obtained as a function of these two input
	values. IMPORTANT: The weights of the used models are loaded from files.
	If these files do not exist, set the load argument of run_model() to False
	wherever it is set to True and the epochs argument to 10000. Then start the
	test and reset the arguments to the previous state afterwards.

"""
if (False):

	# Set target
	target = asymvar_bimodal

	# Generator for varying 1 ... 10 input values simultaneously
	N_vary = 10
	def vary_test_gen():
		while True:
			test_vectors = np.zeros((N_vary*200, elements_per_vector)) # 1 to 10 varying elements and 200 elements in [-1,1]
			for l in range(0, N_vary):
				test_vectors.T[l] = np.concatenate((np.zeros(200*l), np.tile(np.linspace(-1.0, 1.0, 200), N_vary-l)), axis=0)
			yield test_vectors

	# Predict with vary_test_gen generator
	path = "Tests/Varying"
	vary_y_pred = run_model(	load				= True,
								N_layers			= 32,
								N_units				= elements_per_vector,
								opt					= Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.002),
								epochs				= 0,
								N_testvectors		= N_vary*200,
								x_train				= uniform_gen,
								x_test				= vary_test_gen,
								save_results		= False,
								save_weights_hist	= False,
								path				= path,
								plot				= False
							)

	# Save prediction
	for l in range(0, N_vary):
		with open(path+"/N_vary"+str(l+1), "w") as f:
			f.write("#element_i ...\n")
			for i in range(0, 200): # Over all predicted vectors for a fixed amount of varying vector elements
				for j in range(0, elements_per_vector): # Over all vector entries
					f.write(str(vary_y_pred[l*200+i][j]) + "\t")
				f.write("\n")
			f.close()


	# Set all input value equal to each other and vary them all simultaneously
	def vary_all_test_gen():
		while True:
			test_vectors = np.tile(np.linspace(-1.0, 1.0, 200), [elements_per_vector,1]).T
			yield test_vectors

	# Predict with vary_all_test_gen generator
	vary_y_pred = run_model(	load				= True,
								N_layers			= 32,
								N_units				= elements_per_vector,
								opt					= Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.002),
								epochs				= 0,
								N_testvectors		= 200,
								x_train				= uniform_gen,
								x_test				= vary_all_test_gen,
								save_results		= False,
								save_weights_hist	= False,
								path				= path,
								plot				= False
							)

	# Save prediction
	with open(path+"/vary_all", "w") as f:
		f.write("#element_i ...\n")
		for i in range(0, 200): # Over all predicted vectors
			for j in range(0, elements_per_vector): # Over all vector entries
				f.write(str(vary_y_pred[i][j]) + "\t")
			f.write("\n")
		f.close()


	# Calculate the first output element versus the first two input elements for a 2D plot
	def vary2D_test_gen():
		while True:
			test_vectors = np.zeros(((100+1)*(100+1), elements_per_vector)) # grid for first and second input
			test_vectors.T[0] = np.tile(np.linspace(-1.0, 1.0, 100+1), 100+1)
			for i in range(0, 100+1):
				test_vectors.T[1][i*(100+1):(i+1)*(100+1)] = (2.0/100.0*i-1.0)*np.ones(100+1)
			yield test_vectors

	# Predict with vary2D_test_gen generator
	vary_y_pred = run_model(	load				= True,
								N_layers			= 32,
								N_units				= elements_per_vector,
								opt					= Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.002),
								epochs				= 0,
								N_testvectors		= (100+1)*(100+1),
								x_train				= uniform_gen,
								x_test				= vary2D_test_gen,
								save_results		= False,
								save_weights_hist	= False,
								path				= path,
								plot				= False
							)

	# Save prediction
	with open(path+"/vary2D", "w") as f:
		f.write("#input1\tinput2\telement_0\n")
		for x_1 in range(0, 100+1): # second input vector element
			for x_0 in range(0, 100+1): # first input vector element
				f.write(str(2.0/100.0*x_0-1.0) + "\t" + str(2.0/100.0*x_1-1.0) + "\t" + str(vary_y_pred[x_0 + (100+1)*x_1][10]) + "\n")
			f.write("\n")
		f.close()


""" / / / / / END OF FILE / / / / / """