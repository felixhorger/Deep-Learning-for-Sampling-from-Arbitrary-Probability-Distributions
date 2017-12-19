"""

Sample Mapper - Input dim 1 version

A Multilayer-Perceptron is trained to map a random sample of the uniform PDF over [-1, 1]
to a random sample of an arbitrary PDF.

"""





""" / / / / / IMPORTS / / / / / """


# Import os
import os


# Import numpy
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
elements_per_vector = 1
vectors_per_batch = 1000


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
N_testvectors = 100000
def test_gen():
	while True:
		yield np.random.uniform(low=-1.0, high=1.0, size=(N_testvectors, elements_per_vector))





""" / / / / / PDF APPROXIMATION / / / / / """


# Kernel density estimation (KDE) using a gaussian kernel function.
"""
	This produces a single value of the KDE of a vector y at position y', with delta_y[i] = y[i] - y'.
	The width of the kernel function can be defined by h.
"""
def kde_gauss(delta_y, h):
	return np.sum( np.exp(-0.5*((delta_y) / (h) )**2), 0)/ (np.sqrt(2.0*np.pi)*(h)*float(delta_y.size))





""" / / / / / DENSITY / / / / / """


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


# Asymmetric bimodal gaussian PDF with peaks at y = 1 and y = -3 and with sigma = 1
"""
	Gnuplot formula:
	rho(y) = exp(-((y - 1.0)** 2)/2.0) / (2.0*sqrt(2.0*pi)) + exp(-((y + 3.0)** 2)/2.0) / (2.0*sqrt(2.0*pi))
"""
def asym_bimodal(y):
	if (type(y) == np.ndarray):
		return (np.exp(-np.power(y - 1.0, 2)/2.0) + np.exp(-np.power(y + 3.0, 2)/2.0)) / (2.0*np.sqrt(2.0*np.pi))
	else:
		return (K.exp(-(y - 1.0)**2/2.0) + K.exp(-(y + 3.0)**2/2.0)) / (2.0*np.sqrt(2.0*np.pi))


# Asymmetric bimodal gaussian PDF with peaks at y = 1 (sigma = 0.5) and y = -3 (sigma = 1.0)
"""
	Gnuplot formula:
	rho(y) = exp(-(((y - 1.0)/0.5)** 2)/2.0) / (2.0*0.5*sqrt(2.0*pi)) + exp(-((y + 3.0)** 2)/2.0) / (2.0*sqrt(2.0*pi))
"""
def asymvar_bimodal(y):
	if (type(y) == np.ndarray):
		return np.exp(-np.power((y - 1.0)/0.5, 2)/2.0) / (2.0*0.5*np.sqrt(2.0*np.pi)) + np.exp(-np.power(y + 3.0, 2)/2.0) / (2.0*np.sqrt(2.0*np.pi))
	else:
		return K.exp(-((y - 1.0)/0.5)**2/2.0) / (2.0*0.5*np.sqrt(2.0*np.pi)) + K.exp(-(y + 3.0)**2/2.0) / (2.0*np.sqrt(2.0*np.pi))


# Symmetric triangular PDF from -1 to 1
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


# Probability density function to be used in the loss function, set here for simplicity
density = asymvar_bimodal






""" / / / / / LOSS FUNCTION AND METRICS / / / / / """


"""

	Explanation of the loss fct:



	y_pred =	| y_pred(1,1) 					|    first output vector
			 	| y_pred(2,1) 					|    second output vector
				|								|
				|  ...							|
				|								|
				| y_pred(vectors_per_batch,1)	|


				first element of output vectors


	Penalties:
	"vectors":	Not used here, KDE of each output vector is compared to density. Falls away since only a single value is mapped to a single value.
	"elements":	KDE of each element of the output vectors is compared to density.
	"pot":		Since the KDE and density are only compared in [y_min, y_max], there has to be a penalty for points outside of this interval, a linear potential well.



	The KDE and density will be compared using the Mean Squared Error (MSE) or the Jensen-Shannon-Divergence (JSD). These include integrals, which have to approximated
	numerically by transforming them into a sum over the set y_axis_vectors for the vectors and y_axis_elements for the vector elements. These are linspaces from y_min
	to y_max with elements separated by step_vectors or step_elements, respectively.

"""


# Interval, where the PDFs are compared
y_min = -10.0
y_max = 10.0


# y axis steps sizes
step_elements = 0.1
# sigma of KDE is chosen to be 2*step, see sampling theorem f_max*2


# y axis for the comparison of the KDE to rho
y_axis_elements = np.linspace(y_min, y_max, int((y_max-y_min)/step_elements)+1)
y_axis_elements_size = y_axis_elements.size


# Weighting factor w for penalty_elements
w_elements = 1.0


""" Penalties """

def penalty_elements(y_true, y_pred):

	error = K.variable(0.0)

	# Tiled y_axis
	y_axis_elements_tiled = K.tile(K.constant(y_axis_elements), [elements_per_vector])

	# Calculating the KDE of each output vector element across all output vectors
	y_pred = K.transpose(y_pred)
	delta_y = K.transpose(K.transpose(K.reshape(K.tile(y_pred, [1, y_axis_elements_size]), [y_axis_elements_size*elements_per_vector, vectors_per_batch])) - y_axis_elements_tiled)
	kde_elements = K.sum( K.exp(-0.5*((delta_y) / (step_elements*2.0) )**2), axis=1)/ (np.sqrt(2.0*np.pi)*(step_elements*2.0)*float(vectors_per_batch))

	# Normalization factor step_elements omitted in the following

	# MSE
	#error = K.sum((kde_elements - density(y_axis_elements_tiled))**2)/(float(elements_per_vector))

	# JSD
	error = error + K.sum(kde_elements*K.log(K.maximum(kde_elements, 1e-08)/K.maximum(density(y_axis_elements_tiled), 1e-08)) + density(y_axis_elements_tiled)*K.log(K.maximum(density(y_axis_elements_tiled), 1e-08)/K.maximum(kde_elements, 1e-08)))/float(elements_per_vector)

	# Chi^2 distance, actually included in Jensen-Shannon-Div.!
	# error = K.sum((kde_elements - density(y_axis_elements_tiled))**2/density(y_axis_elements_tiled))/(float(elements_per_vector))

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
	return penalty_elements(y_true, y_pred) + penalty_pot(y_true, y_pred)





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
		* x_train = generator yielding array[vectors_per_batch][elements_per_vector], training vectors
		* x_test = generator yielding array[N_testvectors][elements_per_vector], testing vectors
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
		# Needed if training will go on, not implemented yet:
		#if (epochs > 0):
		#	model.compile(loss=loss, optimizer=opt, metrics=[penalty_elements, penalty_pot, maximum])
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
		model.compile(loss=loss, optimizer=opt, metrics=[penalty_elements, penalty_pot, maximum])
		terminator = K_TerminateOnNaN()
		history = model.fit_generator(generator=x_train(), steps_per_epoch=1, epochs=epochs, workers=1, use_multiprocessing=False, callbacks=[terminator], verbose=1)


	# Let the model map the test-input vectors from x_test to output vectors into y_pred
	y_pred = model.predict_generator(generator=x_test(), steps=1, max_queue_size=1, workers=1, use_multiprocessing=False, verbose=0)
	y_pred_t = np.transpose(y_pred)

	# Calculate the KDE of each output vector element across all output vectors
	kde_elements = np.empty((elements_per_vector, y_axis_elements_size))
	for l in range(0, elements_per_vector): # Over all elements in a vector
		for i in range(0, y_axis_elements_size): # Over all elements in y_axis_elements
			kde_elements[l][i] = kde_gauss(y_pred_t[l] - y_axis_elements[i], 2.0*step_elements)
	kde_elements_mean = np.mean(kde_elements, axis=0) # is equal to kde_vectors_mean!
	kde_elements_std = np.std(kde_elements, axis=0) # is not equal to kde_vectors_std!
	# Remark: Mean and std is actually senseless here, performed over a single element...

	# Expected probability density
	rho_elements = density(y_axis_elements)


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
				f.write("#penalty_elements\tpenalty_pot\tmaximum\n")
				for i in range(0, len(history.history["penalty_elements"])):
					f.write(str(history.history["penalty_elements"][i]) + "\t" +
							str(history.history["penalty_pot"][i]) + "\t" +
							str(history.history["maximum"][i]) + "\n")
				f.close()
		else:
			print "History not saved, only possible if load == False and epochs > 0"

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


	# Weight histogram saving ommited for clarity


	# Plot
	if (plot == True):
		# History of losses
		if ((load == False) and (epochs > 0)):
			plt.figure(0)
			plt.title("Losses")
			plt.xlabel("training step")
			plt.ylabel("loss")
			plt.plot(history.history["penalty_elements"], "g", label="penalty_elements")
			plt.plot(history.history["penalty_pot"], "r", label="penalty_pot")
			plt.legend(loc='best', ncol=1)

		# KDE of output values
		plt.figure(2)
		plt.title("KDE of output values")
		plt.xlabel("y")
		plt.ylabel("PDF")
		plt.plot(y_axis_elements, rho_elements, "b", label="target PDF")
		plt.plot(y_axis_elements, kde_elements_mean, "r+", label="KDE")
		plt.errorbar(y_axis_elements, kde_elements_mean, kde_elements_std, ls="", c="r")
		plt.legend(loc='best', ncol=1)

		# Output values
		plt.figure(6)
		plt.xlabel("element")
		plt.ylabel("y")
		plt.title("output vector elements")
		plt.plot(y_pred_t[0], "bo")

		plt.show()


	# Clean up memory
	K.clear_session()
	tf_reset_default_graph()

	# Return prediction == output vectors for x_test
	return y_pred
#





""" / / / / / TESTS / / / / / """
"""
	The model for mapping single values to single values is trained and further fed with
	a linspace over [-1,1], in order to show the similarity of the model as a function
	compared to the CDF that corresponds to the target PDF.

"""

# Normal run
path = "Tests"
run_model(	load			= False,
			N_layers		= 10,
			N_units			= 500,
			opt				= Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.02),
			epochs			= 10000,
			N_testvectors	= N_testvectors,
			x_train			= uniform_gen,
			x_test			= test_gen,
			save_results	= True,
			path			= path,
			plot			= False
		)


# Varying the single input continuously
def vary_test_gen():
	while True:
		test_vectors = np.zeros((200, elements_per_vector))
		test_vectors.T[0] = np.linspace(-1.0, 1.0, 200)
		yield test_vectors


vary_y_pred = run_model(	load			= True,
							N_layers		= 10,
							N_units			= 500,
							opt				= Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.02),
							epochs			= 0,
							N_testvectors	= 200,
							x_train			= uniform_gen,
							x_test			= vary_test_gen,
							save_results	= False,
							path			= path,
							plot			= False
						)

# Save prediction
with open(path+"/prediction", "w") as f:
		f.write("#in\tout\n")
		vary_axis = np.linspace(-1.0, 1.0, 200)
		for i in range(0, 200):
			f.write(str(vary_axis[i]) + "\t" + str(vary_y_pred[i][0]) + "\n")
		f.close()



""" / / / / / END OF FILE / / / / / """