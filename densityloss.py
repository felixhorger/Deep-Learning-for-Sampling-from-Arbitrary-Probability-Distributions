""" 

Not so random

A neural network is used to map a random sample of the uniform density over [-1, 1]
to a random sample of an arbitrary probability density.

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

					 _____________
					|			  |
	input sample ->	|	Network	  | -> output sample
					|_____________|
	
	
	One sample consists of elements_per_sample elements.
	In a batch are samples_per_batch samples.
			
"""


# Dimensions
elements_per_sample = 500
samples_per_batch = 500


# Generator for training data
def uniform_gen():
	while True:
		yield (np.random.uniform(low=-1.0, high=1.0, size=(samples_per_batch, elements_per_sample)), np.empty((samples_per_batch, elements_per_sample)))


# Generator for testing data
# TODO: more test-samples than a batch
N_testsamples = samples_per_batch
def test_gen():
	while True:
		yield np.random.uniform(low=-1.0, high=1.0, size=(N_testsamples, elements_per_sample))





""" / / / / / HISTOGRAM / / / / / """


# Get a single value of the histogram of a vector y at position Y, with delta_y[i] = y[i] - Y
def get_hist_gauss(delta_y, step): # BISHOP Page 123
	return np.sum( np.exp(-0.5*((delta_y) / (step*0.5) )**2), 0)/ (np.sqrt(2.0*np.pi)*(step*0.5)*float(delta_y.size))


# Get the complete histogram of vector y using parzen windowing, with a suitable rectangular window function
def get_hist_rect(y, step, y_axis):
	y_sort = np.sort(y)
	p = np.empty(y_axis.size)
	k = 0
	for i in range(0, y_axis.size): # Over all intervals of length step in [y_min,y_max]
		j = k
		while (k < N):
			if (np.abs(y_sort[k] - y_axis[i]) <= (step/2.0)): # y_i in interval i?
				k += 1
			elif ((y_axis[i] - y_sort[k]) > (step/2.0)): # Is the algorithm stuck? (due to numerical reason)
				k += 1
				j += 1
			else:
				break
		p[i] = (k - j)/(step*float(y.size))
	return p





""" / / / / / DENSITY / / / / / """


# Gaussian pdf with zero mean and sigma=1
def gaussian(y):
	return np.exp(-np.power(y, 2)/2.0)/np.sqrt(2.0*np.pi)


# Bimodal Gaussian pdf at y=-3 and y=3 with sigma=1
def bimodal(y):
	return (np.exp(-np.power(y - 3.0, 2)/2.0) + np.exp(-np.power(y + 3.0, 2)/2.0)) / (2.0*np.sqrt(2.0*np.pi))


# Asymmetric bimodal Gaussian pdf at y=1 and y=-3 with sigma=1
def asym_bimodal(y):
	return (np.exp(-np.power(y - 1.0, 2)/2.0) + np.exp(-np.power(y + 3.0, 2)/2.0)) / (2.0*np.sqrt(2.0*np.pi))


# Asymmetric bimodal with different variances (0.5, 1.0) gaussian pdf at y=1 and y=-3
def asymvar_bimodal(y):
	return np.exp(-np.power((y - 1.0)/0.5, 2)/2.0) / (2.0*0.5*np.sqrt(2.0*np.pi)) + np.exp(-np.power(y + 3.0, 2)/2.0) / (2.0*np.sqrt(2.0*np.pi))
# GNUPLOT d(y) = exp(-(((y - 1.0)/0.5)** 2)/2.0) / (2.0*0.5*sqrt(2.0*pi)) + exp(-((y + 3.0)** 2)/2.0) / (2.0*sqrt(2.0*pi))


# Constant pdf in [-1, 1]
def constant(y):
	return np.where(np.abs(y) <= 1.0, 0.5, 0.0)


# Triangular pdf from -1.0 to 1.0
def triangular(y):
	return np.maximum(0.0, -np.abs(y)+1.0)


# Probability density function to be used in the loss function
density = bimodal






""" / / / / / LOSS FUNCTION AND METRICS / / / / / """


"""

Explanation of the loss fct:



y_pred =	| y_(1,1) 					| y_(2,1) 				|					| y_(samples_per_batch,1)					|   first element of output samples
		 	| y_(1,2) 					| y_(2,2) 				|					|											|   second element of output samples
			|							|						|					|											|
			|  ...						|  ...					|		...			|	...										|    ...
			|							|						|					|											|
			| y_(1,elements_per_sample)	|  						|					| y_(samples_per_batch,elements_per_sample)	|   last element of output samples
				
					
			 first output sample  			second output sample		...			  last output sample


Penalties:
"samples":	histogram of each output sample is compared to density(y)
"elements":	histogram of each element of the output samples is compared to density(y)
"pot":		Since the histograms/pdfs are only compared in [y_min, y_max], there has to be a penalty for points outside of this interval, a linear potential wall

"""


# Interval, where the pdfs are compared
y_min = -10.0
y_max = 10.0


# Steps for histograms
step_samples = 0.1
step_elements = 0.1
# TODO: Rule for bin size in penalty?


# y axis for both histograms
y_axis_samples = np.linspace(y_min, y_max, int((y_max-y_min)/step_samples)+1)
y_axis_elements = np.linspace(y_min, y_max, int((y_max-y_min)/step_elements)+1)
y_axis_samples_size = y_axis_samples.size
y_axis_elements_size = y_axis_elements.size


# Weighting factor w for penalty_samples, penalty_elements
w_samples = 1.0
w_elements = 1.0


#Penalties

def penalty_samples(y_true, y_pred):
	# Mean squared error
	error = K.variable(0.0)
	# Compare histogram of each output vector separately to density(y)
	for y in y_axis_samples:
		delta_y = y_pred - y
		hist_samples = K.sum( K.exp(-0.5*((delta_y) / (step_samples*0.5) )**2), 1)/ (np.sqrt(2.0*np.pi)*(step_samples*0.5)*float(elements_per_sample))
		#ce = ce - K.sum(hist_samples*np.log(density(y)))/float(samples_per_batch)
		error = error + K.sum(K.pow(hist_samples - density(y), 2))/(float(samples_per_batch)*float(y_axis_samples_size))
	return w_samples*error
	# y_true unused


def penalty_elements(y_true, y_pred):
	# Mean squared deviation
	error = K.variable(0.0)
	# Compare histogram of each output vector element separately to density(y)
	for y in y_axis_elements:
		delta_y = y_pred - y
		hist_elements = K.sum( K.exp(-0.5*((delta_y) / (step_elements*0.5) )**2), 0)/ (np.sqrt(2.0*np.pi)*(step_elements*0.5)*float(samples_per_batch))
		error = error + K.sum(K.pow(hist_elements - density(y), 2))/(float(elements_per_sample)*float(y_axis_elements_size)) 
	return w_elements*error
	# y_true unused


def penalty_pot(y_true, y_pred):
	# "Potential wall" to keep the points in [y_min, y_max]
	return 1.0*K.sum(K.sum(K.maximum(y_pred-y_max, 0.0) + K.maximum(-y_pred+y_min, 0.0) ))
	# y_true unused


def maximum(y_true, y_pred):
	# Maximum absolute output value, problem: is averaged over batch
	return K.max(K.max(K.abs(y_pred)))
	


# Loss function - mean squared error between the histogram of the output samples / sample elements and the pdf given by density(y)
def loss(y_true, y_pred):
	return penalty_samples(y_true, y_pred) + penalty_elements(y_true, y_pred) + penalty_pot(y_true, y_pred)





""" / / / / / RUN_MODEL / / / / / """


def run_model(load, N_layers, N_neurons, opt, epochs, N_testsamples, x_train, x_test, save_results, path, plot):
	
	"""
		args:
		
		* load = (True, False), whether to load weights and model from argument path
		* N_layers = unsigned int > 0
		* N_neurons = unsigned int > 0, amount neurons in a hidden layer, in- and output layer have elements_per_sample neurons
		* opt = keras optimizer object
		* epochs = unsigned int > 0
		* N_testsamples = unsigned int > 0, how many samples are to be expected in x_test
		* x_train = generator yielding array[samples_per_batch][elements_per_sample], training samples
		* x_test = generator yielding array[N_testsamples][elements_per_sample], testing samples
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
		# Needed if trainging will go on:
		#if (epochs > 0):
		#	model.compile(loss=loss, optimizer=opt, metrics=[penalty_samples, penalty_elements, penalty_pot, maximum])
		# model fit generator...
	else:
		model = Sequential()
		layers = [Dense(N_neurons, input_shape=(elements_per_sample,), activation="elu")]
		for i in range(1, N_layers-1):
			layers.append(Dense(N_neurons, activation="elu"))
		layers.append(Dense(elements_per_sample, activation=None))
		for layer in layers:
			model.add(layer)
		# Optimize
		model.compile(loss=loss, optimizer=opt, metrics=[penalty_samples, penalty_elements, penalty_pot, maximum])
		terminator = K_TerminateOnNaN()
		history = model.fit_generator(generator=x_train(), steps_per_epoch=1, epochs=epochs, workers=1, use_multiprocessing=False, callbacks=[terminator], verbose=1)


	#Predict
	y_pred = model.predict_generator(generator=x_test(), steps=1, max_queue_size=1, workers=1, use_multiprocessing=False, verbose=0)
	y_pred_t = np.transpose(y_pred)


	# Predicted probability density of each output sample separately
	p_pred_samples = np.empty((N_testsamples, y_axis_samples_size))
	for s in range(0, N_testsamples): # Over all samples
		for i in range(0, y_axis_samples_size): # Over all intervals of length step_samples in [-y_min,y_max]
			p_pred_samples[s][i] = get_hist_gauss(y_pred[s] - y_axis_samples[i], step_samples)
	p_pred_samples_mean = np.mean(p_pred_samples, axis=0)
	p_pred_samples_std = np.std(p_pred_samples, axis=0)


	# Predicted probability density of each output sample element
	p_pred_elements = np.empty((elements_per_sample, y_axis_elements_size))
	for l in range(0, elements_per_sample): # Over all elements in a sample
		for i in range(0, y_axis_elements_size): # Over all intervals of length step_elements in [-y_min,y_max]
			p_pred_elements[l][i] = get_hist_gauss(y_pred_t[l] - y_axis_elements[i], step_elements)
	p_pred_elements_mean = np.mean(p_pred_elements, axis=0) # is equal to p_pred_samples_mean!
	p_pred_elements_std = np.std(p_pred_elements, axis=0) # is not equal to p_pred_samples_std!


	# Expected probability density
	p_star_samples = density(y_axis_samples)
	p_star_elements = density(y_axis_elements)


	# Histogram of weights
	make_weights_hist = False # Set True to enable the calculation, saving and plot of the histogram of weights
	if ((make_weights_hist == True) and (N_neurons == elements_per_sample)): # N_neurons == elements_per_sample is a check for calculation of the binsize... dirty coded
		weights = model.get_weights()
		# Amount bins calculated using the square-root rule, for 32 layers and N_neurons=500=elements_per_sample	
		# For each neuron
		weights_histneuron = np.empty((N_layers, N_neurons, 25)) # [layer, neuron, bin]
		weights_histneuronaxis = np.empty((N_layers, N_neurons, 25+1)) # [layer, neuron, bin] one element more :(

		for i in range(0, N_layers): # Over all layers
			for j in range(0, N_neurons): # Over all neurons
				tmp_histogram = np.histogram(weights[2*i][j], 25)
				weights_histneuron[i][j] = tmp_histogram[0]
				weights_histneuronaxis[i][j] = tmp_histogram[1]
	
		# For each layer
		weights_histlayer = np.empty((N_layers, 500)) # [layer, bin]
		weights_histlayeraxis = np.empty((N_layers, 500+1)) # [layer, bin] one element more :(
		bias_histlayer = np.empty((N_layers, 25)) # [layer, bin]
		bias_histlayeraxis = np.empty((N_layers, 25+1)) # [layer, bin] one element more :(
		for i in range(0, N_layers): # Over all layers
			tmp_histogram = np.histogram(weights[2*i], 500) #TODO
			weights_histlayer[i] = tmp_histogram[0]
			weights_histlayeraxis[i] = tmp_histogram[1]
			tmp_histogram = np.histogram(weights[2*i+1], 25)	
			bias_histlayer[i] = tmp_histogram[0]
			bias_histlayeraxis[i] = tmp_histogram[1]
	
		# Overall histogram
		weights_histall = np.histogram(weights[::2], 2830)
		bias_histall = np.histogram(weights[1::2], 500)
		

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

		# Losses
		if ((load == False) and (epochs > 0)):
			with open(path+"/penalty_epoch", "w") as f:
				f.write("#penalty_samples\tpenalty_elements\tpenalty_pot\tmaximum\n")
				for i in range(0, len(history.history["penalty_samples"])):
					f.write(str(history.history["penalty_samples"][i]) + "\t" + 
							str(history.history["penalty_elements"][i]) + "\t" + 
							str(history.history["penalty_pot"][i]) + "\t" + 
							str(history.history["maximum"][i]) + "\n")
				f.close()
		else:
			print "History not saved, only possible if load == False and epochs > 0"

		# Densities of output samples
		with open(path+"/density_samples", "w") as f:
			f.write("#y\tmean\tstd\tdensity of sample i ...\n")
			for i in range(0, y_axis_samples_size): # Over elements in y_axis_samples
				f.write(str(y_axis_samples[i]) + "\t" +
						str(p_pred_samples_mean[i]) + "\t" +
						str(p_pred_samples_std[i]))
				for s in range(0, N_testsamples): # Over all test samples
					f.write("\t" + str(p_pred_samples[s][i]))
				f.write("\n")
			f.close()

		# Densities of output samples elements
		with open(path+"/density_elements", "w") as f:
			f.write("#y\tmean\tstd\tdensity of element i ...\n")
			for i in range(0, y_axis_elements_size): # Over elements in y_axis_elements
				f.write(str(y_axis_elements[i]) + "\t" +
						str(p_pred_elements_mean[i]) + "\t" +
						str(p_pred_elements_std[i]))
				for l in range(0, elements_per_sample): # Over all sample elements
					f.write("\t" + str(p_pred_elements[l][i]))
				f.write("\n")
			f.close()


	# Weights histogram
	save_weights_hist = False
	if (save_weights_hist == True):
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
			plt.plot(history.history["penalty_samples"], "g", history.history["penalty_elements"], "b")

		# Densities of output samples
		plt.figure(1)
		plt.plot(y_axis_samples, p_star_samples, "bo", np.repeat([y_axis_samples], N_testsamples, axis=0), p_pred_samples, "r+")
		plt.plot(y_axis_samples, p_star_samples, "bo", y_axis_samples, p_pred_samples_mean, "r", y_axis_samples, p_pred_samples_mean - p_pred_samples_std, "g", y_axis_samples, p_pred_samples_mean + p_pred_samples_std, "g")

		# Densities of output sample elements
		plt.figure(2)
		plt.plot(y_axis_elements, p_star_elements, "bo", np.repeat([y_axis_elements], elements_per_sample, axis=0), p_pred_elements, "r+")
		plt.plot(y_axis_elements, p_star_elements, "bo", y_axis_elements, p_pred_elements_mean, "r", y_axis_elements, p_pred_elements_mean - p_pred_elements_std, "g", y_axis_elements, p_pred_elements_mean + p_pred_elements_std, "g")

		# Two examples of densities of output samples
		plt.figure(3)
		plt.plot(y_axis_samples, p_star_samples, "bo", y_axis_samples, p_pred_samples[0], "r", y_axis_samples, p_pred_samples[2], "g")

		# Two examples of densities of output sample elements
		plt.figure(4)
		plt.plot(y_axis_elements, p_star_elements, "bo", y_axis_elements, p_pred_elements[0], "r", y_axis_elements, p_pred_elements[2], "g")

		# Two examples of output samples (to see if randomness is guaranteed)
		plt.figure(5)
		plt.plot(y_pred[0], "bo", y_pred[2], "r+")

		# Two examples of output sample elements
		plt.figure(6)
		plt.plot(np.repeat([np.arange(0,500)], 2, axis=0), y_pred_t[0:2], "bo")
		
		plt.show()
	
	plot_weights_hist = False
	if (plot_weights_hist == True):
		# Histogram of weights for each neuron - example
		plt.figure(7)
		plt.plot(weights_histneuronaxis[0][0][1:], weights_histneuron[0][0], "r+")
		
		# Histogram of weights for each layer - example
		plt.figure(8)
		plt.plot(weights_histlayeraxis[0][1:], weights_histlayer[0], "r+")

		# Histogram of all weights
		plt.figure(9)
		plt.plot(weights_histall[1][1:], weights_histall[0], "r+")
		
		# Weights of a layer displayed as image
		plt.figure(10)
		#plt.imshow(weights[2*N_layers-2], cmap="gray", interpolation="none")
		plt.imshow(weights[0], cmap="gray", interpolation="none")
		
		# Histogram of biases for each layer - example
		plt.figure(11)
		plt.plot(bias_histlayeraxis[N_layers-1][1:], bias_histlayer[N_layers-1], "r+")

		# Histogram of all biases
		plt.figure(12)
		plt.plot(bias_histall[1][1:], bias_histall[0], "r+")
		
		# biases of the network displayed as image
		plt.figure(13)
		plt.imshow(weights[1::2], cmap="gray", interpolation="none")
		
		plt.show()


	# Clean up memory
	K.clear_session()
	tf_reset_default_graph()
	
	# Return prediction
	return y_pred	
#





""" / / / / / TESTS / / / / / """


"""--- Optimizer ---"""

# Testing SGD
"""lr_list = np.linspace(0.1, 0.5, (0.5-0.1)/0.1+1)
decay_list = np.linspace(0.0, 0.05, (0.05)/0.01+1)
momentum_list = np.linspace(0.1, 0.6, (0.6-0.1)/0.1+1)
for lr in lr_list:
	for decay in decay_list:
		for momentum in momentum_list:
			#borders needed?    if ((lr >= 0.2) and ((not (lr == 0.2)) or (decay >= 0.01)) and ((not ((lr == 0.2) and (decay == 0.01))) or (momentum > 0.1))):
			opt = SGD(lr=lr, momentum=momentum, decay=decay, nesterov=False)
			path = "Optimizers/SGD/lr"+str(lr)+"_decay"+str(decay)+"_momentum"+str(momentum)
			run_model(load=False, N_layers=32, N_neurons=elements_per_sample, opt=opt, epochs=1000, N_testsamples=N_testsamples, x_train=uniform_gen, x_test=test_gen, save_results=True, path=path, plot=False)
			print "Run with SGD for lr"+str(lr)+" decay"+str(decay)+" momentum"+str(momentum)+" complete."
"""
"""# Testing SGD2 (second SGD test with different param. range)
lr_list = np.linspace(0.3, 0.5, (0.5-0.3)/0.05+1)
decay_list = np.array([0.0])
momentum_list = np.linspace(0.5, 0.8, (0.8-0.5)/0.05+1)
for lr in lr_list:
	for decay in decay_list:
		for momentum in momentum_list:
			opt = SGD(lr=lr, momentum=momentum, decay=decay, nesterov=False)
			path = "Optimizers/SGD2/lr"+str(lr)+"_decay"+str(decay)+"_momentum"+str(momentum)
			run_model(load=False, N_layers=32, N_neurons=elements_per_sample, opt=opt, epochs=1000, N_testsamples=N_testsamples, x_train=uniform_gen, x_test=test_gen, save_results=True, path=path, plot=False)
			print "Run with SGD2 for lr"+str(lr)+" decay"+str(decay)+" momentum"+str(momentum)+" complete."
"""
"""# Testing Adam
lr_list = np.array([0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]) #np.linspace(0.0001, 0.0, (0.5-0.1)/0.1+1)
decay_list = np.linspace(0.0, 0.01, (0.01)/0.001+1)
for lr in lr_list:
	for decay in decay_list:
		opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
		path = "Optimizers/Adam/lr"+str(lr)+"_decay"+str(decay)
		run_model(load=False, N_layers=32, N_neurons=elements_per_sample, opt=opt, epochs=1000, N_testsamples=N_testsamples, x_train=uniform_gen, x_test=test_gen, save_results=True, path=path, plot=False)
		print "Run with Adam for lr"+str(lr)+" decay"+str(decay)+" complete."
"""
"""# Testing Adam 2
lr_list = np.linspace(0.00006, 0.00014, 5)
decay_list = np.linspace(0.003, 0.007, 5)
for lr in lr_list:
	for decay in decay_list:
		opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
		path = "Optimizers/Adam2/lr"+str(lr)+"_decay"+str(decay)
		run_model(load=False, N_layers=32, N_neurons=elements_per_sample, opt=opt, epochs=1000, N_testsamples=N_testsamples, x_train=uniform_gen, x_test=test_gen, save_results=True, path=path, plot=False)
		print "Run with Adam2 for lr"+str(lr)+" decay"+str(decay)+" complete."
"""

"""
# Testing lr-curve 10000 epochs
lr = 0.0001
decay = 0.004
path = "Optimizers/lrcurve"
opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
run_model(load=False, N_layers=32, N_neurons=elements_per_sample, opt=opt, epochs=30000, N_testsamples=N_testsamples, x_train=uniform_gen, x_test=test_gen, save_results=True, path=path, plot=False)
print "Run with 30000 epochs complete."
"""




"""--- Network dimensions ---"""

# Testing N_layers
"""
N_layers_list = np.arange(1, 41)
opt_param_list = [{"lr": 0.0001, "decay": 0.004},  {"lr": 0.001, "decay": 0.04}]
for N_layers in N_layers_list:
	for opt_param in opt_param_list:
		path = "Networkdimension/layers/lr"+str(opt_param["lr"])+"decay"+str(opt_param["decay"])+"/N_layers"+str(N_layers)
		opt = Adam(lr=opt_param["lr"], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=opt_param["decay"])
		run_model(load=False, N_layers=N_layers, N_neurons=elements_per_sample, opt=opt, epochs=1000, N_testsamples=N_testsamples, x_train=uniform_gen, x_test=test_gen, save_results=True, path=path, plot=False)
		print "Run with N_layers = "+str(N_layers) + " complete." """


# Testing N_neurons
"""

N_neurons_list = np.arange(100, 1050, 50)
opt_param_list = [{"lr": 0.00001, "decay": 0.0004}, {"lr": 0.0001, "decay": 0.004},  {"lr": 0.001, "decay": 0.04}]
for N_neurons in N_neurons_list:
	for opt_param in opt_param_list:
		path = "Networkdimension/neurons/lr"+str(opt_param["lr"])+"decay"+str(opt_param["decay"])+"/N_neurons"+str(N_neurons)
		opt = Adam(lr=opt_param["lr"], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=opt_param["decay"])
		run_model(load=False, N_layers=32, N_neurons=N_neurons, opt=opt, epochs=1000, N_testsamples=N_testsamples, x_train=uniform_gen, x_test=test_gen, save_results=True, path=path, plot=False)
		print "Run with N_neurons = "+str(N_neurons) + " complete."
"""





"""--- Input density ---"""

# Three input density are tested, uniform, gaussian and exponential for a triangular aim density
"""
N_density_test = 5 # For mean calculation later on
aim_density = "triangular" #"asymvarbimodal"

# Test uniform [-1,1]
lr = 0.0001
decay = 0.004
for i in range(0, N_density_test):
	path = "Input/"+aim_density+"/uniform"+str(i)
	opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
	run_model(load=False, N_layers=32, N_neurons=elements_per_sample, opt=opt, epochs=1000, N_testsamples=N_testsamples, x_train=uniform_gen, x_test=test_gen, save_results=True, path=path, plot=False)
	print "Run with uniform input desity complete."
# Why not e.g. uniform [0,1]? This could be transformed to uniform [-1,1] by right choice of biases and input weights


# Test gaussian
def gaussian_gen():
	while True:
		yield (np.random.normal(loc=0.0, scale=1.0, size=(samples_per_batch,elements_per_sample)), np.empty((samples_per_batch,elements_per_sample)))
def gaussian_test_gen():
	while True:
		yield np.random.normal(loc=0.0, scale=1.0, size=(N_testsamples,elements_per_sample))

lr = 0.0001
decay = 0.004
for i in range(0, N_density_test):
	path = "Input/"+aim_density+"/gaussian"+str(i)
	opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
	run_model(load=False, N_layers=32, N_neurons=elements_per_sample, opt=opt, epochs=1000, N_testsamples=N_testsamples, x_train=gaussian_gen, x_test=gaussian_test_gen, save_results=True, path=path, plot=False)
	print "Run with gaussian input density complete."


# Test exponential
def exp_gen():
	while True:
		yield (np.random.exponential(scale=1.0, size=(samples_per_batch,elements_per_sample)), np.empty((samples_per_batch,elements_per_sample)))
def exp_test_gen():
	while True:
		yield np.random.exponential(scale=1.0, size=(N_testsamples,elements_per_sample))

lr = 0.0001
decay = 0.004
for i in range(0, N_density_test):
	path = "Input/"+aim_density+"/exponential"+str(i)
	opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
	run_model(load=False, N_layers=32, N_neurons=elements_per_sample, opt=opt, epochs=1000, N_testsamples=N_testsamples, x_train=exp_gen, x_test=exp_test_gen, save_results=True, path=path, plot=False)
	print "Run with exponential input density complete."

"""





"""--- Weights ---"""


# Weights of a trained network with 32 layers and 500 neurons each
"""lr = 0.0001
decay = 0.004
opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
path = "Weights/trained"
run_model(load=True, N_layers=32, N_neurons=elements_per_sample, opt=opt, epochs=0, N_testsamples=N_testsamples, x_train=uniform_gen, x_test=test_gen, save_results=True, path=path, plot=False)
"""

# Weights of an untrained network of equal dimension as above
"""
lr = 0.0001
decay = 0.004
opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
path = "Weights/glorotuniform"
run_model(load=False, N_layers=32, N_neurons=elements_per_sample, opt=opt, epochs=0, N_testsamples=N_testsamples, x_train=uniform_gen, x_test=test_gen, save_results=True, path=path, plot=False)
"""

# Weights of an trained network with 3 layers and 500 neurons each
"""
lr = 0.003
decay = 0.01
opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
path = "Weights/3layers"
run_model(load=False, N_layers=10, N_neurons=elements_per_sample, opt=opt, epochs=1000, N_testsamples=N_testsamples, x_train=uniform_gen, x_test=test_gen, save_results=True, path=path, plot=False)
"""





"""--- Dirac-delta ---"""


"""def dirac_test_gen():
	while True:
		test_samples = np.zeros((N_testsamples+1, elements_per_sample))
		test_samples[0:N_testsamples, 0:N_testsamples] = np.eye(N_testsamples, elements_per_sample)
		test_samples[N_testsamples, 0] = 1
		test_samples[N_testsamples, 1] = 1
		yield test_samples
lr = 0.0001
decay = 0.004
path = "Diracdelta"
opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
dirac_y_pred = run_model(load=False, N_layers=32, N_neurons=elements_per_sample, opt=opt, epochs=1000, N_testsamples=N_testsamples+1, x_train=uniform_gen, x_test=dirac_test_gen, save_results=False, path=path, plot=False)
with open(path+"/prediction", "w") as f:
	f.write("#element_i ...\n")
	for i in range(0, N_testsamples+1): # over all predicted samples
		for j in range(0, elements_per_sample): # over all vector elements
			f.write(str(dirac_y_pred[i][j]) + "\t")
		f.write("\n")
	f.close()
"""




"""--- Varying a fixed amount of elements only ---"""


"""N_vary = 10
def vary_test_gen():
	while True:
		test_samples = np.zeros((N_vary*200, elements_per_sample)) # 1 to 10 varying elements and 200 elements in [-1,1]
		for l in range(0, N_vary):
			test_samples.T[l] = np.concatenate((np.zeros(200*l), np.tile(np.linspace(-1.0, 1.0, 200), N_vary-l)), axis=0)
		yield test_samples

lr = 0.0001
decay = 0.004
opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
path = "Varying"
#vary_y_pred = run_model(load=False, N_layers=32, N_neurons=elements_per_sample, opt=opt, epochs=1000, N_testsamples=N_vary*200, x_train=uniform_gen, x_test=vary_test_gen, save_results=True, path=path, plot=False)
vary_y_pred = run_model(load=True, N_layers=32, N_neurons=elements_per_sample, opt=opt, epochs=0, N_testsamples=N_vary*200, x_train=uniform_gen, x_test=vary_test_gen, save_results=False, path=path, plot=False)
for l in range(0, N_vary):
	with open(path+"/N_vary"+str(l+1), "w") as f:
		f.write("#element_i ...\n")
		for i in range(0, 200): # Over all predicted samples for a fixed amount of varying vector elements
			for j in range(0, elements_per_sample): # Over all vector entries
				f.write(str(vary_y_pred[l*200+i][j]) + "\t")
			f.write("\n")
		f.close()
"""

"""
# Calculate the first output element versus the first two input elements for a 2D plot
def vary2D_test_gen():
	while True:
		test_samples = np.zeros(((100+1)*(100+1), elements_per_sample)) # grid for first and second input
		test_samples.T[0] = np.tile(np.linspace(-1.0, 1.0, 100+1), 100+1)
		for i in range(0, 100+1):
			test_samples.T[1][i*(100+1):(i+1)*(100+1)] = (2.0/100.0*i-1.0)*np.ones(100+1)
		yield test_samples

lr = 0.0001
decay = 0.004
opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
path = "Varying"
vary_y_pred = run_model(load=True, N_layers=32, N_neurons=elements_per_sample, opt=opt, epochs=0, N_testsamples=(100+1)*(100+1), x_train=uniform_gen, x_test=vary2D_test_gen, save_results=False, path=path, plot=False)
with open(path+"/vary2D", "w") as f:
	f.write("#input1\tinput2\telement_0\n")
	for x_1 in range(0, 100+1): # second input sample element
		for x_0 in range(0, 100+1): # first input sample element
			f.write(str(2.0/100.0*x_0-1.0) + "\t" + str(2.0/100.0*x_1-1.0) + "\t" + str(vary_y_pred[x_0 + (100+1)*x_1][10]) + "\n")
		f.write("\n")	
	f.close()
"""




"""--- Mode test ---"""

"""lr_list = [0.0001, 0.00001, 0.000005]
decay_list = [0.004, 0.0, 0.0]

for i in range(0, len(lr_list)):
	lr = lr_list[i]
	decay = decay_list[i]
	path = "Modetest/lr"+str(lr)+"decay"+str(decay)+"epochs30000"
	opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
	run_model(load=False, N_layers=32, N_neurons=elements_per_sample, opt=opt, epochs=30000, N_testsamples=N_testsamples, x_train=uniform_gen, x_test=test_gen, save_results=True, path=path, plot=False)
"""




"""--- Step size test ---"""


"""lr = 0.0001
decay = 0.004
step_list = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

for step in step_list:

	step_samples = step
	step_elements = step

	y_axis_samples = np.linspace(y_min, y_max, int((y_max-y_min)/step_samples)+1)
	y_axis_elements = np.linspace(y_min, y_max, int((y_max-y_min)/step_elements)+1)
	y_axis_samples_size = y_axis_samples.size
	y_axis_elements_size = y_axis_elements.size

	path = "Stepsize/"+str(step)
	opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
	run_model(load=False, N_layers=32, N_neurons=elements_per_sample, opt=opt, epochs=1000, N_testsamples=N_testsamples, x_train=uniform_gen, x_test=test_gen, save_results=True, path=path, plot=False)
"""


