""" 

Not so random

A neural network is used to map a random sample (u, v) of the 2D uniform density over [-1, 1]x[-1, 1]
to a random sample (x,y) following an arbitrary probability density.

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

					 _____________
					|			  |
	input sample ->	|	Network	  | -> output sample
					|_____________|
	
	
	One sample consists of elements_per_sample elements, which themselves consist of 2 values.
	In a batch are samples_per_batch samples.
			
"""


# Dimensions
elements_per_sample = 500
samples_per_batch = 500


# Generator for training data
def uniform_gen():
	while True:
		yield (np.random.uniform(low=-1.0, high=1.0, size=(samples_per_batch, elements_per_sample, 2)), np.empty((samples_per_batch, elements_per_sample, 2)))


# Generator for testing data
# TODO: more test-samples than a batch
N_testsamples = samples_per_batch
def test_gen():
	while True:
		yield np.random.uniform(low=-1.0, high=1.0, size=(N_testsamples, elements_per_sample, 2))





""" / / / / / HISTOGRAM / / / / / """


# Get a single value of the 2D histogram of a sample y = [[y_11, y_12], [y_21, y_22], ...) at position Y = [y_1, y_2] with delta_y = y - Y
def get_hist_gauss2D(delta_y, step):
	return np.sum( np.exp( -0.5*np.sum(delta_y**2, axis=1)/((step*0.5)**2) ), 0)/ (2.0*np.pi*((step*0.5)**2)*float(delta_y.size)*0.5)





""" / / / / / DENSITY / / / / / """


# 2D Gaussian pdf with zero mean and sigma=1
def gaussian2D(y1, y2):
	return np.exp(-(y1**2 + y2**2)/2.0)/(2.0*np.pi)


# 2D bimodal gaussian pdf at (1.5, 1.5) and (-1.5, -1.5) with sigma=1
def bimodal2D(y1, y2):
	return (np.exp(-((y1-1.5)**2 + (y2-1.5)**2)/2.0) + np.exp(-((y1+1.5)**2 + (y2+1.5)**2)/2.0))/(2.0*np.pi*2.0)


# Probability density function to be used in the loss function
density = bimodal2D





""" / / / / / LOSS FUNCTION AND METRICS / / / / / """


"""

Explanation of the loss fct:



y_pred =	| y_(1,1) 					| y_(2,1) 				|					| y_(samples_per_batch,1)					|   first element of output samples
		 	| y_(1,2) 					| y_(2,2) 				|					|											|   second element of output samples
			|							|						|					|											|
			|  ...						|  ...					|		...			|	...										|    ...
			|							|						|					|											|
			| y_(1,elements_per_sample)	|  						|					| y_(samples_per_batch,elements_per_sample)	|   last element of output samples
				
					
			 first output sample  		  second output sample			...			  last output sample


Penalties: 
1) "density": histogram of each output sample is compared to density(y)
2) "density": histogram of each element of the output samples is compared to density(y)
3) "pot": Since the histograms/pdfs are only compared in [y_min, y_max], there has to be a penalty for points outside of this interval, a linear potential wall

"""


# Interval, where the pdfs are compared
y_min = -5.0
y_max = 5.0


# Steps for histograms
step = 0.5
# TODO: Rule for bin size in penalty?

# y axis for both histograms
y1_axis = np.linspace(y_min, y_max, int((y_max-y_min)/step)+1)
y2_axis = np.linspace(y_min, y_max, int((y_max-y_min)/step)+1)
y1_axis_size = y1_axis.size
y2_axis_size = y2_axis.size


# Weighting factors omitted for the 2D case


# Penalties

def penalty_density(y_true, y_pred):
	# Mean squared error
	error = K.variable(0.0)
	# Compare densites at each point in y1, y2 grid
	for y2 in y2_axis:
		for y1 in y1_axis:
			delta_y = y_pred - [y1, y2]
			tmp = K.exp(-0.5*K.sum((delta_y)**2, axis=2)/((step*0.5)**2))
			# Samples
			hist = K.sum(tmp, axis=1) / (2.0*np.pi*(step*0.5)*float(elements_per_sample))
			error = error + K.sum((hist - density(y1, y2))**2)/(float(samples_per_batch)*float(y1_axis_size))
			# Sample elements
			hist = K.sum(tmp, axis=0) / (2.0*np.pi*(step*0.5)*float(samples_per_batch))
			error = error + K.sum((hist - density(y1, y2))**2)/(float(elements_per_sample)*float(y2_axis_size))
	return error
	# y_true unused


def penalty_pot(y_true, y_pred):
	# "Potential wall" to keep the points in [y_min, y_max]
	return K.sum(K.sum(K.maximum(y_pred-y_max, 0.0) + K.maximum(-y_pred+y_min, 0.0) ))
	# y_true unused


def maximum(y_true, y_pred):
	# Maximum absolute output value, problem: is averaged over batch before displayed during training
	return K.max(K.max(K.abs(y_pred)))


# Loss function - mean squared deviation between the pdf/histogram of the output and the pdf given by density(y)
def loss(y_true, y_pred):
	# y_true unused
	return penalty_density(y_true, y_pred) + penalty_pot(y_true, y_pred)





""" / / / / / RUN_MODEL / / / / / """


def run_model(load, N_layers, N_neurons, opt, epochs, N_testsamples, x_train, x_test, save_results, path, plot):
	
	"""
		args:
		
		* load = (True, False), whether to load weights and model from argument path
		* N_layers = unsigned int > 0
		* N_neurons = unsigned int > 0, amount neurons in a hidden layer, in- and output layer have elements_per_sample*2 neurons
		* opt = keras optimizer object
		* epochs = unsigned int > 0
		* N_testsamples = unsigned int > 0, how many samples are to be expected in x_test
		* x_train = generator yielding array[samples_per_batch][elements_per_sample][2], training samples
		* x_test = generator yielding array[N_testsamples][elements_per_sample][2], testing samples
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
		#	model.compile(loss=loss, optimizer=opt, metrics=[penalty_density, penalty_pot, maximum])
		# model fit generator...
	else:
		model = Sequential()
		layers = [Reshape((2*elements_per_sample,),input_shape=(elements_per_sample, 2))]
		N = 1
		for i in range(1, N_layers):
			layers.append(Dense(2*elements_per_sample, activation="elu"))
		layers.append(Dense(2*elements_per_sample, activation=None))
		layers.append(Reshape((elements_per_sample, 2)))
		for layer in layers:
			model.add(layer)

		# Optimize
		model.compile(loss=loss, optimizer=opt, metrics=[penalty_density, penalty_pot, maximum])
		terminator = K_TerminateOnNaN()
		history = model.fit_generator(generator=x_train(), steps_per_epoch=1, epochs=epochs, workers=1, use_multiprocessing=False, callbacks=[terminator], verbose=1)


	#Predict
	y_pred = model.predict_generator(generator=x_test(), steps=1, max_queue_size=1, workers=1, use_multiprocessing=False, verbose=0)
	y_pred_t = np.transpose(y_pred, axes=(1,0,2))


	# Predicted probability density of each output sample separately
	p_pred_samples = np.empty((N_testsamples, y1_axis_size, y2_axis_size))
	for s in range(0, N_testsamples): # Over all samples
		for i in range(0, y1_axis_size): # Over all intervals of length step in [-y_min,y_max] along the y1 axis
			for j in range(0, y2_axis_size): # Over all intervals of length step in [-y_min,y_max] along the y2 axis			
				p_pred_samples[s][i][j] = get_hist_gauss2D(y_pred[s] - [y1_axis[i], y2_axis[j]], step)
	p_pred_samples_mean = np.mean(p_pred_samples, axis=0)
	p_pred_samples_std = np.std(p_pred_samples, axis=0)


	# Predicted probability density of each output sample element
	p_pred_elements = np.empty((elements_per_sample, y1_axis_size, y2_axis_size))
	for l in range(0, elements_per_sample): # Over all elements in a sample
		for i in range(0, y1_axis_size): # Over all intervals of length step in [-y_min,y_max] along the y1 axis
			for j in range(0, y2_axis_size): # Over all intervals of length step in [-y_min,y_max] along the y2 axis			
				p_pred_elements[l][i][j] = get_hist_gauss2D(y_pred_t[l] - [y1_axis[i], y2_axis[j]], step)
	p_pred_elements_mean = np.mean(p_pred_elements, axis=0) # is equal to p_pred_samples_mean! fuer egal welches parzen window! 
	p_pred_elements_std = np.std(p_pred_elements, axis=0) # is not equal to p_pred_samples_std!
	
	
	# Expected probability density
	p_star = np.empty((y1_axis_size, y2_axis_size))
	for i in range(0, y1_axis_size): # Over all intervals of length step in [-y_min,y_max] along the y1 axis
		for j in range(0, y2_axis_size): # Over all intervals of length step in [-y_min,y_max] along the y2 axis	
			p_star[i][j] = density(y1_axis[i], y2_axis[j])


	# Histogram of weights
	make_weights_hist = False # Set True to enable the calculation, saving and plot of the histogram of weights
	if ((make_weights_hist == True) and (N_neurons == 2*500)): # N_neurons == 2*500 is a check for calculation of the binsize... dirty coded
		weights = model.get_weights()
		# Amount bins calculated using the square-root rule, for 32 layers and N_neurons=500=vector_size	
		# For each neuron
		weights_histneuron = np.empty((N_layers, N_neurons, 30)) # [layer, neuron, bin]
		weights_histneuronaxis = np.empty((N_layers, N_neurons, 30+1)) # [layer, neuron, bin] one element more :(

		for i in range(0, N_layers): # Over all layers
			for j in range(0, N_neurons): # Over all neurons
				tmp_histogram = np.histogram(weights[2*i][j], 30)
				weights_histneuron[i][j] = tmp_histogram[0]
				weights_histneuronaxis[i][j] = tmp_histogram[1]
	
		# For each layer
		weights_histlayer = np.empty((N_layers, 2*500)) # [layer, bin]
		weights_histlayeraxis = np.empty((N_layers, 2*500+1)) # [layer, bin] one element more :(
		bias_histlayer = np.empty((N_layers, 30)) # [layer, bin]
		bias_histlayeraxis = np.empty((N_layers, 30+1)) # [layer, bin] one element more :(
		for i in range(0, N_layers): # Over all layers
			tmp_histogram = np.histogram(weights[2*i], 2*500)
			weights_histlayer[i] = tmp_histogram[0]
			weights_histlayeraxis[i] = tmp_histogram[1]
			tmp_histogram = np.histogram(weights[2*i+1], 30)	
			bias_histlayer[i] = tmp_histogram[0]
			bias_histlayeraxis[i] = tmp_histogram[1]
	
		# Overall histogram
		weights_histall = np.histogram(weights[::2], int(np.sqrt(N_layers*(2*500)**2)))
		bias_histall = np.histogram(weights[1::2], 2*500)


	# Save results
	if (save_results == True):
		if (load == True):
			raw_in = raw_input("Insert save path: ")
			if (raw_in != ""):
				path = raw_in

		# Make directory is it does not exist
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
				f.write("#penalty_density\tpenalty_pot\tmaximum\n")
				for i in range(0, len(history.history["penalty_density"])):
					f.write(str(history.history["penalty_density"][i]) + "\t" + 
							str(history.history["penalty_pot"][i]) + "\t" +
							str(history.history["maximum"][i]) + "\n")
				f.close()
		else:
			print "History not saved, only possible if load == False and epochs > 0"

		# Densities of vectors
		with open(path+"/density_samples", "w") as f:
			f.write("#y1\ty2\tmean\tstd\tdensity of sample i ...\n")
			for i in range(0, y1_axis_size): # Over all intervals of length step in [-y_min,y_max] along the y1 axis
				for j in range(0, y2_axis_size): # Over all intervals of length step in [-y_min,y_max] along the y2 axis
					# Mean and std. deviation
					f.write(str(y1_axis[i]) + "\t" + str(y2_axis[j]) + "\t" +
							str(p_pred_samples_mean[i][j]) + "\t" +
							str(p_pred_samples_std[i][j]))
					# Individual samples
					for s in range(0, N_testsamples): # Over all samples
						f.write("\t" + str(p_pred_samples[s][i][j]))
					f.write("\n")
				f.write("\n") # Newline for gnuplot's pm3d
			f.close()

		# Densities of elements
		with open(path+"/density_element", "w") as f:
			f.write("#y1\ty2\tmean\tstd\tdensity of element i ...\n")
			for i in range(0, y1_axis_size): # Over all intervals of length step in [-y_min,y_max] along the y1 axis
				for j in range(0, y2_axis_size): # Over all intervals of length step in [-y_min,y_max] along the y2 axis			
					f.write(str(y1_axis[i]) + "\t" + str(y2_axis[j]) + "\t" +
							str(p_pred_elements_mean[i][j]) + "\t" +
							str(p_pred_elements_std[i][j]))
					for l in range(0, elements_per_sample): # Over all elements in a sample
						f.write("\t" + str(p_pred_elements[l][i][j]))
					f.write("\n")
				f.write("\n") # Newline for gnuplot's pm3d
			f.close()


	# Weights histogram
	save_weights_hist = False
	if ((save_weights_hist == True) and (make_weights_hist == True)):
		# histlayer
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

		# histall
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
			plt.plot(history.history["penalty_density"], "g", history.history["penalty_pot"], "b") #history.history["loss"], "r",  , history.history["penalty3"], "k" 
		
		y1_grid, y2_grid = np.meshgrid(y1_axis, y2_axis)
		
		# Mean of densities of samples
		fig = plt.figure(1)
		ax = fig.add_subplot(111, projection="3d")
		ax.plot_wireframe(y1_grid, y2_grid, p_pred_samples_mean, color='b')
		#fig = plt.figure(2)
		#ax = fig.add_subplot(111, projection="3d")
		#ax.plot_wireframe(y1_grid, y2_grid, p_pred_samples_std, color='r')
				
		# Mean of densities of elements
		fig = plt.figure(3)
		ax = fig.add_subplot(111, projection="3d")
		ax.plot_wireframe(y1_grid, y2_grid, p_pred_elements_mean, color='b')
		#fig = plt.figure(4)
		#ax = fig.add_subplot(111, projection="3d")
		#ax.plot_wireframe(y1_grid, y2_grid, p_pred_elements_std, color='r')
				
		# Example of densities of vectors
		fig = plt.figure(5)
		ax = fig.add_subplot(111, projection="3d")
		ax.plot_wireframe(y1_grid, y2_grid, p_pred_samples[0])

		# Example of densities of vector elements
		fig = plt.figure(6)
		ax = fig.add_subplot(111, projection="3d")
		ax.plot_wireframe(y1_grid, y2_grid, p_pred_elements[0])

		plt.show()
	
	plot_weights_hist = False
	if ((plot_weights_hist == True) and (make_weights_hist == True)):
		# Histogram of weights for each neuron - example
		plt.figure(9)
		plt.plot(weights_histneuronaxis[0][0][1:], weights_histneuron[0][0], "r+")
		
		# Histogram of weights for each layer - example
		plt.figure(10)
		plt.plot(weights_histlayeraxis[0][1:], weights_histlayer[0], "r+")

		# Histogram of all weights
		plt.figure(11)
		plt.plot(weights_histall[1][1:], weights_histall[0], "r+")
		
		# Weights of a layer displayed as image
		plt.figure(12)
		#plt.imshow(weights[2*N_layers-2], cmap="gray", interpolation="none")
		plt.imshow(weights[0], cmap="gray", interpolation="none")
		
		# Histogram of biases for each layer - example
		plt.figure(13)
		plt.plot(bias_histlayeraxis[N_layers-1][1:], bias_histlayer[N_layers-1], "r+")

		# Histogram of all biases
		plt.figure(14)
		plt.plot(bias_histall[1][1:], bias_histall[0], "r+")
		
		# biases of the network displayed as image
		plt.figure(15)
		plt.imshow(weights[1::2], cmap="gray", interpolation="none")
		
		plt.show()


	# Clean up memory
	K.clear_session()
	tf_reset_default_graph()
	
	# Return prediction
	return y_pred	
#





""" / / / / / TESTS / / / / / """


"""path = "test2D/lr0.0001decay0.004epochs10000_2"
lr = 0.0001
decay = 0.004
opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
run_model(load=False, N_layers=32, N_neurons=2*elements_per_sample, opt=opt, epochs=10000, N_testsamples=N_testsamples, x_train=uniform_gen, x_test=test_gen, save_results=True, path=path, plot=False)
"""

path = "test2D/lr0.000001decay0.0epochs10000_2"
lr = 0.000001
decay = 0.0
opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
run_model(load=False, N_layers=32, N_neurons=2*elements_per_sample, opt=opt, epochs=10000, N_testsamples=N_testsamples, x_train=uniform_gen, x_test=test_gen, save_results=True, path=path, plot=False)


#run_model(load=True, N_layers=32, N_neurons=2*elements_per_sample, opt=opt, epochs=0, N_testsamples=N_testsamples, x_train=uniform_gen, x_test=test_gen, save_results=False, path=path, plot=True)

