from __future__ import division
import numpy as np
import scipy.stats
from scipy.special import erfc, gammaln
import pickle
from copy import deepcopy

class Event(object):
	def __init__(self, index, timestamp, type):
		super(Event, self).__init__()
		self.index = index
		self.timestamp = timestamp
		self.type = type
		
class Mode(object):
	def __init__(self, index):
		super(Mode, self).__init__()
		self.index = index
		self.alpha = None
		self.type_distribution = None

	def add_event(self, event):
		if self.type_distribution is None:
			self.type_distribution = np.copy(event.type)
		else:
			self.type_distribution += event.type

class Particle(object):
	"""docstring for Particle"""
	def __init__(self, weight):
		super(Particle, self).__init__()
		self.weight = weight
		self.log_update_prob = 0
		self.modes = {} 
		self.events2Mode_ID = [] 
		self.isInsider = []
		self.active_modes = {} # dict key = mode_index, value = list of timestamps in specific mode (queue)
		self.mode_num_by_now = 0

	def __repr__(self):
		return 'particle document list to mode IDs: ' + str(self.events2Mode_ID) + '\n' + 'weight: ' + str(self.weight)
		

def dirichlet(prior):
	''' Draw 1-D samples from a dirichlet distribution to multinomial distritbution. Return a multinomial probability distribution.
		@param:
			1.prior: Parameter of the distribution (k dimension for sample of dimension k).
		@rtype: 1-D numpy array
	'''
	return np.random.dirichlet(prior).squeeze()

def multinomial(exp_num, probabilities):
	''' Draw samples from a multinomial distribution.
		@param:
			1. exp_num: Number of experiments.
			2. probabilities: multinomial probability distribution (sequence of floats).
		@rtype: 1-D numpy array
	'''
	return np.random.multinomial(exp_num, probabilities).squeeze()
	
def categorical(probabilities,exp_num=1):
	''' Draw samples from a categorical distribution.
		@param:
			1. exp_num: Number of experiments.
			2. probabilities: categorical probability distribution (sequence of floats).
		@rtype: 1-D numpy array
	'''
	return np.random.multinomial(exp_num, probabilities).squeeze()

def EfficientImplementation(tn, reference_time, bandwidth, epsilon = 1e-5):
	''' return the time we need to compute to update the triggering kernel
		@param:
			1.tn: float, current document time
			2.reference_time: list, reference_time for triggering_kernel
			3.bandwidth: int, bandwidth for triggering_kernel
			4.epsilon: float, error tolerance
		@rtype: float
	'''
	max_ref_time = max(reference_time)
	max_bandwidth = max(bandwidth)
	tu = tn - ( max_ref_time + np.sqrt( -2 * max_bandwidth * np.log(0.5 * epsilon * np.sqrt(2 * np.pi * max_bandwidth**2)) ))
	return tu

def log_Dirichlet_CDF(outcomes, prior):
	''' the function only applies to the symmetry case when all prior equals to 1.
		@param:
			1.outcomes: output variables vector
			2.prior: must be list of 1's in our case, avoiding the integrals.
		@rtype: 
	'''
	return np.sum(np.log(outcomes)) + scipy.stats.dirichlet.logpdf(outcomes, prior)

def RBF_kernel(reference_time, time_interval, bandwidth):
	''' RBF kernel for Hawkes process.
		@param:
			1.reference_time: np.array, entries larger than 0.
			2.time_interval: float/np.array, entry must be the same.
			3. bandwidth: np.array, entries larger than 0.
		@rtype: np.array
	'''
	numerator = - (time_interval - reference_time) ** 2 / (2 * bandwidth ** 2) 
	denominator = (2 * np.pi * bandwidth ** 2 ) ** 0.5
	return np.exp(numerator) / denominator

def triggering_kernel(alpha, reference_time, time_intervals, bandwidth):
	''' triggering kernel for Hawkes porcess.
		@param:
			1. alpha: np.array, entres larger than 0
			2. reference_time: np.array, entries larger than 0.
			3. time_intervals: float/np.array, entry must be the same.
			4. bandwidth: np.array, entries larger than 0.
		@rtype: np.array
	'''
	time_intervals = time_intervals.reshape(-1, 1)
	if len(alpha.shape) == 3:
		return np.sum(np.sum(alpha * RBF_kernel(reference_time, time_intervals, bandwidth), axis = 1), axis = 1)
	else:
		return np.sum(np.sum(alpha * RBF_kernel(reference_time, time_intervals, bandwidth), axis = 0), axis = 0)

def g_theta(timeseq, reference_time, bandwidth, max_time):
	''' g_theta for DHP
		@param:
			2. timeseq: 1-D np array time sequence before current time
			3. base_intensity: float
			4. reference_time: 1-D np.array
			5. bandwidth: 1-D np.array
		@rtype: np.array, shape(3,)
	'''
	timeseq = timeseq.reshape(-1, 1)
	results = 0.5 * ( erfc(- reference_time / (2 * bandwidth ** 2) ** 0.5) - erfc( (max_time - timeseq - reference_time) / (2 * bandwidth ** 2) **0.5) )
	return np.sum(results, axis = 0)

def update_triggering_kernel(timeseq, alphas, reference_time, bandwidth, base_intensity, max_time, log_priors):
	''' procedure of triggering kernel for SMC
		@param:
			1. timeseq: list, time sequence including current time
			2. alphas: 2-D np.array with shape (sample number, length of alpha)
			3. reference_time: np.array
			4. bandwidth: np.array
			5. log_priors: 1-D np.array with shape (sample number,), p(alpha, alpha_0)
			6. base_intensity: float
			7. max_time: float
		@rtype: 1-D numpy array with shape (length of alpha0,)
	'''
	logLikelihood = log_likelihood(timeseq, alphas, reference_time, bandwidth, base_intensity, max_time)
	log_update_weight = log_priors + logLikelihood
	log_update_weight = log_update_weight - np.max(log_update_weight) # avoid overflow
	update_weight = np.exp(log_update_weight); update_weight = update_weight / np.sum(update_weight)
	update_weight = update_weight.reshape(-1,1)
	alpha = np.sum(update_weight * alphas, axis = 0)
	return alpha

def log_likelihood(timeseq, alphas, reference_time, bandwidth, base_intensity, max_time):
	''' compute log_likelihood for a time sequence for a cluster for SMC
		@param:
			1. timeseq: list, time sequence including current time
			2. alphas: 2-D np.array with shape (sample number, length of alpha)
			3. reference_time: np.array
			4. bandwidth: np.array
			5. log_priors: 1-D np.array, p(alpha, alpha_0)
			6. base_intensity: float
			7. max_time: float
		@rtype: 1-D numpy array with shape (sample number,)
	'''
	Lambda_0 = base_intensity * max_time
	alphas_times_gtheta = np.sum(alphas * g_theta(timeseq, reference_time, bandwidth, max_time), axis = 1) # shape = (sample number,)
	if len(timeseq) == 1:
		raise Exception('The length of time sequence must be larger than 1.')
	time_intervals =  timeseq[-1] - timeseq[:-1]
	alphas = alphas.reshape(-1, 1, alphas.shape[-1])
	triggers = np.log(triggering_kernel(alphas, reference_time, time_intervals, bandwidth))
	return -Lambda_0-alphas_times_gtheta+triggers

def log_dirichlet_multinomial_distribution(cls_word_distribution, doc_word_distribution, cls_word_count, doc_word_count, vocabulary_size, priors):
	''' compute the log dirichlet multinomial distribution
		@param:
			1. cls_word_distribution: 1-D numpy array, including document word_distribution
			2. doc_word_distribution: 1-D numpy array
			3. cls_word_count: int, including document word_distribution
			4. doc_word_count: int
			5. vocabulary_size: int
			6. priors: 1-d np.array
		@rtype: float
	'''
	priors_sum = np.sum(priors)
	log_prob = 0
	log_prob += gammaln(cls_word_count - doc_word_count + priors_sum)
	log_prob -= gammaln(cls_word_count + priors_sum)
	log_prob += np.sum(gammaln(cls_word_distribution + priors))
	log_prob -= np.sum(gammaln(cls_word_distribution - doc_word_distribution + priors))
	return log_prob
	
def log_dirichlet_categorical_distribution(mode_type_distribution, event_type, priors):
	''' compute the log dirichlet categorical distribution
		@param:
			1. mode_type_distribution: 1-D numpy array, including event type_distribution
			2. event_type: one-hot vector with non-zero index as activity type.
			3. priors: 1-d np.array
		@rtype: float
	'''
	priors_sum = np.sum(priors)
	mode_type_count = np.sum(mode_type_distribution)

	log_prob = 0
	log_prob += np.log(priors[event_type.astype(bool)][0] + mode_type_distribution[event_type.astype(bool)][0])
	log_prob -= np.log(priors_sum + mode_type_count)

	# print log_prob
	# exit(0)
	return log_prob