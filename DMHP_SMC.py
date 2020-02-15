from __future__ import print_function
from __future__ import division

import sys
import os
sys.path.append("./")
from DMHP_Utils import *

try:
    import copy_reg
except:
    import copyreg as copy_reg

from copy import deepcopy
import gc

def normalization(x):
    return (x - min(x) + 1e-6) / float(max(x) - min(x) + 1e-5)

class Dirichlet_Hawkes_Process(object):
	"""docstring for Dirichlet Hawkes Prcess"""
	def __init__(self, particle_num, base_intensity, theta0, alpha0, reference_time, vocabulary_size, bandwidth, sample_num):
		super(Dirichlet_Hawkes_Process, self).__init__()
		self.particle_num = particle_num
		self.base_intensity = base_intensity
		self.theta0 = theta0
		self.alpha0 = alpha0
		self.reference_time = reference_time
		self.vocabulary_size = vocabulary_size
		self.bandwidth = bandwidth
		self.sample_num = sample_num
		# initilize particles
		self.particles = []
		for i in range(particle_num):
			self.particles.append(Particle(weight = 1.0/self.particle_num))
		alphas = []; log_priors = []
		for _ in range(sample_num):
			alpha = dirichlet(alpha0); log_prior = log_Dirichlet_CDF(alpha, alpha0)
			alphas.append(alpha); log_priors.append(log_prior)
		self.alphas = np.array(alphas)
		self.log_priors = np.array(log_priors)
		self.active_interval = None # [tu, tn]

	def sequential_monte_carlo(self, event, threshold):
		# print('\n\nhandling event %d' %event.index)
		if isinstance(event, Event): # deal with the case of exact timing
			# get active interval (globally)
			tu = EfficientImplementation(event.timestamp, self.reference_time, self.bandwidth)
			self.active_interval = [tu, event.timestamp]
			# print('active_interval',self.active_interval)
			
			#sequential
			particles = []
			sus_scores = []
			lambda_scores = []
			P_scores = []
			weights = []
			for particle in self.particles:
				_particle, _lambda_score, _P_score, _sus_score = self.particle_sampler(particle, event)
				particles.append(_particle)
				weights.append(_particle.weight)
				sus_scores.append(_sus_score)
				lambda_scores.append(_lambda_score)
				P_scores.append(_P_score)
				# particles.append(self.particle_sampler(particle, event))
			weighted_sus_score = np.inner(sus_scores, weights)
			weighted_lambda_score = np.inner(lambda_scores, weights)
			weighted_P_score = np.inner(P_scores, weights)

			self.particles = particles
			self.particles = self.particles_normal_resampling(self.particles, threshold)
			if (event.index+1) % 100 == 0:
				gc.collect()

			return weighted_lambda_score, weighted_P_score, weighted_sus_score

		else: # deal with the case of exact timing
			raise ValueError('deal with the case of exact timing')
			# print('deal with the case of exact timing')

	def particle_sampler(self, particle, event):
		# sampling mode label
		particle, selected_mode_index, lambda_score, P_score, sus_score= self.sampling_mode_label(particle, event)
		# update the triggering kernel
		particle.modes[selected_mode_index].alpha = self.parameter_estimation(particle, selected_mode_index)
		# calculate the weight update probability
		particle.log_update_prob = self.calculate_particle_log_update_prob(particle, selected_mode_index, event)
		return particle, lambda_score, P_score, sus_score

	def sampling_mode_label(self, particle, event):
		if len(particle.modes) == 0: # the case of the first event comes
			# sample cluster label
			particle.mode_num_by_now += 1
			selected_mode_index = particle.mode_num_by_now
			selected_mode = Mode(index = selected_mode_index)
			selected_mode.add_event(event)
			particle.modes[selected_mode_index] = selected_mode #.append(selected_cluster)
			particle.events2Mode_ID.append(selected_mode_index)
			# update active cluster
			particle.active_modes = self.update_active_modes(particle)
			# if it is the first event, we assume it is a normal event and don't consider its suspicioius score.
			sus_weighted_score_particle = 0
			lambda_weighted_score_particle = 0
			P_weighted_score_particle = 0

		else: # the case of the following document to come
			active_mode_indexes = [0] # zero for new cluster
			active_mode_rates = [self.base_intensity]
			mode0_log_dirichlet_categorical_distribution = \
							log_dirichlet_categorical_distribution(event.type, event.type, self.theta0)
			 
			active_mode_type_probs = [mode0_log_dirichlet_categorical_distribution]
			# first update the active cluster
			particle.active_modes = self.update_active_modes(particle)
			# then calculate rates for each cluster in active interval
			# for active_mode_index, timeseq in particle.active_modes.iteritems():
			for active_mode_index, timeseq in particle.active_modes.items():
				active_mode_indexes.append(active_mode_index)
				time_intervals = event.timestamp - np.array(timeseq)
				alpha = particle.modes[active_mode_index].alpha
				rate = triggering_kernel(alpha, self.reference_time, time_intervals, self.bandwidth)
				active_mode_rates.append(rate)
				# mode_type_distribution = particle.modes[active_mode_index].type_distribution + event.type
				mode_type_distribution = particle.modes[active_mode_index].type_distribution
				event_type  = event.type
				mode_log_dirichlet_catgorical_distribution = \
							log_dirichlet_categorical_distribution(mode_type_distribution, event_type, self.theta0)
				active_mode_type_probs.append(mode_log_dirichlet_catgorical_distribution)

			# print('active_mode_indexes', active_mode_indexes)
			# print('active_mode_rates', active_mode_rates)
			# print('active_mode_type_probs', active_mode_type_probs)

			# FOR MODE_SELECTION_PROBS_BY_TIME & TYPE

			active_mode_logrates = np.log(active_mode_rates)
			mode_selection_probs = active_mode_logrates + active_mode_type_probs  # in log scale
			mode_selection_probs = mode_selection_probs - np.max(mode_selection_probs)  # prevent overflow
			mode_selection_probs = np.exp(mode_selection_probs)
			mode_selection_probs = mode_selection_probs / np.sum(mode_selection_probs)
			# print('mode_selection_probs', mode_selection_probs)

			# FOR MODE_SELECTION_PROBS_BY_TIME
			mode_selection_probs_by_time = active_mode_logrates - np.max(active_mode_logrates) # prevent overflow
			mode_selection_probs_by_time = np.exp(mode_selection_probs_by_time)
			mode_selection_probs_by_time = mode_selection_probs_by_time/np.sum(mode_selection_probs_by_time)
			# print('mode_selection_probs_by_time', mode_selection_probs_by_time)

			# FOR MODE_SELECTION_PROBS_BY_TYPE
			mode_selection_probs_by_type = active_mode_type_probs - np.max(active_mode_type_probs) # prevent overflow
			mode_selection_probs_by_type = np.exp(mode_selection_probs_by_type)
			mode_selection_probs_by_type = mode_selection_probs_by_type/np.sum(mode_selection_probs_by_type)
			# print('mode_selection_probs_by_type', mode_selection_probs_by_type)

			# DERIVE THE SUSPICIOUS SCORE BY TIME, TYPE AND TIME&TYPE
			lambda_weighted_score_particle = np.inner(np.array(active_mode_rates), mode_selection_probs_by_time)
			P_weighted_score_particle = np.inner(np.exp(active_mode_type_probs), mode_selection_probs_by_type)

			lambda_weighted_by_mode_weight = np.inner(np.array(active_mode_rates), mode_selection_probs)
			P_weighted_by_mode_weight = np.inner(np.exp(active_mode_type_probs), mode_selection_probs)
			sus_weighted_score_particle = lambda_weighted_by_mode_weight * P_weighted_by_mode_weight

			np.random.seed()
			selected_mode_array = multinomial(exp_num=1, probabilities = mode_selection_probs)
			selected_mode_index = np.array(active_mode_indexes)[np.nonzero(selected_mode_array)][0]
			# print('selected_mode_index', selected_mode_index)
			if selected_mode_index == 0: # the case of new cluster
				particle.mode_num_by_now += 1
				selected_mode_index = particle.mode_num_by_now
				selected_mode = Mode(index=selected_mode_index)
				selected_mode.add_event(event)
				particle.modes[selected_mode_index] = selected_mode
				particle.events2Mode_ID.append(selected_mode_index)
				particle.active_modes[selected_mode_index] = [self.active_interval[1]] # create a new list containing the current time
			else: # the case of the previous used cluster, update active cluster and add document to cluster
				selected_mode = particle.modes[selected_mode_index]
				selected_mode.add_event (event)
				particle.events2Mode_ID.append(selected_mode_index)
				particle.active_modes[selected_mode_index].append(self.active_interval[1])

		return particle, selected_mode_index, lambda_weighted_score_particle, P_weighted_score_particle, sus_weighted_score_particle

	def parameter_estimation(self, particle, selected_mode_index):
		timeseq = np.array( particle.active_modes[selected_mode_index])
		if len(timeseq) == 1: # the case of first document in a brand new cluster
			np.random.seed()
			alpha = dirichlet(self.alpha0)
			return alpha
		T = self.active_interval[1] + 1 #;print('updating triggering kernel ..., len(timeseq)', len(timeseq))
		alpha = update_triggering_kernel(timeseq, self.alphas, self.reference_time, self.bandwidth, self.base_intensity, T, self.log_priors)
		return alpha

	def update_active_modes(self, particle):
		if not particle.active_modes: # the case of the first document comes
			particle.active_modes[1] = [self.active_interval[1]]
		else: # update the active clusters
			tu = self.active_interval[0]
			# for mode_index in particle.active_modes.keys():
			for mode_index in list(particle.active_modes.keys()):
				timeseq = particle.active_modes[mode_index]
				active_timeseq = [t for t in timeseq if t > tu]
				if not active_timeseq:
					del particle.active_modes[mode_index]
				else:
					particle.active_modes[mode_index] = active_timeseq
		return particle.active_modes
	
	def calculate_particle_log_update_prob(self, particle, selected_mode_index, event):
		mode_type_distribution = particle.modes[selected_mode_index].type_distribution
		event_type = event.type
		log_update_prob = log_dirichlet_categorical_distribution(mode_type_distribution, event_type, self.theta0)
		# print('log_update_prob', log_update_prob)
		return log_update_prob

	def particles_normal_resampling(self, particles, threshold):
		# print('\nparticles_normal_resampling')
		weights = []; log_update_probs = []
		for particle in particles:
			weights.append(particle.weight)
			log_update_probs.append(particle.log_update_prob)

		weights = np.array(weights); log_update_probs = np.array(log_update_probs); 
		# print('weights before update:', weights); print('log_update_probs', log_update_probs)
		log_update_probs = log_update_probs - np.max(log_update_probs) # prevent overflow
		update_probs = np.exp(log_update_probs); #print('update_probs',update_probs)

		weights = weights * update_probs #update 
		weights = weights / np.sum(weights) # normalization
		resample_num = len(np.where(weights + 1e-5 < threshold)[0])
		# print('weights:', weights)
		# print('resample_num:', resample_num)
		if resample_num == 0: 
			for i, particle in enumerate(particles):
				particle.weight = weights[i]
			return particles
		else:
			remaining_particles = [particle for i, particle in enumerate(particles) if weights[i] + 1e-5 > threshold]
			# resample_probs = weights[np.where(weights > threshold + 1e-5)]; resample_probs = resample_probs/np.sum(resample_probs)
			resample_probs = weights[np.where(weights + 1e-5 > threshold )]; resample_probs = resample_probs/np.sum(resample_probs)  # for threshold judge consistency.
			# remaining_particle_weights = weights[np.where(weights > threshold + 1e-5)]
			remaining_particle_weights = weights[np.where(weights + 1e-5 > threshold)]

			# print("***: ", len(remaining_particles), len(remaining_particle_weights))
			for i,_ in enumerate(remaining_particles):
				remaining_particles[i].weight = remaining_particle_weights[i]
			np.random.seed()
			resample_distribution = multinomial(exp_num = resample_num, probabilities = resample_probs)
			if not resample_distribution.shape: # the case of only one particle left
				for _ in range(resample_num):
					new_particle = deepcopy(remaining_particles[0])
					remaining_particles.append(new_particle)
			else: # the case of more than one particle left
				for i, resample_times in enumerate(resample_distribution):
					for _ in range(resample_times):
						new_particle = deepcopy(remaining_particles[i])
						remaining_particles.append(new_particle)
			# normalize the particle weight again
			update_weights = np.array([particle.weight for particle in remaining_particles]);
			update_weights = update_weights / np.sum(update_weights)
			for i, particle in enumerate(remaining_particles):
				particle.weight = update_weights[i]
			assert np.abs(np.sum(update_weights) - 1) < 1e-5
			assert len(remaining_particles) == self.particle_num
			self.particles = None
			return remaining_particles


def parse_activity_2_event(activity, vocabulary_size):
	''' convert (id, timestamp, type) to the form of event
	'''
	index = activity[0]
	timestamp = activity[1]/3600.0 # unix time in hour
	type = np.zeros(vocabulary_size)
	type[activity[2]] = 1   # one-hot embedding for mark type.
	event = Event(index, timestamp, type.astype(int))
	return event

def SMC(usrid, _lambda_0, root_path):

	if usrid in ["ACM2278", "CMP2946", "PLJ1771", "CDE1846", "MBG3183"]:
		streaming = np.load("%s/InsiderData/time_activity_%s.npy"%(root_path, usrid))
	else:
		streaming = np.load("%s/NormalData/time_activity_%s.npy"%(root_path, usrid))

	activities = []
	for i, _activity in enumerate(streaming):
		activities.append([i, float(_activity[0]), int(_activity[1])])

	# parameter initialization
	vocabulary_size = 23
	particle_num = 8
	base_intensity = _lambda_0
	theta0 = np.array([0.01] * vocabulary_size)

	alpha0 = np.array([0.1] * 7)
	reference_time = np.array([3, 7 ,11, 24, 24*2, 24*4, 24*8])
	bandwidth = np.array([1, 5, 7, 12, 24, 24, 24])


	sample_num = 2000
	threshold = 1.0/particle_num

	DHP = Dirichlet_Hawkes_Process(particle_num = particle_num, base_intensity = base_intensity, theta0 = theta0, alpha0 = alpha0, \
		reference_time = reference_time, vocabulary_size = vocabulary_size, bandwidth = bandwidth, sample_num = sample_num)

	# begin sampling
	# for simple experiment
	events_sus_scores = []  # for the first event, we don't consider its suspicious score and set it as 0 by default.
	events_lambda_scores = []
	events_P_scores = []
	for activity in activities:
		event = parse_activity_2_event(activity = activity, vocabulary_size = vocabulary_size)
		lambda_score, P_score, sus_score = DHP.sequential_monte_carlo(event, threshold)
		events_sus_scores.append(sus_score)
		events_lambda_scores.append(lambda_score)
		events_P_scores.append(P_score)

	events_sus_scores = normalization(events_sus_scores)
	events_lambda_scores = normalization(events_lambda_scores)
	events_P_scores = normalization(events_P_scores)


	### Check if output path exists or not.
	if not os.path.exists("./dmhpOutput/score/"):
		os.makedirs("./dmhpOutput/score/")

	if not os.path.exists("./dmhpOutput/f_t/"):
		os.makedirs("./dmhpOutput/f_t/")

	if not os.path.exists("./dmhpOutput/f_y/"):
		os.makedirs("./dmhpOutput/f_y/")

	np.save('./dmhpOutput/score/%s_score'%(usrid), events_sus_scores)
	np.save('./dmhpOutput/f_t/%s_f_t'%(usrid), events_lambda_scores)
	np.save('./dmhpOutput/f_y/%s_f_y'%(usrid), events_P_scores)

	with open('./dmhpOutput/%s.pkl'%(usrid), 'wb') as w:
		pickle.dump(DHP.particles, w)