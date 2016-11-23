# Joint modeling of neighborhoods and decisions
import argparse

import networkx as nx
import numpy as np
import scipy as sp

from itertools import product

np.seterr(over='ignore')

class MyGraph:

	def __init__(self, nx_graph, dimensions, neighborhood_size, is_directed=False, node_attributes=False, 
				 walks_per_node=1):
		self.is_directed = is_directed
		self.node_attributes = node_attributes
		self.neighborhood_size = neighborhood_size
		self.walks_per_node = walks_per_node
		self.dimensions = dimensions
		self.edge_data = dict()

		for edge in nx_graph.edges_iter(data=True):
			self.edge_data = merge_dicts(self.edge_data, edge[2])

		if is_directed:
			self.g = nx.DiGraph(nx_graph)
		else:
			self.g = nx.Graph(nx_graph)


	def bfs_neighborhood(self, node):
		neighborhood = self.g.edge[node]
		nodes = np.random.choice(neighborhood.keys(), 
								 size=min(self.neighborhood_size, len(self.g.edge[node])),
								 replace=False)
		n_bfs = dict()
		for node in nodes:
			n_bfs[node] = neighborhood[node]
		return n_bfs


	def cnt_edge_data(self, neighborhood):
		cnt = dict()
		for attr in self.edge_data:
			cnt[attr] = dict()
			for val in self.edge_data[attr]:
				cnt[attr][val] = 0

		for neighbor in neighborhood:
			edge_data = neighborhood[neighbor]
			for attr in edge_data:
				try:
					cnt[attr][edge_data[attr]] += 1
				except TypeError:
					cnt[attr][edge_data[attr][0]] += 1

		return cnt


	def get_D_D_noise(self):
		D, D_noise = [], []
		for node in self.g.nodes_iter():
			for walk in range(self.walks_per_node):

				neighborhood = self.bfs_neighborhood(node)
				cnt = self.cnt_edge_data(neighborhood)

				for neighbor in neighborhood:
					D.append((node, neighbor, neighborhood[neighbor],
							  cnt, len(neighborhood)))

				noise_neighbors = list(set(self.g.nodes_iter()) - \
								       set(neighborhood))
				noise_neighbors = np.random.choice(noise_neighbors, 
													  size=len(neighborhood),
													  replace=False)

				for noise_neighbor in noise_neighbors:

					noise_data, noise_neighborhood = dict(), dict()

					# In case there is no edge
					data = None
					try:
						data = self.g[node][noise_neighbor]
						for attr in data:
							data_attr = set()
							data[attr] = [data[attr]]
							for val in data[attr]:
								data_attr.add(val)
							noise_data_attr = self.edge_data[attr] - data_attr
							noise_data[attr] = np.random.choice(list(noise_data_attr), size=1)[0]
					except KeyError:
						data = self.edge_data
						for attr in data:
							noise_data[attr] = np.random.choice(list(data[attr]), size=1)[0]
			
					noise_neighborhood[noise_neighbor] = noise_data
					noise_cnt = self.cnt_edge_data(noise_neighborhood)

					D_noise.append((node, noise_neighbor, noise_data,
									noise_cnt, len(noise_neighbors)))

					# the noise neighborhood might include edge data iff
					# an edge exists, or otherwise, no node data, which
					# means that no edge has ever existed between them
		return remove_one_elem_lists(D, D_noise)

		# todo: extend to node attributes

		# D[0] = source node
		# D[1] = destination node
		# D[2] = edge data
		# D[3] = cnt(edge data) in N_S(source) -- {attr : {val : cnt}}
		# D[4] = the size of the neighborhood the edge belongs to



def remove_one_elem_lists(D, D_noise):
	for i in range(len(D)):
		for attr in D[i][2]:
			if type(D[i][2][attr]) == type(list()):
				D[i][2][attr] = D[i][2][attr][0]

	for i in range(len(D_noise)):
		for attr in D_noise[i][2]:
			if type(D_noise[i][2][attr]) == type(list()):
				D_noise[i][2][attr] = D_noise[i][2][attr][0]

	return D, D_noise






def merge_dicts(d1, d2):
	d = dict()

	if len(d1) > 0:
		for k1 in d1:
			d[k1] = set(list(d1[k1]))
	for k2 in d2:
		if len(d1) > 0:
			try:
				d[k2] = d[k2] | set(list(d2[k2]))
			except TypeError:
				d[k2] = d[k2] | set([d2[k2]])
		else:
			d[k2] = set([d2[k2]])
	return d




def sigmoid(x):
	return 1. / (1. + np.exp(float(-x)))
	

def binom(n, p, k):
	return sp.special.binom(n, k) * p ** k * (1. - p) ** (n - k)


def objective(theta, balance, D_map, D_noise_map, nodes_batch, graph):
	res_nodes = 0.
	res_decisions = 0.

	try:
		assert len(D) == len(D_noise)
	except AssertionError:
		print 'D and D_noise have different lengths, computing separately...'

	for node in nodes_batch:
		for record in D_map[node]:
			res_nodes += np.log(sigmoid(np.dot(theta[record[0]], theta[record[1]])))
			p_decision = 1.
			for attr in graph.edge_data:
				theta_average_decision_i01 = np.zeros(graph.dimensions)
				cnt_decision = 0
				for j in range(len(D)):
					# for all decisions of the same type as i-edge 
					if D[j][2][attr] == record[2][attr]:
						cnt_decision += 1
						theta_average_decision_i01 += theta[record[1]]
				p_decision *= sigmoid(np.dot(theta[D[i][0]], 1./float(cnt_decision) * theta_average_decision_i01))

		for record in D_noise_map[node]:
			res_nodes += np.log(1. - sigmoid(np.dot(theta[D_noise[i][0]], theta[D_noise[i][1]])))
			p_decision_noise = 1.
			for attr in graph.edge_data:
				theta_average_decision_i01_noise = np.zeros(graph.dimensions)
				cnt_decision_noise  0
				for j in range(len(D)):
					if D_noise[j][2][attr] == record[2][attr]:
						cnt_decision_noise += 1
						theta_average_decision_i01_noise += theta[D_noise[j][1]]
				p_decision_noise *= (1. - sigmoid(np.dot(theta[D_noise[i][0]], 1./float(cnt_decision_noise) * theta_average_decision_i01_noise)))

		res_decisions += np.log(p_decision) +  np.log(p_decision_noise)

	return balance * res_nodes + (1. - balance) * res_decisions


def objective_gradient(theta, balance, D_map, D_noise_map, nodes_batch, graph):
	grad_nodes, grad_decisions = dict(theta), dict(theta)


	for node in D_map.keys():
		for k in range(graph.dimensions):
			grad_nodes[node][k] = 0.
			for record in D_map[node]:
				grad_nodes[node][k] += theta[record[1]][k] * sigmoid(-np.dot(theta[node], theta[record[1]]))
			for record in D_noise_map[node]:
				grad_nodes[node][k] -= theta[record[1]][k] * sigmoid(np.dot(theta[node], theta[record[1]]))

		grad_nodes[node] *= balance

		for k in range(graph.dimensions):
			grad_decisions[node][k] = 0.
			for record in D_map[node]:
				for attr in record[2]:
					theta_average_decision_i01 = np.zeros(graph.dimensions)
					cnt_decision = 0
					for neighbor_record in D_map[node]:
						if neighbor_record[2][attr] == record[2][attr]:
							theta_average_decision_i01 += theta[neighbor_record[1]]
							cnt_decision += 1

					grad_decisions[node][k] += sigmoid(-np.dot(theta[node], 1./float(cnt_decision) * theta_average_decision_i01))

			for record in D_noise_map[node]:
				for attr in record[2]:
					theta_average_decision_i01_noise = np.zeros(graph.dimensions)
					cnt_decision_noise = 0
					for neighbor_record in D_noise_map[node]:
						if neighbor_record[2][attr] == record[2][attr]:
							theta_average_decision_i01_noise += theta[neighbor_record[1]]
							cnt_decision_noise += 1
					grad_decisions[node][k] -= sigmoid(np.dot(theta[node], 1./float(cnt_decision_noise) * theta_average_decision_i01_noise))

	grad = dict(grad_nodes)
	for node in grad:
		grad[node] = balance * grad_nodes[node] + (1. - balance) * grad_decisions[node]

	return grad


def setup_sgd(graph):
	theta = dict()
	for node in graph.g.nodes_iter():
		theta[node] =  np.random.random(graph.dimensions)

	return theta


def stochastic_grad_desc(obj, obj_grad, graph, balance, alpha=0.05, iters=100, tol=1e-5, batch_size=1):

	theta = setup_sgd(graph)
	prev, curr, i = 1e10, 0, 0
	D, D_noise = graph.get_D_D_noise()

	print 'Found noise 	'
	D_map, D_noise_map = dict(), dict()
	for rec in D:
		D_map[rec[0]] = []
	for rec in D_noise:
		D_noise_map[rec[0]] = []
	for rec in D:
		D_map[rec[0]].append(rec)
	for rec in D_noise:
		D_noise_map[rec[0]].append(rec)

	print 'Starting SGD...'
	iterator = 0
	#abs(curr - prev) > tol and 
	while iterator < iters:
		prev = curr

		# SGD sampling step
		batch = np.random.choice(range(len(D)), size=batch_size, replace=False)
		batch_D, batch_D_noise = [D[i] for i in batch], [D_noise[i] for i in batch]
		batch_D_map, batch_D_noise_map = dict(), dict()
		for i in batch:
			batch_D_map[D[i][0]] = D_map[D[i][0]]
			batch_D_noise_map[D[i][0]] = D_noise_map[D[i][0]]
		# SGD update step (maximization)
		
		grad = obj_grad(theta, balance, batch_D_map, batch_D_noise_map, graph)
		
		for node in batch_D_map.keys():
			theta[node] += alpha * (grad[node])

		# print grad['10']
		if iterator % 10 == 0:
		 	curr = obj(theta, balance, D, D_noise, graph)

			print iterator, curr
		iterator += 1

	return theta

def read_graph():
	if args.directed:
		g = nx.read_edgelist(args.input, nodetype=str, data=[('Release_Detained', int)], create_using=nx.DiGraph())
	else:
		g = nx.read_edgelist(args.input, nodetype=str, data=[('Release_Detained', int)], create_using=nx.Graph())
	# instantiate MyGraph using G
	myg = MyGraph(g, args.dimensions, args.window_size, is_directed=False, node_attributes=False, walks_per_node=args.num_walks)
	print 'Graph created...'
	return myg

def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run decision2vec.")

	parser.add_argument('--input', nargs='?', default='graph/baildata-top20k-judge-client-decision.edgelist',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='emb/baildata-top20k-judge-client-decision.emb',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
                      	help='Number of epochs in SGD')

	parser.add_argument('--balance', default=0.5, type=float,
						help='Balance between node and decision learning objectives')
	parser.add_argument('--alpha', default=0.2, type=float,
						help='SGD learning rate')

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)


	return parser.parse_args()

def main(args):
	g = read_graph()
	theta = stochastic_grad_desc(objective, objective_gradient, g, alpha = args.alpha, iters=args.iter, balance=args.balance)
	
	out = np.empty(shape=(len(theta), g.dimensions+1), dtype='|S50')
	i = 0
	for node in theta:
		vec = theta[node]
		out[i][0] = node
		for j in range(len(vec)):
			out[i][j+1] = str(vec[j])
		i += 1	

	np.savetxt(args.output, out, fmt="%s")



args = parse_args()
main(args)
