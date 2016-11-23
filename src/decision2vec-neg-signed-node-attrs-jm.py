# Joint modeling of neighborhoods and decisions
import argparse
import StringIO

import networkx as nx
import numpy as np
import scipy as sp

from itertools import product
from scipy import stats

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
		self.pattern_nodes = dict()
		self.patterns = dict()
		self.pattern_node_neighbors = dict()

		# pattern_nodes structure: {source_node : List of pattern nodes ids p1, p2, ...}
		# pattern_data structure: {pattern_node_id : (attr, val)}
		for edge in nx_graph.edges_iter(data=True):
			self.edge_data = merge_dicts(self.edge_data, edge[2])

		if is_directed:
			self.g = nx.DiGraph(nx_graph)
		else:
			self.g = nx.Graph(nx_graph)


	def create_pattern_nodes(self):
		vals = {}
		pid = 0
		for node_data in self.g.nodes_iter(data=True):
			# create a pattern node for each attribute value
			node_id = node_data[0]
			data = node_data[1] 
			self.pattern_nodes[node_id] = list()
			temp_id = pid
			for attr in data:
				val = data[attr]
				if attr not in vals:
					vals[attr] = {}
				elif val not in vals[attr]:
					vals[attr][val] = pid
				else:
					temp_id = vals[attr][val]

				pattern_id = 'p' + str(temp_id)

				# add an pattern_id entry for our node
				self.pattern_nodes[node_id].append(pattern_id)
				# register the pattern node
				self.patterns[pattern_id] = (attr, val)
				if temp_id == pid:
					pid += 1

		
		for pattern_node in self.patterns:
			self.pattern_node_neighbors[pattern_node] = []
			for node in self.g.nodes_iter():
				if pattern_node in self.pattern_nodes[node]:
					self.pattern_node_neighbors[pattern_node].append(pattern_node)

		


	def bfs_neighborhood(self, node):
		neighborhood = self.g.edge[node]
		nodes = np.random.choice(neighborhood.keys(), 
								 size=min(self.neighborhood_size, len(self.g.edge[node])),
								 replace=False)
		n_bfs = dict()
		for node in nodes:
			n_bfs[node] = neighborhood[node]
		return n_bfs

	def bfs_neighborhood_pattern(self, pattern_node, neighborhood_map):

		pattern_neighbors = set()

		# 1. Find all nodes connected to this pattern node
		nodes = []
		for node in self.pattern_nodes:
			if pattern_node in self.pattern_nodes[node]:
				nodes.append(node)

		# 2. Get all pattern nodes of all neighbors of a node
		for node in nodes:
			for neighbor in neighborhood_map[node]:
				pattern_neighbors = pattern_neighbors | set(self.pattern_nodes[neighbor])

		# 3. Sample self.neighborhood_size pattern neighbors. 
		neighborhood =  np.random.choice(np.array(list(pattern_neighbors)), size=min(self.neighborhood_size, len(pattern_neighbors)), replace=False).tolist()

		# 4. Sample self.neighborhood_size noisy pattern neighbors
		others = list(set(self.patterns.keys()) - set(pattern_neighbors))

		if len(others) > 0:
			neighborhood_noise = np.random.choice(others, 
											   	  size=min(self.neighborhood_size, len(others)),
											      replace=False).tolist()
		else:
			neighborhood_noise = []
		


		return neighborhood, neighborhood_noise


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
		D, D_noise, D_a, D_a_noise = [], [], [], []
		# structure: {node: list(neighbors)}
		neighborhoods_raw = dict()
		for node in self.g.nodes_iter():
			for walk in range(self.walks_per_node):

				neighborhood = self.bfs_neighborhood(node)
				cnt = self.cnt_edge_data(neighborhood)

				# save the neighborhood and use it for patterns later
				neighborhoods_raw[node] = neighborhood.keys()


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


		for pattern_node in self.patterns:
			pattern_neighbors, pattern_neighbors_noise = self.bfs_neighborhood_pattern(pattern_node, neighborhoods_raw)
			for pattern_neighbor in pattern_neighbors:
				D_a.append((pattern_node, pattern_neighbor))

			# If it wasn't possible to generate noise, we simply don't have noise for this node
			for noisy_pattern_neighbor in pattern_neighbors_noise:
				D_a_noise.append((pattern_node, noisy_pattern_neighbor))
			

		return remove_one_elem_lists(D, D_noise, D_a, D_a_noise)

		# todo: extend to node attributes

		# D[0] = source node
		# D[1] = destination node
		# D[2] = edge data
		# D[3] = cnt(edge data) in N_S(source) -- {attr : {val : cnt}}
		# D[4] = the size of the neighborhood the edge belongs to

		# D_a[0] = src pattern node
		# D_a[1] = dst pattern node





def remove_one_elem_lists(D, D_noise, D_a, D_a_noise):
	for i in range(len(D)):
		for attr in D[i][2]:
			if type(D[i][2][attr]) == type(list()):
				D[i][2][attr] = D[i][2][attr][0]

	for i in range(len(D_noise)):
		for attr in D_noise[i][2]:
			if type(D_noise[i][2][attr]) == type(list()):
				D_noise[i][2][attr] = D_noise[i][2][attr][0]

	

	return D, D_noise, D_a, D_a_noise






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
	ret = 1. / (1. + np.exp(float(-x)))
	if ret > 0.5:
		ret -= 1e-10
	elif ret < 0.5:
		ret += 1e-10
	return ret

def binom(n, p, k):
	return sp.special.binom(n, k) * p ** k * (1. - p) ** (n - k)

def cosine_distance(v1, v2):
	return 1. - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

np.seterr(invalid='raise')
def objective(theta, D_map, D_noise_map, D_a_map, D_a_noise_map, nodes_batch, patterns_batch, graph):
	res = 0.

	for i in nodes_batch:
		# D_map[i] has num_walks records
		for record in D_map[i]:
			p = sigmoid(np.dot(theta[record[0]], theta[record[1]]))
			for attr in record[2]:
				# We use Laplace smoothing for p
				p *= binom(record[4], 
					float(record[3][attr][record[2][attr]] + 1) / float(record[4] + len(graph.edge_data[attr])),
					record[3][attr][record[2][attr]])
			
			res += np.log(p) 

		for record in D_noise_map[i]:
			p_noise = sigmoid(np.dot(theta[record[0]], theta[record[1]]))
			for attr in record[2]:
				p_noise *= binom(record[4],
				float(record[3][attr][record[2][attr]] + 1) / float(record[4] + len(graph.edge_data[attr])),
				record[3][attr][record[2][attr]])

			res += np.log(1. - p_noise)

	for node in nodes_batch:
		distance = 0.
		for pattern_node in graph.pattern_nodes[node]:
			distance += cosine_distance(theta[node], theta[pattern_node])
		res -= distance

	for pattern in patterns_batch:
		for pattern_neighbor in D_a_map[pattern]:
			res += np.log(sigmoid(np.dot(theta[pattern], theta[pattern_neighbor])))
		for pattern_neighbor in D_a_noise_map[pattern]:
			res += np.log(1. - sigmoid(np.dot(theta[pattern], theta[pattern_neighbor])))
    
	return res


def objective_gradient(theta, D_map, D_noise_map, D_a_map, D_a_noise_map, nodes_batch, patterns_batch, graph):
	"""
	Maps structure is {node : [neighbor records]}.
	Assumes that we have one random walk per node
	"""
	grad = dict(theta)
ararrrrr 





	# Initialize gradients
	for node in nodes_batch:
		grad[node] = np.zeros(graph.dimensions)
	for pattern in patterns_batch:
		grad[pattern] = np.zeros(graph.dimensions)

	# Partial derivatives w.r.t. nodes in nodes_batch
	for node in nodes_batch:
		for k in range(graph.dimensions):

			# Gradient section: nodes + decisions
			grad[node][k] = 0.
			for record in D_map[node]:
				binom_p = 1.
				for attr in record[2]:
					# record[1] is the destination node (neighbor)
					grad[node][k] += theta[record[1]][k] * (1. - sigmoid(np.dot(theta[node], theta[record[1]])))
			for record in D_noise_map[node]:
				binom_p_noise = 1.
				for attr in record[2]:
					binom_p_noise *= binom(len(record), float(record[3][attr][record[2][attr]] + 1) / float(record[4] + \
								 	 len(record[3][attr])), record[3][attr][record[2][attr]])
				grad[node][k] += theta[record[1]][k] * binom_p_noise * sigmoid(np.dot(theta[node], theta[record[1]])) * \
								 sigmoid(-np.dot(theta[node], theta[record[1]])) / (1. - binom_p_noise * sigmoid(np.dot(theta[node], theta[record[1]])))

			# Gradient section: patterns
			for k in range(graph.dimensions):
				for pattern in graph.pattern_nodes[node]:
					grad[node][k] -= theta[pattern][k] * np.linalg.norm(theta[node]) ** 2 - theta[node][k] * np.dot(theta[node], theta[pattern]) /\
									 (np.linalg.norm(theta[node]) ** 3 * np.linalg.norm(theta[pattern]))


	# Partial derivatives w.r.t. patterns in patterns_batch
	for pattern_node in patterns_batch:
		for k in range(graph.dimensions):
			grad[pattern_node][k] = 0.
			for pattern_neighbor in D_a_map[pattern_node]:
				grad[pattern_node][k] += sigmoid(-np.dot(theta[pattern_node], theta[pattern_neighbor])) * theta[pattern_neighbor][k]

	for pattern_node in patterns_batch:
		for k in range(graph.dimensions):
			for pattern_neighbor in D_a_noise_map[pattern_node]:
				grad[pattern_node][k] -= sigmoid(np.dot(theta[pattern_node], theta[pattern_neighbor])) * theta[pattern_neighbor][k]

	for pattern in patterns_batch:
		for node in graph.pattern_node_neighbors[pattern_node]:
			grad[pattern_node][k] -= (theta[node][k] * np.linalg.norm(theta[pattern_node]) ** 2 - \
									  theta[pattern_node][k] * np.dot(theta[node], theta[pattern_node])) / \
									 (np.linalg.norm(theta[node] * np.linalg.norm(theta[pattern_node] ** 3)))

	return grad



def setup_sgd(graph):
	theta = dict()
	for node in graph.g.nodes_iter():
		theta[node] =  1e-5 * np.random.random(graph.dimensions)
	for pattern_node in graph.patterns:
		theta[pattern_node] = 1e-10 * np.random.random(graph.dimensions)

	return theta

def create_maps(D, D_noise, D_a, D_a_noise):
	D_map, D_noise_map, D_a_map, D_a_noise_map = dict(), dict(), dict(), dict()

	for rec in D:
		D_map[rec[0]] = []
	for rec in D_noise:
		D_noise_map[rec[0]] = []
	for rec in D:
		D_map[rec[0]].append(rec)
	for rec in D_noise:
		D_noise_map[rec[0]].append(rec)

	for rec in D_a:
		D_a_map[rec[0]] = []
	for rec in D_a_noise:
		D_a_noise_map[rec[0]] = []
	for rec in D_a:
		D_a_map[rec[0]].append(rec[1])
	for rec in D_a_noise:
		D_a_noise_map[rec[0]].append(rec[1])

	return D_map, D_noise_map, D_a_map, D_a_noise_map



def stochastic_grad_desc(obj, obj_grad, graph, alpha=0.4, iters=100, tol=1e-5, batch_size=20):

	theta = setup_sgd(graph)
	prev, curr, i = 1e10, 0, 0
	D, D_noise, D_a, D_a_noise = graph.get_D_D_noise()
	print 'Found noise 	'
	D_map, D_noise_map, D_a_map, D_a_noise_map = create_maps(D, D_noise, D_a, D_a_noise)
	

	print 'Starting SGD...'
	iterator = 0
	while iterator < iters:
		prev = curr

		# SGD sampling step
		
		# 1. Get a sample from the vertex set
		batch = np.random.choice(graph.g.nodes(), size=batch_size, replace=False)
		batch_D_map, batch_D_noise_map = {i : D_map[i] for i in batch}, {i : D_noise_map[i] for i in batch}
		
		# batch_D_map, batch_D_noise_map = dict(), dict()
		# for i in batch:
		# 	batch_D_map[D[i][0]] = D_map[D[i][0]]
		# 	batch_D_noise_map[D[i][0]] = D_noise_map[D[i][0]]

		# 2. Get a sample from the patterns set
		batch_a = np.random.choice(graph.patterns.keys(), size=min(batch_size, len(D_a)), replace=False)
		batch_D_a_map, batch_D_a_noise_map = {i : D_a_map[i] for i in batch_a}, {i : D_a_noise_map[i] for i in batch_a}

		# batch_D_a_map, batch_D_a_noise_map = dict(), dict()
		# for i in batch_a:
		# 	batch_D_a_map[D_a[i][0]] = D_a_map[D_a[i][0]]
		# 	batch_D_a_noise_map[D_a[i][0]] = D_a_noise_map[D_a[i][0]]

		# SGD update step (maximization)
		
		grad = obj_grad(theta, batch_D_map, batch_D_noise_map, batch_D_a_map, batch_D_a_noise_map, batch, batch_a, graph)
		
		for node in batch_D_map.keys():
			theta[node] += alpha * grad[node]

		for pattern in batch_D_a_map.keys():
			theta[pattern] += alpha * grad[pattern]

		# print grad['10']
		if iterator % 10 == 0:
		 	#curr = obj(theta, D_map, D_noise_map, D_a_map, D_a_noise_map, graph.g.nodes(), graph.patterns.keys(), graph)
			#print iterator, curr
			print iterator
		iterator += 1

	return theta

def read_graph():

	colnames = []
	edge_attributes = args.edge_attribute_names.split(',')
	colnames += edge_attributes

	node_attributes = dict()
	for node_input in args.node_names_attributes.split(':'):
		names = node_input.split(',')
		node_attributes[names[0]] = names[1:]
		colnames += names

	edge_attribute_types = []
	for edge_attr in edge_attributes:
		edge_attribute_types.append((edge_attr, str))


	if args.directed:
		g = nx.read_edgelist(args.input, nodetype=str, data=edge_attribute_types, create_using=nx.DiGraph())
	else:
		g = nx.read_edgelist(args.input, nodetype=str, data=edge_attribute_types, create_using=nx.Graph())

	x = np.genfromtxt(args.attributes_file, delimiter='\t', dtype=None, names=True, usecols=colnames)
	
	for i in range(len(x)):
		for node_type in node_attributes:
			for node_attr in node_attributes[node_type]:
				g.add_node(str(x[i][node_type]), {str(node_attr): x[i][node_attr]})
	myg = MyGraph(g, args.dimensions, args.window_size, is_directed=False, node_attributes=False, walks_per_node=args.num_walks)
	myg.create_pattern_nodes()

	print 'Graph created...'
	return myg


def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run decision2vec.")

	parser.add_argument('--input', nargs='?',
	                    help='Input edgelist path')

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

	parser.add_argument('--attributes-file', default=1, type=str,
                      	help='File to read node attributes from')

	parser.add_argument('--node-names-attributes', type=str,
						help='Dictionary that maps node id columns to node attribute names')
						# format: node_id_col_name,attr1,attr2,...,attrk;node_id_col_name,attr1,attr2,...

	parser.add_argument('--edge-attribute-names', type=str,
						help='Attribute names for edge data.')

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')

	parser.add_argument('--alpha', type=float,
						help='SGD learning rate')

	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)


	return parser.parse_args()


def create_column_dictionary(argument):
	mapping = dict()
	
	node_ids = argument.split(':')
	for node_id in node_ids:
		lst = node_id.split(',')
		nid = lst[0]
		attrs = list[1:]
		mapping[nid] = attrs

	return mapping

def main(args):
	print args
	g = read_graph()
	theta = stochastic_grad_desc(objective, objective_gradient, g, iters=args.iter, alpha=args.alpha)
	
	out = np.empty(shape=(len(theta), g.dimensions+1), dtype='|S50')
	i = 0
	for node in theta:
		# skip pattern nodes 
		if node in g.patterns.keys():
			continue

		vec = theta[node]
		out[i][0] = node
		for j in range(len(vec)):
			out[i][j+1] = str(vec[j])
		i += 1	

	np.savetxt(args.output, out, fmt="%s")



args = parse_args()
main(args)
