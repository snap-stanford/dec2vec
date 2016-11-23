# Joint modeling of neighborhoods and decisions
import argparse
import random

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
	return 1. / (1. + np.exp(-x))
	

def binom(n, p, k):
	return  p ** k * (1. - p) ** (n - k)



def objective(theta, D_map, D_noise_map, nodes_batch, graph):
	res = 0.

	for node in nodes_batch:
		for record in D_map[node]:
			# for each edge attribute
			p = sigmoid(np.dot(theta[record[0]], theta[record[1]]))
			for attr in record[2]:
				# We use Laplace smoothing for p
				p *= binom(record[4], 
					float(record[3][attr][record[2][attr]] + 1) / float(record[4] + len(graph.edge_data[attr])),
					record[3][attr][record[2][attr]])
				
			res += np.log(p) 

		for record in D_noise_map[node]:
			p_noise = sigmoid(np.dot(theta[record[0]], theta[record[1]]))
			for attr in record[2]:
				p_noise *= binom(record[4],
					float(record[3][attr][record[2][attr]] + 1) / float(record[4] + len(graph.edge_data[attr])),
					record[3][attr][record[2][attr]])
			res += np.log(1. - p_noise)
	return res


def objective_gradient(theta, D_map, D_noise_map, nis, graph):
	grad = {node : np.zeros(graph.dimensions) for node in theta.keys()}

	# assumes that we have only one random walk
	for node in graph.g.nodes():
		# Choose a random (noise) neighbor to evaluate the gradient over
		
		ni, ni_noise = nis[node], nis[node]
		c2 = 1.
		for attr in D_noise_map[node][ni_noise][2]:
			c2 *= binom(len(D_noise_map[node]), float(D_noise_map[node][ni_noise][3][attr][D_noise_map[node][ni_noise][2][attr]] + 1) / float(D_noise_map[node][ni_noise][4] + \
						len(D_noise_map[node][ni_noise][3][attr])), D_noise_map[node][ni_noise][3][attr][D_noise_map[node][ni_noise][2][attr]])

		sig_node_neigh = sigmoid(-np.dot(theta[node], theta[D_map[node][ni][1]]))
		sig_node_neigh_noise = sigmoid(-np.dot(theta[node], theta[D_noise_map[node][ni_noise][1]]))
		for k in range(graph.dimensions):
			grad[node][k] += float(theta[D_map[node][ni][1]][k] * sig_node_neigh + c2 * (1. - sig_node_neigh) * sig_node_neigh_noise) /\
								float(1. - c2 * (1. - sig_node_neigh_noise))
		nis[node] += 1
		nis[node] %= len(D_map[node])
	return grad, nis

def setup_sgd(graph):
	theta = dict()
	for node in graph.g.nodes_iter():
		theta[node] =  np.random.random(graph.dimensions)

	return theta


def create_maps(D, D_noise):
	D_map, D_noise_map, D_a_map, D_a_noise_map = dict(), dict(), dict(), dict()

	for rec in D:
		D_map[rec[0]] = []
	for rec in D_noise:
		D_noise_map[rec[0]] = []
	for rec in D:
		D_map[rec[0]].append(rec)
	for rec in D_noise:
		D_noise_map[rec[0]].append(rec)

	
	return D_map, D_noise_map



def stochastic_grad_desc(obj, obj_grad, graph, alpha=0.4, iters=100, tol=1e-5, batch_size=1):
    theta = setup_sgd(graph)
    prev, curr, i = 1e10, 0, 0
    D, D_noise = graph.get_D_D_noise()
    print 'Found noise 	'
    D_map, D_noise_map = create_maps(D, D_noise)
    print 'Starting SGD...'
    iterator = 0
    nis = {node : 0 for node in theta.keys()}
    print 'starting while'
    while iterator < iters:
        prev = curr
#        print 'starting grad'
        grad, nis = obj_grad(theta, D_map, D_noise_map, nis, graph)
#        print 'grad finished'
#        print grad
        for weight in theta.keys():
            theta[weight] += alpha * grad[weight]
        
        curr = obj(theta, D_map, D_noise_map, graph.g.nodes(), graph)
        print curr
#        alpha *= .9
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

	parser.add_argument('--batch-size', default=1, type=int,
                      	help='SGD batch size')

	parser.add_argument('--alpha', default=1, type=float,
                      	help='SGD learning rate')

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)


	return parser.parse_args()

def main(args):
	g = read_graph()
	theta = stochastic_grad_desc(objective, objective_gradient, g, iters=args.iter, alpha=args.alpha, batch_size=args.batch_size)
	
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
