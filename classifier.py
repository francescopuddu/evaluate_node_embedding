from gensim.models import KeyedVectors
import numpy as np
from scipy import sparse
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle as skshuffle
from collections import defaultdict
from sklearn.metrics import f1_score
import json
import sys
import warnings
warnings.filterwarnings("ignore")

from tensorboardX import SummaryWriter
from datetime import datetime

class TopKRanker(OneVsRestClassifier):

	def predict(self, X, top_k_list):
		assert X.shape[0] == len(top_k_list)

		probs = np.asarray(super(TopKRanker, self).predict_proba(X))
		all_labels = sparse.lil_matrix(probs.shape)
		
		for i, k in enumerate(top_k_list):
			probs_ = probs[i, :]
			labels = self.classes_[probs_.argsort()[-k:]].tolist()
			for label in labels:
				all_labels[i,label] = 1
		return all_labels

class ClassificationArguments(): 

    def __init__(self, config_path):

        with open(config_path, 'r') as j:
            conf = json.loads(j.read())

            self.emb = conf["embedding_path"]
            self.label = conf["labels_path"]
            self.shuffle = conf["shuffles"]
            self.metric = conf["metric"]
            self.experiment_name = conf["experiment_name"]

def load_embeddings(embeddings_file):

    model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
    features_matrix = np.asarray([model[str(node)] for node in range(len(model.index_to_key))])

    print("Embeddings successfully loaded.")

    return features_matrix

def load_labels(labels_file, nodesize):

	with open(labels_file) as f:    
		context = f.readlines()
		label = sparse.lil_matrix((nodesize, len(context)))

		for i, line in enumerate(context):
			line = map(int,line.strip().split('\t'))
			for node in line:
				label[node, i] = 1

		print("Labels successfully loaded.")

	return label

def parse_args(config_path):
    args = ClassificationArguments(config_path)
    return args

def evaluate(config_path):
	args = parse_args(config_path)
	features_matrix = load_embeddings(args.emb)
	nodesize = features_matrix.shape[0]
	label_matrix = load_labels(args.label, nodesize)

	# make a number of shuffled copies of the data
	number_shuffles = args.shuffle
	shuffles = []
	for _ in range(number_shuffles):
		shuffles.append(skshuffle(features_matrix, label_matrix))

	all_results = defaultdict(list) # by default, returns an empty list

	training_percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

	for train_percent in training_percents:
		for shuf in shuffles:
			X, y = shuf
			training_size = int(train_percent * nodesize)

			X_train = X[:training_size, :]
			y_train = y[:training_size, :]

			X_test = X[training_size:, :]
			y_test = y[training_size:,:]

			clf = TopKRanker(LogisticRegression())
			clf.fit(X_train, y_train)

			# find out how many labels should be predicted
			# usually not needed
			top_k_list = list(map(int, y_test.sum(axis=1).T.tolist()[0]))
			preds = clf.predict(X_test, top_k_list)

			results = {}
			averages = ["micro", "macro", "samples", "weighted"]
			for average in averages:
				results[average] = f1_score(y_test,  preds, average=average)

			all_results[train_percent].append(results)

	if len(args.experiment_name) > 0:
		writer = SummaryWriter("./tensorboard/" + args.experiment_name + "__" + datetime.now().strftime("%Y%m%d-%H%M%S"))
		print("Starting tensorboard.")
	else:
		print('-------------------')
		print('Train percent :', 'metric value')

	for train_percent in sorted(all_results.keys()):
		av = 0
		stder = np.ones(number_shuffles)
		i = 0
		for x in all_results[train_percent]:
			stder[i] = x[args.metric]
			i += 1
			av += x[args.metric]
		av /= number_shuffles
		if len(args.experiment_name) > 0:
			writer.add_scalar("f1", av, global_step=int(train_percent*100)) 
		else:
			print(train_percent, ":", av)

if __name__ == '__main__':

	if len(sys.argv) == 2: 
		conf_path = sys.argv[1]
	else:
		conf_path = "config.json"
	evaluate(conf_path)
	print("Test completed successfully.")