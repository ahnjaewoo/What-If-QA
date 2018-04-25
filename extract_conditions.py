# coding: utf-8

from collections import defaultdict
import matplotlib.pyplot as plt
from random import shuffle
from random import randint
import numpy as np


dir = '/home/rudvlf0413/kge/What-If-QA/'
files = ['train.txt', 'test.txt']

entity_count = defaultdict(int)
relation_count = defaultdict(int)
triple_indices = defaultdict(set)
triple_set = set()
# entity_r = dict()


for file in files:
	with open(dir+file) as f:
		for i, line in enumerate(f.readlines()):
			if i % 10000 == 0:
				print(i)

			if line == '':
				break
			head, relation, tail = line.split('\t')
			
			entity_count[head] += 1
			entity_count[tail] += 1
			relation_count[relation] += 1
			triple_set.add((head, relation, tail))


triples = list(triple_set)
triples_idx = set(range(len(triples)))
del triple_set
shuffle(triples)
for i, (head, relation, tail) in enumerate(triples):
	triple_indices[head].add(i)
	triple_indices[tail].add(i)


entity_name = list(entity_count.keys())
entity_prob = np.array(list(entity_count.values()), dtype=np.float32)


"""
plt.hist(entity_prob, bins=1000)
plt.show()
"""


# min_degree보다 작은 경우는 condition sampling 안하도록
min_degree = 20
entity_prob[entity_prob<min_degree] = 0
entity_prob /= sum(entity_prob)

num_node = int(len(entity_prob)*0.3)
random_idx = np.random.choice(len(entity_name), size=num_node, p=entity_prob, replace=False)

train_list = []
condition_list = []
query_list = []

queryRate = 0.04
validatio_ratio = 0.09

for k, idx in enumerate(random_idx):
	if k % 1000 == 0:
		print(k)

	target = entity_name[idx]

	selected_triple_indices = list(triple_indices[target])
	if len(selected_triple_indices) < min_degree:
		continue


	queryNum = int(len(selected_triple_indices)*queryRate)
	i = queryNum
	for index in selected_triple_indices[:queryNum]:
		if index in triples_idx:
			# select query
			triple = triples[index]
			query_list.append(list(triple))

			triples_idx.remove(index)
			triple_indices[triple[0]].remove(index)
			triple_indices[triple[2]].remove(index)

			# select condition
			numConditionPerQuery = randint(2,5)
			conditions_per_query = []
			for condition_index in selected_triple_indices[i:i+numConditionPerQuery]:
				if condition_index in triples_idx:
					triple = triples[condition_index]
					conditions_per_query.append(list(triple))
					triples_idx.remove(condition_index)

					triple_indices[triple[0]].remove(condition_index)
					triple_indices[triple[2]].remove(condition_index)

			condition_list.append(conditions_per_query)	
			i += numConditionPerQuery


	for index in selected_triple_indices[i:]:
		if index in triples_idx:
			triple = triples[index]
			train_list.append(list(triple))

			triples_idx.remove(index)
			triple_indices[triple[0]].remove(index)
			triple_indices[triple[2]].remove(index)


#나머지들 다 train에 넣기
triples_idx = list(triples_idx)
for idx in triples_idx:
	train_list.append(list(triples[idx]))


with open(dir+"train_.txt" , "w") as f:
	for triple in train_list:
		f.write("{}".format("\t".join(triple)))

with open(dir+"condition_.txt" , "w") as f:
	for conditions_per_query in condition_list:
		for triple in conditions_per_query:
			f.write("{}".format("\t".join(triple)))
		f.write("-----\n")

with open(dir+"query_.txt" , "w") as f:
	for triple in query_list:
		f.write("{}".format("\t".join(triple)))
