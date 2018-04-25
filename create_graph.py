# coding: utf-8

from collections import defaultdict
from random import shuffle
from random import randint
import numpy as np


dir = '/home/rudvlf0413/kge/What-If-QA/'
files = ['dev.txt', 'train.txt', 'test.txt']

entity_id2name = set()
entity_count = defaultdict(int)
relation_id2name = set()


for file in files:
	with open(dir+file) as f:
		for i, line in enumerate(f.readlines()):
			if i % 10000 == 0:
				print(i)

			if line == '':
				break
			head, relation, tail = line.split('\t')
			
			entity_id2name.add(head)
			entity_id2name.add(tail)
			entity_count[head] += 1
			entity_count[tail] += 1

			relation_id2name.add(relation)


entity_id2name = list(entity_id2name)
relation_id2name = list(relation_id2name)

entity_name2id = {n: i for i, n in enumerate(entity_id2name)}
relation_name2id = {n: i for i, n in enumerate(relation_id2name)}

fwrite = open("graph.lg", "w")
fwrite.write("# t 1\n")

for i, entity in enumerate(entity_id2name):
	fwrite.write("v {} 1\n".format(i))

for file in files:
	with open(dir+file) as f:
		for i, line in enumerate(f.readlines()):
			if i % 10000 == 0:
				print(i)

			if line == '':
				break
			head, relation, tail = line.split('\t')

			fwrite.write("e {} {} {}\n".format(entity_name2id[head], entity_name2id[tail], relation_name2id[relation]))
			#fwrite.write("e {} {} 1\n".format(entity_name2id[head], entity_name2id[tail]))
