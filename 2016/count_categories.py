import json
from collections import Counter
with open('train_val_videodatainfo.json', 'r') as handle:
	parsed = json.load(handle)


cat_label_map = {}

with open('category.txt', 'r') as handle:
	content = handle.readlines()
	for line in content:
		cat_name, cat_num = line.rstrip().split('\t')
		cat_label_map[int(cat_num)] = cat_name



cat_labels = []

for video in parsed['videos']:
	cat_labels.append(cat_label_map[video['category']])

print Counter(cat_labels)
#print json.dumps(parsed, indent=4, sort_keys=True)
