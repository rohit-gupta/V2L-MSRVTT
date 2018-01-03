import json
import re
import nltk
import collections
import numpy as np
import pickle


entity_tags = ["NN", "NNP", "NNPS", "NNS", "PRP"]
action_tags = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
attribute_tags = ["JJ", "JJR", "JJS"]
all_tag_types = entity_tags + action_tags + attribute_tags

tag_type_map = {}
for tag in entity_tags:
	tag_type_map[tag] = 'entity'

for tag in action_tags:
	tag_type_map[tag] = 'action'

for tag in attribute_tags:
	tag_type_map[tag] = 'attribute'

with open('train_val_videodatainfo.json') as data_file:
    train_data = json.load(data_file)

with open('test_videodatainfo.json') as data_file:
    test_data = json.load(data_file)

captions = test_data['sentences'] + train_data['sentences']

videos = sorted(list(set([x['video_id'] for x in captions])), key=lambda x: int(re.split('(\d+)',x)[-2]))
video_tags = [{'action':[], 'attribute': [], 'entity':[]} for _ in range(len(videos))]
all_tags = {'action':[], 'attribute': [], 'entity':[]}

for sentence in captions:
	tokens = nltk.word_tokenize(sentence['caption'])
	tagged_tokens = nltk.pos_tag(tokens)
	for word,tag in tagged_tokens:
		if tag in all_tag_types:
			video_tags[videos.index(sentence['video_id'])][tag_type_map[tag]] += [word]
			all_tags[tag_type_map[tag]].append(word)


# Number of tags
# entities = 1121, action = 415, attribute = 207
# entities = 625, action = 242, attribute = 110
# Entities = 264, action = 139, attribute = 54


all_entities = collections.Counter(all_tags['entity']).most_common(1121)
all_actions = collections.Counter(all_tags['action']).most_common(415)
all_attributes = collections.Counter(all_tags['attribute']).most_common(207)

selected_tags = {}
selected_tags['entity_tags_long'] = [word for word, _ in all_entities]
selected_tags['action_tags_long'] = [word for word, _ in all_actions]
selected_tags['attribute_tags_long'] = [word for word, _ in all_attributes]


selected_tags['entity_tags_short'] = selected_tags['entity_tags_long'][:264]
selected_tags['action_tags_short'] = selected_tags['action_tags_long'][:139]
selected_tags['attribute_tags_short'] = selected_tags['attribute_tags_long'][:54]
selected_tags['entity_tags_med'] = selected_tags['entity_tags_long'][:625]
selected_tags['action_tags_med'] = selected_tags['action_tags_long'][:242]
selected_tags['attribute_tags_med'] = selected_tags['attribute_tags_long'][:110]


for tag in selected_tags['entity_tags_med']:
	print tag


#for tag_type in selected_tags.keys():
#	words = sorted(selected_tags[tag_type])
#	print len(words)
#	tag_dict = {}
#	for idx,tags in enumerate(video_tags):
#		tagvec = np.zeros(len(words))
#		sel_tags = list(set(tags[tag_type.split("_")[0]]))
#		for tag in sel_tags:
#			if tag in words:
#				tagvec[words.index(tag)] = 1
#		tag_dict[videos[idx]] = tagvec
#	with open(tag_type + '.pickle', 'wb') as handle:
#		pickle.dump(tag_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
