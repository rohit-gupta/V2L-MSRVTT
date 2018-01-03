import json
import re
# Load Captions Data
annotations_file = "train_val_videodatainfo.json"
with open(annotations_file) as data_file:
                        train_data = json.load(data_file)

# Load Captions Data
test_annotations_file = "test_videodatainfo.json"
with open(test_annotations_file) as data_file:
                        test_data = json.load(data_file)

print len(train_data['sentences']), len(test_data['sentences'])

test_captions = test_data['sentences']
train_captions = train_data['sentences']



train_videos = sorted(list(set([x['video_id'] for x in train_captions])), key=lambda x: int(re.split('(\d+)',x)[-2]))
test_videos = sorted(list(set([x['video_id'] for x in test_captions])), key=lambda x: int(re.split('(\d+)',x)[-2]))

for caption in train_videos:
	print caption
# for caption in sorted_captions:
#	print caption['video_id'] +","+ caption['caption'].encode('ascii','ignore')

