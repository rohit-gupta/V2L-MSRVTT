import json
import re

with open('test_videodatainfo.json') as data_file:
    test_data = json.load(data_file)

with open('train_val_videodatainfo.json') as data_file:
    train_data = json.load(data_file)

videos = sorted(list(set([x['video_id'] for x in test_data['sentences']+train_data['sentences']])), key=lambda x: int(re.split('(\d+)',x)[-2]))


for video in videos:
	print 'cat <(echo "' + video + ',") <(ffprobe -v 0 -of compact=p=0 -select_streams 0 -select_streams 0 -show_entries stream=r_frame_rate ' + video + '.mp4) <(echo "++++")'
