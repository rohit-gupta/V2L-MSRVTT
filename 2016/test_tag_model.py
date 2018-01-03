from keras.utils import Sequence
import json
import re
import pickle
import argparse
from glob import glob

import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

def preprocess_image(img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x


NUM_FRAMES = 40
FRAME_RATE = 2 

def load_videos(video_ids, video_folder, video_fps_dict):
	frames = []
	for video_name in video_ids:
		fps = video_fps_dict[video_name]
		frame_files = sorted(glob(video_folder + video_name +"/*.jpg"))
		num_frames = len(frame_files)
		gap = int(round(fps/FRAME_RATE)) # If FPS = 30, FRAME_RATE = 3, Frames at ID 0,10,20,30 ... are sampled
		frame_data = []
		for idx,frame_file in enumerate(frame_files):
			if len(frame_data) >= NUM_FRAMES:
				break
			if idx%gap == 0:
				frame_data.append(preprocess_image(frame_file)[0])
		actual_frame_length = len(frame_data)
		# If Video is shorter than 8 seconds repeat the short video
		if len(frame_data) < NUM_FRAMES: 
			if NUM_FRAMES/len(frame_data) > 1: # Video is < 1/2 of 10 Seconds
				num_repeats = NUM_FRAMES/len(frame_data) - 1
				for _ in range(num_repeats):
					for itr in range(len(frame_data[:actual_frame_length])):
						frame_data.append(frame_data[:actual_frame_length][itr])
			dup_frame_length = len(frame_data)
			if NUM_FRAMES/len(frame_data) == 1 and NUM_FRAMES > len(frame_data): # Video is slightly smaller than 10 Seconds
				for itr in range(0, NUM_FRAMES -len(frame_data)):
					frame_data.append(frame_data[itr])
		if len(frame_data) != NUM_FRAMES:
			print actual_frame_length, num_repeats, dup_frame_length, len(frame_data)
			raise Exception, 'Incorrect number of frames sampled'
		frame_data = np.array(frame_data)
		frames.append(frame_data)
	return np.array(frames)

def load_tags(video_ids,tag_dict):
	tags_arr = []
	for video in video_ids:
		tags_arr.append(tag_dict[video])
	return np.array(tags_arr)

# Here, `videos` is list of path to the videos and `tags` are the associated classes.
class MSRVTTSequence(Sequence):
	def __init__(self, captions_dict, video_folder,fps_dict, tag_dict, batch_size):
		self.batch_size = batch_size
		self.video_folder = video_folder
		self.fps_dict = fps_dict
		self.tag_dict = tag_dict
		self.videos = sorted(list(set([x['video_id'] for x in captions_dict])), key=lambda x: int(re.split('(\d+)',x)[-2]))

	def __len__(self):
		return len(self.videos) // self.batch_size

	def __getitem__(self,idx):
		batch_videos = self.videos[idx*self.batch_size:(idx+1)*self.batch_size]
		#return load_videos(batch_videos,self.video_folder,self.fps_dict),load_tags(batch_videos, self.tag_dict)
		return load_videos(batch_videos,self.video_folder,self.fps_dict)

	def on_epoch_end(self):
		pass





parser = argparse.ArgumentParser()
parser.add_argument('-t', action='store', dest='tag_type', help='(action/entity/attribute) Type of Tags to train model on')
parser.add_argument('-l', action='store', dest='lstm_size', type=int, help='LSTM Hidden State Size')
# parser.add_argument('-d', action='store', dest='gpu', help='GPU to use')
#parser.add_argument('-s', action='store_true', default=False, dest='save_predictions', help='Save predicted tags')

results = parser.parse_args()




# Load Video Meta-data
fname = "../train-video/video_fps.csv"
with open(fname) as f:
    content = f.readlines()

video_fps_list = [(x.strip()).split(",")[:2] for x in content]

video_fps = {}
for x in video_fps_list:
	y = x[1].split("/")
	video_fps[x[0]] = float(y[0])/float(y[1])

# Load Captions Data
annotations_file = "test_videodatainfo.json"
videos_folder = "../train-video/frames/"

with open(annotations_file) as data_file:
			data = json.load(data_file)

captions = data['sentences']

test_captions = sorted(captions, key=lambda k:  int(re.split('(\d+)',k['video_id'])[-2]))

tags = pickle.load(open("tags/" + results.tag_type + ".pickle","rb"))
NUM_TAGS = tags[tags.keys()[0]].shape[0]

videos = sorted(list(set([x['video_id'] for x in captions])), key=lambda x: int(re.split('(\d+)',x)[-2]))

actual_tags = np.array([tags[video] for video in videos])
print actual_tags.shape

# Create generator
test_generator = MSRVTTSequence(test_captions, video_folder=videos_folder, fps_dict=video_fps, tag_dict=tags, batch_size=2)

train_annotations_file = "train_val_videodatainfo.json"
with open(train_annotations_file) as data_file:
                        data = json.load(data_file)

train_captions = data['sentences']
train_captions = sorted(train_captions, key=lambda k:  int(re.split('(\d+)',k['video_id'])[-2]))


train_videos = sorted(list(set([x['video_id'] for x in train_captions])), key=lambda x: int(re.split('(\d+)',x)[-2]))

# train_generator = MSRVTTSequence(train_captions, video_folder=videos_folder, fps_dict=video_fps, tag_dict=tags, batch_size=2)
# NUM_TRAIN_STEPS = len(train_generator)
NUM_STEPS = len(test_generator) 

from keras.applications.resnet50 import ResNet50
from keras.layers import TimeDistributed, Bidirectional
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau

from keras import backend as K
K.set_learning_phase(1)

# Define Model
video_input = Input(shape=(NUM_FRAMES, 224, 224, 3))
convnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
for layer in convnet_model.layers:
    layer.trainable = False

encoded_frame_sequence = TimeDistributed(convnet_model)(video_input)
#encoded_video = Bidirectional(LSTM(results.lstm_size,implementation=1,dropout=0.5))(encoded_frame_sequence)
encoded_video = LSTM(results.lstm_size,implementation=2,dropout=0.2)(encoded_frame_sequence)
#encoded_video = LSTM(results.lstm_size,implementation=2,dropout=0.2)(encoded_frame_sequence)
# doubly_encoded_video = LSTM(results.lstm_size,implementation=2,dropout=0.2)(encoded_video)
# triply_encoded_video = LSTM(results.lstm_size,implementation=2,dropout=0.2)(doubly_encoded_video)
output = Dense(NUM_TAGS, activation='sigmoid')(encoded_video)



tag_model = Model(inputs=video_input, outputs=output)
tag_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
tag_model.summary()

# tag_model.load_weights("models/"+results.tag_type+"_double_layer_40frames_tag_model2.h5")
tag_model.load_weights("models/"+results.tag_type+"_single_layer_40frames_2fps_tag_model.h5")

pred_tags = tag_model.predict_generator(test_generator, NUM_STEPS,verbose=1)
# pred_train_tags = tag_model.predict_generator(train_generator, NUM_TRAIN_STEPS,verbose=1)

actual_tags = actual_tags[:pred_tags.shape[0],:]

pred_tags_dict = {}

# for idx,video_name in enumerate(train_videos):
#	pred_tags_dict[video_name] = pred_train_tags[idx]

#for idx,video_name in enumerate(videos):
#	pred_tags_dict[video_name] = pred_tags[idx]

# pickle.dump(pred_tags_dict,open("tags/lstm_predicted_"+results.tag_type+".pickle","wb"))


from sklearn.metrics import hamming_loss
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt


# Calculate Micro Averaged Precision
# For each class
precision = dict()
recall = dict()
average_precision = dict()
# for i in range(NUM_TAGS):
#     precision[i], recall[i], _ = precision_recall_curve(test_tags[:, i], test_preds[:, i])
#     average_precision[i] = average_precision_score(test_tags[:, i], test_preds[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(actual_tags.ravel(), pred_tags.ravel())
average_precision["micro"] = average_precision_score(actual_tags, pred_tags, average="micro")
print 'Average precision score, micro-averaged over all classes:', average_precision["micro"]

# Plot uAP v Recall curve
plt.switch_backend("agg")
plt.figure()
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2, where='post')
plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2, color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Average precision score, micro-averaged over all classes: AUC={0:0.2f}'.format(average_precision["micro"]))
plt.savefig('PR_Curve_'+results.tag_type+'_single_layer_40frames_2fps_tag_model.png')


