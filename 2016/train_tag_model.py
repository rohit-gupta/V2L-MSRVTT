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
		return load_videos(batch_videos,self.video_folder,self.fps_dict),load_tags(batch_videos, self.tag_dict)

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
annotations_file = "train_val_videodatainfo.json"
videos_folder = "../train-video/frames/"

with open(annotations_file) as data_file:
			data = json.load(data_file)

captions = data['sentences']

sorted_captions = sorted(captions, key=lambda k:  int(re.split('(\d+)',k['video_id'])[-2]))

# Video0 to Video6512 are Training Videos
train_captions = sorted_captions[:6513*20]
# Video6513 to Video7009 are Validation Videos
validation_captions =sorted_captions[6513*20:]

tags = pickle.load(open("tags/" + results.tag_type + ".pickle","rb"))
NUM_TAGS = tags[tags.keys()[0]].shape[0]

# Create generator
train_generator = MSRVTTSequence(train_captions, video_folder=videos_folder, fps_dict=video_fps, tag_dict=tags, batch_size=16)
validation_generator = MSRVTTSequence(validation_captions, video_folder=videos_folder, fps_dict=video_fps, tag_dict=tags, batch_size=16)


from keras.applications.resnet50 import ResNet50
from keras.layers import TimeDistributed, Bidirectional
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from keras import backend as K
K.set_learning_phase(1)

# Define Model
video_input = Input(shape=(NUM_FRAMES, 224, 224, 3))
convnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
for layer in convnet_model.layers:
    layer.trainable = False

encoded_frame_sequence = TimeDistributed(convnet_model)(video_input)
encoded_video = LSTM(results.lstm_size,implementation=2,dropout=0.5)(encoded_frame_sequence)
#encoded_video = LSTM(results.lstm_size,implementation=2,dropout=0.2, return_sequences=True)(encoded_frame_sequence)
#doubly_encoded_video = LSTM(results.lstm_size,implementation=2,dropout=0.2)(encoded_video)
# triply_encoded_video = LSTM(results.lstm_size,implementation=2,dropout=0.2)(doubly_encoded_video)
output = Dense(NUM_TAGS, activation='sigmoid')(encoded_video)

tag_model = Model(inputs=video_input, outputs=output)
tag_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
tag_model.summary()

#tag_model.load_weights('models/'+results.tag_type+'_double_layer_80frames_tag_model.h5')

# Train Model

csv_logger = CSVLogger('logs/'+results.tag_type+'_single_layer_40frames_2fps_tag_model.log')
checkpointer = ModelCheckpoint(filepath='models/'+results.tag_type+'_single_layer_40frames_2fps_tag_model.h5', verbose=1, save_best_only=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001)
# early_stop = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.000001)

#tag_model.fit(augmented_train_frames, augmented_train_tags, epochs=10, batch_size=16, validation_split=0.2, callbacks=[csv_logger, checkpointer, reduce_lr])

tag_model.fit_generator(train_generator, steps_per_epoch=len(train_generator), epochs=20, verbose=1, callbacks=[csv_logger,checkpointer],validation_data=validation_generator,validation_steps=len(validation_generator), max_queue_size=10, workers=1, use_multiprocessing=True, shuffle=True)
