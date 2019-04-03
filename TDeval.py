import scipy.io.wavfile as wav

import argparse

import tensorflow as tf 
import numpy as np
import os
import sys
from sklearn.metrics import roc_curve, auc
sys.path.append("DeepSpeech")

tf.load_op_library = lambda x: x
tmp = os.path.exists
os.path.exists = lambda x: True
import DeepSpeech
os.path.exists = tmp

from util.text import ctc_label_dense_to_sparse
from tf_logits3 import get_logits
from tqdm import tqdm
toks = " abcdefghijklmnopqrstuvwxyz'-"

DeepSpeech.TrainingCoordinator.__init__ = lambda x: None

DeepSpeech.TrainingCoordinator.start = lambda x: None
import loss
# Use the same decode framework as carlini used :) 
class Decoder:
	def __init__(self, sess, max_audio_len):
		self.sess = sess
		self.max_audio_len = max_audio_len
		self.original = original = tf.Variable(np.zeros((1, max_audio_len), dtype=np.float32), name='qq_original')
		self.lengths = lengths = tf.Variable(np.zeros(1, dtype=np.int32), name='qq_lengths')

		with tf.variable_scope("", reuse=tf.AUTO_REUSE):
			logits, features = get_logits(original, lengths)

		self.logits = logits
		self.features = features
		saver = tf.train.Saver([x for x in tf.global_variables() if 'qq' not in x.name])
		saver.restore(sess, "models/session_dump")
		self.decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=1000)

	def transcribe(self, audio, lengths):
		sess = self.sess
		sess.run(self.original.assign(np.array(audio)))
		sess.run(self.lengths.assign((np.array(lengths)-1)//320))
		out, logits = sess.run((self.decoded, self.logits))
		chars = out[0].values
		res = np.zeros(out[0].dense_shape)+len(toks)-1		
		for ii in range(len(out[0].values)):
			x,y = out[0].indices[ii]
			res[x,y] = out[0].values[ii]
		res = ["".join(toks[int(x)] for x in y).replace("-","") for y in res]
		return res[0]

def decode(audio, num):
	global maxlen
	global D
	global ref
	audios = [list(audio)]
	lengths = [int(maxlen * num)]
	audios = np.array(audios)
	res = D.transcribe(audios, lengths)
	return res

num_samples = 50
y_test = np.zeros(num_samples * 2)
roc_auc = np.zeros(3)
TD = np.zeros((3, num_samples * 2), dtype = np.float32)
count = 0
if __name__ == '__main__':
	sess = tf.Session()
	parser = argparse.ArgumentParser(description = None)
	parser.add_argument('--cut', type = float, required = True)
	args = parser.parse_args()


	ratio = args.cut
	pbar = tqdm(range(num_samples), unit='steps', ascii = True)
	for epoch in pbar:
		x, y = wav.read("librisample" + str(epoch) + ".wav")
		z, w = wav.read("librifinal" + str(epoch) + ".wav")
		maxlen = len(y)

		#ratio = np.random.random_sample() * 0.6 + 0.2 
		#ratio = (numcut) * 1.0 / (numcut - 1)
		D = Decoder(sess, maxlen)
		stry = decode(y, 1)
		strw = decode(w, 1)
		halfy = decode(y, ratio)
		halfw = decode(w, ratio)

		#print ("Origin: " + stry)
		#print ("Half of Origin: " + halfy)
		s1 = loss.newWER(stry, halfy)
		s2 = loss.newCER(stry, halfy)
		s3 = loss.lcp(stry, halfy)

		y_test[count] = 0
		TD[0][count] = float(s1)
		TD[1][count] = float(s2)
		TD[2][count] = float(s3)

		count += 1
		#print ("WER: " + str(s1) + " CER: " + str(s2) + " LCP: " + str(s3))
		#print ("Adv: " + strw)
		#print ("Half of Adv: " + halfw)
		s1 = loss.newWER(strw, halfw)
		s2 = loss.newCER(strw, halfw)
		s3 = loss.lcp(strw, halfw)
		
		y_test[count] = 1
		TD[0][count] = float(s1)
		TD[1][count] = float(s2)
		TD[2][count] = float(s3)
		count += 1
		#print ("WER: " + str(s1) + " CER: " + str(s2) + " LCP: " + str(s3))


	for i in range(3):
		if (i == 2):
			y_test = 1 - y_test
		fpr, tpr, threshold = roc_curve(y_test, TD[i])
		roc_auc[i] = auc(fpr, tpr)

	print ("WER: " + str(roc_auc[0]) + " CER: " + str(roc_auc[1]) + " LCP: " + str(roc_auc[2]))



