from __future__ import print_function
import os
import io
import re
import matplotlib.pyplot as plt
import gensim
#from six.moves import cPickle as pickle
import numpy as np
import scipy.stats as stats
import pickle


truthful_pos = 'op_spam_v1.4/positive_polarity/truthful_from_TripAdvisor/'
truthful_neg = 'op_spam_v1.4/negative_polarity/truthful_from_Web/'
deceptive_pos = 'op_spam_v1.4/positive_polarity/deceptive_from_MTurk/'
deceptive_neg = 'op_spam_v1.4/negative_polarity/deceptive_from_MTurk/'

def get_reviews_link(link_dir):
	reviews_link = []
	for fold in os.listdir(link_dir):
		foldLink = os.path.join(link_dir, fold)
		if os.path.isdir(foldLink):
			for f in os.listdir(foldLink):
				fileLink = os.path.join(foldLink, f)
				reviews_link.append(fileLink)
	return reviews_link

truthful_reviews_link = get_reviews_link(truthful_pos)
truthful_reviews_link += get_reviews_link(truthful_neg)

deceptive_reviews_link = get_reviews_link(deceptive_pos)
deceptive_reviews_link += get_reviews_link(deceptive_neg)

print('Number of truthfuls reviews ', len(truthful_reviews_link))
print('Number of deceptives reviews ', len(deceptive_reviews_link))

def clean_str(string):
	"""
	Tokenization/string cleaning for all datasets except for SST.
	Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	"""
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower()

def handleFile(filePath):
	with open(filePath, "r") as f:
		lines=f.readlines()
		file_voc = []
		file_numWords = 0
		for line in lines:
			cleanedLine = clean_str(line)
			cleanedLine = cleanedLine.strip()
			cleanedLine = cleanedLine.lower()
			words = cleanedLine.split(' ')
			file_numWords = file_numWords + len(words)
			file_voc.extend(words)
	return file_voc, file_numWords


allFilesLinks = truthful_reviews_link + deceptive_reviews_link
vocabulary = []
numWords = []
for fileLink in allFilesLinks:
	file_voc, file_numWords = handleFile(fileLink)
	vocabulary.extend(file_voc)
	numWords.append(file_numWords)

vocabulary = set(vocabulary)
vocabulary = list(vocabulary)

print('The total number of files is ', len(numWords))
print('The total number of words in the files is ', sum(numWords))
print('Vocabulary size is ', len(vocabulary))
print('The average number of words in the files is', sum(numWords)/len(numWords))

'''
w2v_model =  gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
wordsVectors = []
notFoundwords = []
for word in vocabulary:
	try:
		vector = w2v_model[word]
		wordsVectors.append(vector)
	except Exception as e:
		notFoundwords.append(word)
		wordsVectors.append(np.random.uniform(-0.25,0.25,300))  

del w2v_model
wordsVectors = np.asarray(wordsVectors)

print('The number of missing words is ', len(notFoundwords))

"""Save"""
save = 	{
		'wordsVectors': wordsVectors,
		'vocabulary': vocabulary,
		'notFoundwords': notFoundwords
		}
pickle.dump([save], open('save.pickle','wb'), protocol = 2)
'''
MAX_SEQ_LENGTH = 160
def convertFileToIndexArray(filePath):
	doc = np.zeros(MAX_SEQ_LENGTH, dtype='int32')
	content = []
	with open(filePath, "r") as f:
		lines=f.readlines()
		indexCounter = 0
		for line in lines:
			cleanedLine = clean_str(line)
			cleanedLine = cleanedLine.strip()
			cleanedLine = cleanedLine.lower()
			words = cleanedLine.split(' ')
			for word in words:
				doc[indexCounter] = vocabulary.index(word)
				content.append(word)
				indexCounter = indexCounter + 1
				if (indexCounter >= MAX_SEQ_LENGTH):
					break
			if (indexCounter >= MAX_SEQ_LENGTH):
				break
	return doc, " ".join(content)

totalFiles = len(truthful_reviews_link) + len(deceptive_reviews_link)
idsMatrix = np.ndarray(shape=(totalFiles, MAX_SEQ_LENGTH), dtype='int32')
contents = []
labels = np.ndarray(shape=(totalFiles,2), dtype='int32')

counter = 0
for filePath in truthful_reviews_link:
	
	idsMatrix[counter], content = convertFileToIndexArray(filePath)
	counter = counter + 1
	contents.append(content)

for filePath in deceptive_reviews_link:
	idsMatrix[counter], content = convertFileToIndexArray(filePath)
	contents.append(content)
	counter = counter + 1


contents = np.array(contents)
labels[0:len(truthful_reviews_link)] = [1,0]
labels[len(truthful_reviews_link):totalFiles] = [0,1]

print('The shape of the ids matrix is ', idsMatrix.shape)
print('The shape of the labels is ', labels.shape)


"""
Create a training set, a validation set and a test set after mixing the data
80% for the training set
10% for the validation set
10% for the test set
"""
size = idsMatrix.shape[0]
testSize = int(size * 0.1)
shuffledIndex = np.random.permutation(size)
testIndexes = shuffledIndex[0:testSize]
validationIndexes = shuffledIndex[testSize:2*testSize]
trainIndexes = shuffledIndex[2*testSize:size]

test_data = idsMatrix[testIndexes]
test_content = contents[testIndexes]
test_labels = labels[testIndexes]

validation_data = idsMatrix[validationIndexes]
validation_content = contents[validationIndexes]
validation_labels = labels[validationIndexes]

train_data = idsMatrix[trainIndexes]
train_content = contents[trainIndexes]
train_labels = labels[trainIndexes]

print('train data shape ', train_data.shape)
print('train content shape ', train_content.shape)
print('train labels shape ', train_labels.shape)
print('validation data shape ', validation_data.shape)
print('validation labels shape ', validation_labels.shape)
print('test data shape ', test_data.shape)
print('test labels shape ', test_labels.shape)


save =	{
		'train_data': train_data,
		'train_content': train_content,
		'train_labels': train_labels,
		'validation_data': validation_data,
		'validation_content': validation_content,
		'validation_labels': validation_labels,
		'test_data': test_data,
		'test_content': test_content,
		'test_labels': test_labels
		}

pickle.dump([save],open('data_saved.pickle','wb'), protocol = 2)