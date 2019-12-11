import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pickle
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 160
input_size = 28
hidden_size = 128
num_layers = 5
num_classes = 2
batch_size = 32
num_epochs = 3000
learning_rate = 1e-3

#vocab = pickle.load(open('save.pickle','rb'))[0]
data = pickle.load(open('data_saved.pickle','rb'))[0]


#preprocess data
def get_bert_input(x, y, MAX_LEN = 160):
	x = ["[CLS] " + sentence + " [SEP]" for sentence in x]
	
	
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
	tokenized_texts = [tokenizer.tokenize(sent) for sent in x]
	#print ("Tokenize the first sentence:")
	#print (tokenized_texts[0])
	
	input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
	
	# Create attention masks
	attention_masks = []

	# Create a mask of 1s for each token followed by 0s for padding
	for seq in input_ids:
		seq_mask = [float(i>0) for i in seq]
		attention_masks.append(seq_mask)
	#print(attention_masks[0])
	
	input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
	attention_masks = pad_sequences(attention_masks, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
	#print(np.array(attention_masks).shape)
	return input_ids, attention_masks, y
	

train_inputs, train_masks, train_labels =  get_bert_input(data['train_content'], data['train_labels'])
validation_inputs, validation_masks, validation_labels =  get_bert_input(data['validation_content'], data['validation_labels'])
test_inputs, test_masks, test_labels =  get_bert_input(data['test_content'], data['test_labels'])

# Convert all of our data into torch tensors, the required datatype for our model
train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(np.argmax(train_labels,1))
train_masks = torch.tensor(train_masks)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(np.argmax(validation_labels,1))
validation_masks = torch.tensor(validation_masks)
test_inputs = torch.tensor(test_inputs)
test_labels = torch.tensor(np.argmax(test_labels,1))
test_masks = torch.tensor(test_masks)


# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
# with an iterator the entire dataset does not need to be loaded into memory

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

from pytorch_pretrained_bert import modeling

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
# config = modeling.BertConfig(
#   attention_probs_dropout_prob = 0.1,
#   hidden_act= "gelu",
#   hidden_dropout_prob= 0.1,
#   hidden_size= 768,
#   initializer_range= 0.02,
#   intermediate_size=3072,
#   max_position_embeddings= 512,
#   num_attention_heads= 12,
#   num_hidden_layers= 12,
#   type_vocab_size= 2,
#   vocab_size= 32006
# )
# model = BertForSequenceClassification(config)
#model.bert.load_state_dict(torch.load('drive/My Drive/data_colab/bert-japanese/pytorch_model.bin'), strict=False)
model.cuda()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

# This variable contains all of the hyperparemeter information our training loop needs
optimizer = BertAdam(optimizer_grouped_parameters, lr=2e-5, warmup=.1)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
	pred_flat = np.argmax(preds, axis=1).flatten()
	labels_flat = labels.flatten()
	return np.sum(pred_flat == labels_flat) / len(labels_flat)

def print_predict_label(logits):
	return np.argmax(logits, axis=1)



def evaluate_model(model, test_dataloader):
	# Put model in evaluation mode
	model.eval()
	
	# Tracking variables 
	predictions , true_labels = [], []

	eval_loss, eval_accuracy = 0, 0
	nb_eval_steps, nb_eval_examples = 0, 0


	# Predict 
	for batch in test_dataloader:
		# Add batch to GPU
		batch = tuple(t.to(device) for t in batch)
		# Unpack the inputs from our dataloader
		b_input_ids, b_input_mask, b_labels = batch
		# Telling the model not to compute or store gradients, saving memory and speeding up prediction
		with torch.no_grad():
			# Forward pass, calculate logit predictions
			logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

		# Move logits and labels to CPU
		logits = logits.detach().cpu().numpy()
		label_ids = b_labels.to('cpu').numpy()
    

		tmp_eval_accuracy = flat_accuracy(logits, label_ids)
	
		eval_accuracy += tmp_eval_accuracy
		nb_eval_steps += 1
	
		# Store predictions and true labels
		predictions.append(logits)
		true_labels.append(label_ids)
	return (eval_accuracy/nb_eval_steps), predictions, true_labels
	
# Store our loss and accuracy for plotting
train_loss_set = []
'''
# Number of training epochs (authors recommend between 2 and 4)
epochs = 15
acc = 0
# trange is a tqdm wrapper around the normal python range
for epoch in range(epochs):
	# Training
	# Set our model to training mode (as opposed to evaluation mode)
	model.train()

	# Tracking variables
	tr_loss = 0
	nb_tr_examples, nb_tr_steps = 0, 0
  
	# Train the data for one epoch
	for step, batch in enumerate(train_dataloader):
		# Add batch to GPU
		batch = tuple(t.to(device) for t in batch)
		# Unpack the inputs from our dataloader
		b_input_ids, b_input_mask, b_labels = batch
		
		# Clear out the gradients (by default they accumulate)
		optimizer.zero_grad()
		# Forward pass
		loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
		train_loss_set.append(loss.item())    
		# Backward pass
		loss.backward()
		# Update parameters and take a step using the computed gradient
		optimizer.step()
		
	
		# Update tracking variables
		tr_loss += loss.item()
		nb_tr_examples += b_input_ids.size(0)
		nb_tr_steps += 1

	print("Train loss: {}".format(tr_loss/nb_tr_steps))
	
		
    # Validation
	eval_accuracy, predict_logits, labels = evaluate_model(model, validation_dataloader)
	test_acc, predict_logits, true_label = evaluate_model(model, test_dataloader)
	if (eval_accuracy > acc):
		acc = eval_accuracy
		torch.save(model.state_dict(), 'bert_amazon_classification_e4_128.bin')
	print("Validation Accuracy: {}".format(eval_accuracy))
	print("Test Accuracy: {}".format(test_acc))


# Save model
#model_to_save = model.module if hasattr(model, 'module') else model
#torch.save(model_to_save.state_dict(), 'bert_amazon_classification_e4_128.bin')
with open('bert_amazon_classification_e4_128_config', 'w') as f:
	f.write(model.config.to_json_string())
'''
#Predict and Evaluation

path = 'bert_amazon_classification_e4_128.bin'
model.load_state_dict(torch.load(path))
#test model
test_acc, predict_logits, true_labels = evaluate_model(model, test_dataloader)
print(test_acc)
print('Gold Labels: \t \t', true_labels[0].flatten())
print('Predicted Labels: \t', print_predict_label(predict_logits[0]))


