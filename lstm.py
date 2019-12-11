import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pickle
import numpy as np
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 160
input_size = 28
hidden_size = 128
num_layers = 5
num_classes = 2
batch_size = 100
num_epochs = 3000
learning_rate = 1e-3

vocab = pickle.load(open('save.pickle','rb'))[0]
data = pickle.load(open('data_saved.pickle','rb'))[0]


weights_matrix = torch.from_numpy(vocab['wordsVectors']).to(device)
x_train = torch.from_numpy(data['train_data'].astype('long')).to(device)
y_train = torch.from_numpy((data['train_labels']).astype('float32')).to(device)
x_val = torch.from_numpy(data['validation_data'].astype('long')).to(device)
y_val = torch.from_numpy((data['validation_labels']).astype('float32')).to(device)
x_test = torch.from_numpy(data['test_data'].astype('long')).to(device)
y_test = torch.from_numpy((data['test_labels']).astype('float32')).to(device)

print(y_train.size())

def create_emb_layer(weights_matrix, non_trainable=False):
	num_embeddings, embedding_dim = weights_matrix.size()
	emb_layer = nn.Embedding(num_embeddings, embedding_dim)
	emb_layer.load_state_dict({'weight': weights_matrix})
	if non_trainable:
		emb_layer.weight.requires_grad = False

	return emb_layer, num_embeddings, embedding_dim

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
	def __init__(self, weights_matrix, hidden_size, num_layers, num_classes):
		super(RNN, self).__init__()
		self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, num_classes)
		self.sigmoid = nn.Sigmoid()
	
	def forward(self, x):
		x = self.embedding(x)
		# Set initial hidden and cell states 
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
		c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
	
		# Forward propagate LSTM
		
		out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
	
		# Decode the hidden state of the last time step
		out = self.fc(out[:, -1, :])
		out = self.sigmoid(out)
		return out

model = RNN(weights_matrix, hidden_size, num_layers, num_classes).to(device)


# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
from sklearn.metrics import accuracy_score, f1_score
def get_measurement(y_true, y_pred, type = 'acc'):
	if (device != 'cpu'):
		y_true = y_true.cpu().clone()
		y_pred = y_pred.cpu().clone()
	y_pred = y_pred.numpy()
	y_true = y_true.numpy()
	if (type == 'acc'):
		return accuracy_score(np.argmax(y_true,1), y_pred)
	elif (type == 'f1'): return f1_score(y_true, y_pred)

def validation(model, x_val, y_val, pre_score, path, measure = 'acc', type = 'max' ):
	model.eval()
	outs = model(x_val)
	_, predicted = torch.max(outs.data, 1)
	score = get_measurement(y_val, predicted, measure)
	#print(score)
	if (type == 'max'):
		if (score > pre_score):
			pre_score = score
			torch.save(model.state_dict(), path)
			print('Validation ' + measure + ' increase to ' + str(score))
	else:
		if (score < pre_score):
			pre_score = score
			torch.save(model.state_dict(), path)
			print('Validation ' + measure + ' decrease to ' + str(score))
	return pre_score

def test(model, x_test, y_test, measure): 
	model.eval()
	outs = model(x_test)
	_, predicted = torch.max(outs.data, 1)
	score = get_measurement(y_test, predicted, measure)
	return score, predicted

def print_predict_label(logits):
	return logits

acc = 0
path = 'model_lstm.ckpt'

for epoch in range(num_epochs):
	model.train()
	# Forward pass
	outputs = model(x_train)
	loss = criterion(outputs, y_train)
	
	# Backward and optimize
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	
	#validation
	acc = validation(model, x_val, y_val, acc, path, 'acc', 'max')
	if (epoch % 100 == 0):
		test_acc, test_predict = test(model, x_test, y_test, 'acc')
		print ('Epoch [{}/{}]], Loss: {:.4f}, Test_Acc: {:.2f}'  
				.format(epoch+1, num_epochs, loss.item(), test))

model.load_state_dict(torch.load(path))
test_acc, test_predict = test(model, x_test, y_test, 'acc')
print('Test Acc: {:.2f}'.format(test_acc))
y_test_print = y_test.cpu().clone().numpy()
test_predict_print = test_predict.cpu().clone().numpy()
print('Gold Labels: \t \t', np.argmax(y_test_print[:20], axis = 1))
print('Predicted Labels: \t', test_predict_print[:20])