#encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm_notebook
from collections import defaultdict


INPUT_SIZE = (1, 28, 28)
transforms = torchvision.transforms.Compose([
	torchvision.transforms.RandomCrop(INPUT_SIZE[1:], padding=2),
	torchvision.transforms.ToTensor(),
])


trn_dataset = torchvision.datasets.MNIST('D:/data/MNIST', train=True, download=True, transform=transforms)
tst_dataset = torchvision.datasets.MNIST('D:/data/MNIST', train=False, download=True, transform=transforms)
#print('Images for training: %d' % len(trn_dataset))
#print('Images for testing: %d' % len(tst_dataset))

BATCH_SIZE = 3 # Batch size not specified in the paper
trn_loader = torch.utils.data.DataLoader(trn_dataset, BATCH_SIZE, shuffle=True)
tst_loader = torch.utils.data.DataLoader(tst_dataset, BATCH_SIZE, shuffle=False)


class Conv1(torch.nn.Module):
	def __init__(self, in_channels, out_channels=256, kernel_size=9):
		super(Conv1, self).__init__()
		self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
		self.activation = torch.nn.ReLU()
        
	def forward(self, x):
		x = self.conv(x)
		x = self.activation(x)
		return x

class PrimaryCapsules(torch.nn.Module):
	def __init__(self, input_shape=(256, 20, 20), capsule_dim=8,
			out_channels=32, kernel_size=9, stride=2):
		super(PrimaryCapsules, self).__init__()
		self.input_shape = input_shape
		self.capsule_dim = capsule_dim
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.in_channels = self.input_shape[0]

		self.conv = torch.nn.Conv2d(
			self.in_channels,
			self.out_channels * self.capsule_dim,
			self.kernel_size,
			self.stride
			)

	def forward(self, x):
		x = self.conv(x)
		x = x.permute(0, 2, 3, 1).contiguous()
		x = x.view(-1, x.size()[1], x.size()[2], self.out_channels, self.capsule_dim)
		#print(x.shape)
		return x


class Routing(torch.nn.Module):
	def __init__(self, caps_dim_before=8, caps_dim_after=16,
						n_capsules_before=(6 * 6 * 32), n_capsules_after=10):
		super(Routing, self).__init__()
		self.n_capsules_before = n_capsules_before
		self.n_capsules_after = n_capsules_after
		self.caps_dim_before = caps_dim_before
		self.caps_dim_after = caps_dim_after

		# Parameter initialization not specified in the paper
		n_in = self.n_capsules_before * self.caps_dim_before
		variance = 2 / (n_in)
		std = np.sqrt(variance)
		self.W = torch.nn.Parameter(
				torch.randn(
					self.n_capsules_before,
					self.n_capsules_after,
					self.caps_dim_after,
					self.caps_dim_before) * std,
					requires_grad=True)	#self.n_capsules_before =6 * 6 * 32，需要根据输入输入进行调整， self.n_capsules_after=10,需要调整，caps_dim_before=16需要调整
    
	# Equation (1)
	@staticmethod
	def squash(s):
		s_norm = torch.norm(s, p=2, dim=-1, keepdim=True)
		s_norm2 = torch.pow(s_norm, 2)
		v = (s_norm2 / (1.0 + s_norm2)) * (s / s_norm)
		return v
    
	# Equation (2)
	def affine(self, x):
		#print(x.unsqueeze(2).expand(-1, -1, 10, -1).unsqueeze(-1).shape)
		x = self.W @ x.unsqueeze(2).expand(-1, -1, 10, -1).unsqueeze(-1)	#expand(-1,-1,10,-1)中的10需要调整
		return x.squeeze()

	# Equation (3)
	@staticmethod
	def softmax(x, dim=-1):
		exp = torch.exp(x)
		return exp / torch.sum(exp, dim, keepdim=True)

	# Procedure 1 - Routing algorithm.
	def routing(self, u, r, l):#u为输入，r为迭代次数，l = （1152，10）
		b = Variable(torch.zeros(u.size()[0], l[0], l[1]), requires_grad=False).cuda() # torch.Size([?, 1152, 10])
		#print(b.shape)
		for iteration in range(r):
			c = Routing.softmax(b) # torch.Size([?, 1152, 10])
			s = (c.unsqueeze(-1).expand(-1, -1, -1, u.size()[-1]) * u).sum(1) # torch.Size([?, 1152, 16])
			v = Routing.squash(s) # torch.Size([?, 10, 16])
			b += (u * v.unsqueeze(1).expand(-1, l[0], -1, -1)).sum(-1)
		return v
    
	def forward(self, x, n_routing_iter):
		#print(x.shape)
		x = x.view((-1, self.n_capsules_before, self.caps_dim_before))
		#print(x.shape)
		x = self.affine(x) # torch.Size([?, 1152, 10, 16])	#求得权重和输入相乘
		
		x = self.routing(x, n_routing_iter, (self.n_capsules_before, self.n_capsules_after))
		return x


class Norm(torch.nn.Module):
	def __init__(self):
		super(Norm, self).__init__()

	def forward(self, x):
		x = torch.norm(x, p=2, dim=-1)
		return x


class Decoder(torch.nn.Module):
	def __init__(self, in_features, out_features, output_size=INPUT_SIZE):
		super(Decoder, self).__init__()
		self.decoder = self.assemble_decoder(in_features, out_features)
		self.output_size = output_size

	def assemble_decoder(self, in_features, out_features):
		HIDDEN_LAYER_FEATURES = [512, 1024]
		return torch.nn.Sequential(
			torch.nn.Linear(in_features, HIDDEN_LAYER_FEATURES[0]),
			torch.nn.ReLU(),
			torch.nn.Linear(HIDDEN_LAYER_FEATURES[0], HIDDEN_LAYER_FEATURES[1]),
			torch.nn.ReLU(),
			torch.nn.Linear(HIDDEN_LAYER_FEATURES[1], out_features),
			torch.nn.Sigmoid(),
		)
    
	def forward(self, x, y):
		#print(np.arange(0, x.size()[0]))
		#print(y.cpu().data.numpy())
		x = x[np.arange(0, x.size()[0]), y.cpu().data.numpy(), :].cuda()
		#print(x.shape)
		x = self.decoder(x)
		#print(x.shape)
		x = x.view(*((-1,) + self.output_size))
		#print(x.shape)
		return x

class CapsNet(torch.nn.Module):
	def __init__(self, input_shape=INPUT_SIZE, n_routing_iter=3, use_reconstruction=True):
		super(CapsNet, self).__init__()
		assert len(input_shape) == 3

		self.input_shape = input_shape
		self.n_routing_iter = n_routing_iter
		self.use_reconstruction = use_reconstruction

		self.conv1 = Conv1(input_shape[0], 256, 9)
		self.primary_capsules = PrimaryCapsules(
			input_shape=(256, 20, 20),
			capsule_dim=8,
			out_channels=32,
			kernel_size=9,
			stride=2
			)
		self.routing = Routing(
			caps_dim_before=8,
			caps_dim_after=16,
			n_capsules_before=6 * 6 * 32,
			n_capsules_after=10
			)
		self.norm = Norm()
        
		if (self.use_reconstruction):
			self.decoder = Decoder(16, int(np.prod(input_shape)))
    
	def n_parameters(self):
		return np.sum([np.prod(x.size()) for x in self.parameters()])
    
	def forward(self, x, y=None):
		#print(x.shape)
		conv1 = self.conv1(x)
		#print(conv1.shape)
		primary_capsules = self.primary_capsules(conv1)
		#print(primary_capsules.shape)
		digit_caps = self.routing(primary_capsules, self.n_routing_iter)
		#print(digit_caps.shape)
		scores = self.norm(digit_caps)
		#print(scores.shape)

		if (self.use_reconstruction and y is not None):
			reconstruction = self.decoder(digit_caps, y).view((-1,) + self.input_shape)
			return scores, reconstruction

		return scores



def to_categorical(y, num_classes):
	""" 1-hot encodes a tensor """
	new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]	#类似于one-hot编码
	#print(new_y)
	#print(torch.eye(num_classes).shape)
	if (y.is_cuda):
		return new_y.cuda()
	return new_y



class MarginLoss(torch.nn.Module):
	def __init__(self, m_pos=0.9, m_neg=0.1, lamb=0.5):	#为什么按照0.9 或者0.1来取
		super(MarginLoss, self).__init__()
		self.m_pos = m_pos
		self.m_neg = m_neg
		self.lamb = lamb
    
    # Equation (4)
	def forward(self, scores, y):	#scores是一个向量
		y = Variable(to_categorical(y, 10))

		Tc = y.float()
		#print(Tc.shape)
		loss_pos = torch.pow(torch.clamp(self.m_pos - scores, min=0), 2)
		#print(loss_pos.shape)
		loss_neg = torch.pow(torch.clamp(scores - self.m_neg, min=0), 2)
		loss = Tc * loss_pos + self.lamb * (1 - Tc) * loss_neg
		loss = loss.sum(-1)
		return loss.mean()


class SumSquaredDifferencesLoss(torch.nn.Module):
	def __init__(self):
		super(SumSquaredDifferencesLoss, self).__init__()

	def forward(self, x_reconstruction, x):
		loss = torch.pow(x - x_reconstruction, 2).sum(-1).sum(-1)
		return loss.mean()

#criterion(x, y, x_reconstruction, y_pred.cuda())

class CapsNetLoss(torch.nn.Module):
	def __init__(self, reconstruction_loss_scale=0.0005):
		super(CapsNetLoss, self).__init__()
		self.digit_existance_criterion = MarginLoss()	#只和这个损失函数有关系
		self.digit_reconstruction_criterion = SumSquaredDifferencesLoss()
		self.reconstruction_loss_scale = reconstruction_loss_scale
    
	def forward(self, x, y, x_reconstruction, scores):
		margin_loss = self.digit_existance_criterion(y_pred.cuda(), y)	#这个损失是胶囊网络的损失
		reconstruction_loss = self.reconstruction_loss_scale *\
								self.digit_reconstruction_criterion(x_reconstruction, x)	#decoder的损失
		loss = margin_loss + reconstruction_loss	#总的损失
		return loss, margin_loss, reconstruction_loss



#exponential_decay(optimizer, LEARNING_RATE, global_epoch, 1, 0.90)

def exponential_decay(optimizer, learning_rate, global_step, decay_steps, decay_rate, staircase=False):
	if (staircase):
		decayed_learning_rate = learning_rate * np.power(decay_rate, global_step // decay_steps)
	else:
		decayed_learning_rate = learning_rate * np.power(decay_rate, global_step / decay_steps)
        
	for param_group in optimizer.param_groups:
		#print(param_group)
		param_group['lr'] = decayed_learning_rate

	return optimizer


LEARNING_RATE = 0.001



#Training
def save_checkpoint(epoch, train_accuracy, test_accuracy, model, optimizer, path=None):
	if (path is None):
		path = 'checkpoint-%f-%04d.pth' % (test_accuracy, epoch)
	state = {
		'epoch': epoch,
		'train_accuracy': train_accuracy,
		'test_accuracy': test_accuracy,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
	}
	torch.save(state, path)

def show_example(model, x, y, x_reconstruction, y_pred):
	
	x = x.squeeze().cpu().data.numpy()
	y = y.cpu().data.numpy()[0]
	print(y)
	x_reconstruction = x_reconstruction.squeeze().cpu().data.numpy()
	_, y_pred = torch.max(y_pred, -1)
	y_pred = y_pred.cpu().data.numpy()[0]

	fig, ax = plt.subplots(1, 2)
	ax[0].imshow(x, cmap='Greys')
	ax[0].set_title('Input: %d' % y)
	ax[1].imshow(x_reconstruction, cmap='Greys')
	ax[1].set_title('Output: %d' % y_pred)
	plt.show()

def test(model, loader):
	metrics = defaultdict(lambda:list())
	for batch_id, (x, y) in tqdm_notebook(enumerate(loader), total=len(loader)):
		x = Variable(x).float().cuda()
		y = Variable(y).cuda()
		y_pred, x_reconstruction = model(x, y)
		_, y_pred = torch.max(y_pred, -1)	#进行测试
		metrics['accuracy'].append((y_pred == y).cpu().data.numpy())
	metrics['accuracy'] = np.concatenate(metrics['accuracy']).mean()
	return metrics
    
    
global_epoch = 0
global_step = 0
best_tst_accuracy = 0.0

history = defaultdict(lambda:list())

COMPUTE_TRN_METRICS = False


n_epochs = 15 # Number of epochs not specified in the paper

model = CapsNet().cuda()	#模型
criterion = CapsNetLoss()	#损失函数

optimizer = torch.optim.Adam(
	model.parameters(),
	lr=LEARNING_RATE,
	betas=(0.9, 0.999),
	eps=1e-08
)

for epoch in range(n_epochs):
    
	for batch_id, (x, y) in tqdm_notebook(enumerate(trn_loader), total=len(trn_loader)):
		#print(batch_id)
		#print(x.shape)
		#print(y)
		
		optimizer = exponential_decay(optimizer, LEARNING_RATE, global_epoch, 1, 0.90) # Configurations not specified in the paper	调节学习率

		x = Variable(x).float().cuda()
		y = Variable(y).cuda()

		y_pred, x_reconstruction = model(x, y)
		#print(y_pred.shape)
		#print(x_reconstruction.shape)
		loss, margin_loss, reconstruction_loss = criterion(x, y, x_reconstruction, y_pred.cuda())
		history['margin_loss'].append(margin_loss.cpu().data.numpy()[0])
		history['reconstruction_loss'].append(reconstruction_loss.cpu().data.numpy()[0])
		history['loss'].append(loss.cpu().data.numpy()[0])

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		global_step += 1

	trn_metrics = test(model, trn_loader) if COMPUTE_TRN_METRICS else None
	tst_metrics = test(model, tst_loader)

	print('Margin Loss: %f' % history['margin_loss'][-1])
	print('Reconstruction Loss: %f' % history['reconstruction_loss'][-1])
	print('Loss: %f' % history['loss'][-1])
	print('Train Accuracy: %f' % (trn_metrics['accuracy'] if COMPUTE_TRN_METRICS else 0.0))
	print('Test Accuracy: %f' % tst_metrics['accuracy'])

	print('Example:')
	idx = np.random.randint(0, len(x))
	show_example(model, x[idx], y[idx], x_reconstruction[idx], y_pred[idx])

	if (tst_metrics['accuracy'] >= best_tst_accuracy):
		best_tst_accuracy = tst_metrics['accuracy']
		save_checkpoint(
			global_epoch + 1,
			trn_metrics['accuracy'] if COMPUTE_TRN_METRICS else 0.0,
			tst_metrics['accuracy'],
			model,
			optimizer
        )
	global_epoch += 1
