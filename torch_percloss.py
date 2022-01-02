import torch
import torchvision as vision
from torch.nn.functional import l1_loss as l1_loss

class Perceptual_loss(torch.nn.Module):
	def __init__(self):
		super(Perceptual_loss,self).__init__()# Invoke constructor of the parent class
		# Break into four blocks [0:23]
		# Since we are only in a evaluation mode with no need for gradient descent, requires_grad = False 
		# We change the model mode to EVAL since we don't need pairs of two images fed to the network -> 
		# We instead have parallel input

		# Define nn.Sequential for each chunk and label as blocks1~4
		self.vgg16_block1 = torch.nn.Sequential(vision.models.vgg16(pretrained=True).features[:4].eval())
		self.vgg16_block2 = torch.nn.Sequential(vision.models.vgg16(pretrained=True).features[4:9].eval())
		self.vgg16_block3 = torch.nn.Sequential(vision.models.vgg16(pretrained=True).features[9:16].eval())
		self.vgg16_block4 = torch.nn.Sequential(vision.models.vgg16(pretrained=True).features[16:23].eval())

		#Since we are in evaluation mode, gradient backpropagation is not needed
		for p in self.parameters():
			p.requires_grad = False

	def forward(self,yhat,y,blocks=[0,0,0,0]):
		
		# If number of channels is not 3 -> Force them to have size of 3:
		if yhat.size(dim=1) != 3:
			yhat = yhat.repeat((1,3,1,1))
			y = y.repeat((1,3,1,1))

		y1 = self.vgg16_block1(y)
		y2 = self.vgg16_block2(y1)
		y3 = self.vgg16_block3(y2)
		y4 = self.vgg16_block4(y3)

		yhat1 = self.vgg16_block1(yhat)
		yhat2 = self.vgg16_block2(yhat1)
		yhat3 = self.vgg16_block3(yhat2)
		yhat4 = self.vgg16_block4(yhat3)


		# Define content loss as the sum of l1_loss for each block output
		content_loss = blocks[0]*l1_loss(yhat1,y1) + blocks[1]*l1_loss(yhat2,y2)+blocks[2]*l1_loss(yhat3,y3)+blocks[3]*l1_loss(yhat4,y4)
		return content_loss


