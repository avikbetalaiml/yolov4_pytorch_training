import os
import numpy as np



class train_val_split:
	def __init__(self,image_path='', split=None):
		self.image_path = image_path
		self.all_images = np.array(os.listdir(self.image_path+'/images'))
		self.val = split
		#print(self.all_images,'print up')

	def generate_text(self):
		np.random.shuffle(self.all_images)
		
		
		val_data = int(len(self.all_images)*self.val/100)
		train_data = len(self.all_images) - val_data
		
		train_images = self.all_images[:train_data]
		test_images = self.all_images[-val_data:]
		f_train=open("train.txt", "w+")
		f_test=open("val.txt", "w+")
		for image in train_images:
			#f.write("This is line %d\r\n" % (i+1))
			f_train.write("datasets/"+self.image_path+'/images/'+image+'\n')
		for image in test_images:
			f_test.write("datasets/"+self.image_path+'/images/'+image+'\n')




if __name__ == '__main__':
	test = train_val_split(image_path = 'training_data', split=30)
	test.generate_text()
