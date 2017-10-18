import numpy as np
import pickle
import os.path
from matplotlib import pyplot as plt
import random

class cifar10_data:
    def __init__(self):
        self.img_bag=None
        self.img_label_lookup=None

    def showImageSample(self,SampleCategory=10,SampelImgCount=7):
        sample_img_bag={}
        idx_cat=np.random.choice(range(len(self.img_label_lookup)),SampleCategory,replace=False)
        for ic in idx_cat:
            idx_img=np.random.choice(range(self.img_bag.get(0).shape[0]),SampelImgCount,replace=False)
            sample_img_bag[ic]=self.img_bag[ic][idx_img,]
        fig,sp=plt.subplots(SampelImgCount,SampleCategory)
        fig.suptitle('Sample cifar10 Images')
        spidx=0
        for k in sample_img_bag:
            img_arr=sample_img_bag[k]
            for i in range(SampelImgCount):
                sp[i,spidx].imshow(self.returnImageFrom(img_arr[i,]).astype('uint8'))
                sp[i,spidx].axis('off')
            spidx=spidx+1
        plt.show()

    def returnImageFrom(self,data):
        img=data.reshape(3,32,32).transpose(1,2,0)
        return img

    def drawImage(self,data):
        plt.imshow(data)
        plt.show(block=True)

    def readImageLables(self):
        with open(os.path.join('cifar-10-batches-py','batches.meta'),'rb') as imgMeta:
            img_meta=pickle.load(imgMeta,encoding='bytes')
            img_label_lookup=[item.decode('utf-8') for item in img_meta[b'label_names']]
        self.img_label_lookup=img_label_lookup
        return img_label_lookup

    def readCIFRData(self,dirPath='./cifar-10-batches-py'):
        img_data=np.empty((0,3072))
        img_labels_int=np.empty((0))
        for i in range(1,6):
            with open(os.path.join(dirPath,'data_batch_'+str(i)),'rb') as imgfile:
                img_dict=pickle.load(imgfile,encoding='bytes')
                img_data=np.concatenate([img_data,img_dict[b'data']])
                img_labels_int=np.concatenate([img_labels_int,img_dict[b'labels']])
        img_bag={}
        for i in range(0,10):
            img_idx=[j for j,v in enumerate(img_labels_int) if v==i]
            #img_bag[img_label_lookup[i]]=img_data[img_idx,]
            img_bag[i]=img_data[img_idx,]
        self.img_bag=img_bag
        return img_bag

    def splitTrainTest(self,img_bag, num_train, num_test):
        x_train=np.zeros((0,list(img_bag.values())[0].shape[1]))
        y_train=np.zeros(0)
        x_test=np.zeros((0,list(img_bag.values())[0].shape[1]))
        y_test_actual=np.zeros(0)
        num_test_per_cat=int(num_test/len(img_bag.keys()))
        num_train_per_cat=int(num_train/len(img_bag.keys()))
        for k in img_bag.keys():
            idx=np.random.randint(1,img_bag[k].shape[0],num_train_per_cat)
            x_train=np.concatenate([x_train,img_bag[k][idx,]])
            y_train=np.concatenate([y_train,np.repeat(k,num_train_per_cat)])
            rest_idx=set(range(0,int(num_train_per_cat)))-set(idx)
            idx_test=np.array(random.sample(rest_idx,num_test_per_cat))
            x_test=np.concatenate([x_test,img_bag[k][idx_test,]])
            y_test_actual=np.concatenate([y_test_actual,np.repeat(k,num_test_per_cat)])
        return x_train,y_train,x_test,y_test_actual
