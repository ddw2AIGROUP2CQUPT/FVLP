import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from PIL import Image
from data.utils import pre_caption
import random


class face450_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):        
       
        filename = 'train_450.json'
        self.annotation = json.load(open(os.path.join(ann_root, filename),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann['photo_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['photo'])        
        image = Image.open(image_path).convert('RGB') 
        image = self.transform(image)

        idx = random.randint(0, len(ann['sketch_sde'])-1)
        sketch_sde_path = os.path.join(self.image_root, ann['sketch_sde'][idx])        
        sketch_sde = Image.open(sketch_sde_path).convert('RGB')
        sketch_sde = self.transform(sketch_sde)
        
        caption = self.prompt + pre_caption(ann['caption'], self.max_words) 

        return image, sketch_sde, caption, self.img_ids[ann['photo_id']] 
    
    
class face450_test(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=65):
        
        filenames = {'test': 'test_450.json'}
        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]),'r'))     
        self.transform = transform
        self.image_root = image_root
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['photo'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.annotation[index]['photo'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        sketch_sdes = []
        sketch_sde_paths = self.annotation[index]['sketch_sde']
        for sketch_sde_path in sketch_sde_paths:
            sketch_sde_path = os.path.join(self.image_root, sketch_sde_path)
            sketch_sde = Image.open(sketch_sde_path).convert('RGB')
            sketch_sdes.append(self.transform(sketch_sde))
        sketch_sdes = torch.stack(sketch_sdes)

        return image, sketch_sdes, index