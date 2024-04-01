import torch.utils.data as data
import torch
import os
import json
import random
from PIL import Image
import numpy as np
import torchvision.transforms as tr
import monai.transforms as mtr

class EvalDataset(data.Dataset):
    '''
    Args:
        dataset_list: random choice one dataset from list 
        class_dict: 
            dict, { 'dataset_name1':['class_name1', 'class_name2',...], ...}
        scale_dict: 
            dict, { 'class_name1': [low, high], 'class_name2': [low, high], ...}
            WW: window width
            WL: window level
        n_pairs
        dataset_info_json_dir 
        input_size
        is_train
        transform
        target_transform
        fullset
        use_monai: 
            bool, use monai transform or not
        random_sample_ratio: 
            float, random sample ratio, random sample one image from all classes
            0 means no random sample
        random_first:
            bool, random sample first image in sentence or not
        shuffle_epoch: 
            bool, shuffle dataset every epoch or not
        epoch: 
            int, epoch number, begin from 1
    '''
    def __init__(self, 
                 dataset_list: list, 
                 class_dict: dict,
                 scale_dict:dict,
                 n_pairs: int, 
                 dataset_info_json_dir : str, 
                 input_size: int,
                 is_train='train', 
                 transform=None, 
                 target_transform=None, 
                 fullset=True,
                 use_monai=False,
                 image_channel=3,
                 exclusive_classes=None,
                 eval_all=True,
                #  shuffle_epoch=True,
                #  epoch=1):
    ):
        self.dataset_list = dataset_list
        self.class_dict = class_dict
        self.scale_dict = scale_dict 
        self.n_pairs = n_pairs
        self.dataset_info_json_dir = dataset_info_json_dir
        self.input_size = input_size
        self.is_train = is_train
        self.transform = transform
        self.target_transform = target_transform
        self.fullset = fullset
        self.use_monai = use_monai
        # self.shuffle_epoch = shuffle_epoch
        # self.epoch = epoch
        self.image_channel = image_channel
        self.exclusive_classes = exclusive_classes
        self.eval_all = eval_all

        self.dataset_info_list = self._get_dataset_info()
        self.sentences_list = self._get_sentences_list()
    
    def _get_dataset_info(self):
        '''
        Returns: list, contains all dataset info
            dataset_info : dict, contains one dataset info
                dataset_info['train_patient'] : list, contains all train patient name
                dataset_info['val_patient'] : list, contains all val patient name
                dataset_info['train_image'] : list, contains all train image info, num = class_num + 2, 
                                            list[0] contains all background image info, 
                                            list[1] contains all class 1 image info,
                                            list[-1] contains all image info
                    image_info : dict, contains slice path list, class name
                        image_info['slice_path'] : list, slice hdf5 file path
                        image_info['class_name'] : str, patient id
                            slice_path: str, slice hdf5 file path
                            each hdf5 file contains 2 keys: image and label
                dataset_info['val_image'] : list, contains all val image info like train_image
        '''
        dataset_info_list = []
        for dataset_name in self.dataset_list:
            if self.is_train == 'test':
                split_json_path = os.path.join(self.dataset_info_json_dir, dataset_name+'--test_json.json')
            else:
                split_json_path = os.path.join(self.dataset_info_json_dir, dataset_name+'--train_split_json.json')
            # check if split json exists and raise error if not
            if not os.path.exists(split_json_path):
                raise FileNotFoundError(f"Split json file not found: {split_json_path}")
            with open(split_json_path, 'r') as f:
                dataset_info_list.append(json.load(f))
        return dataset_info_list
    
    def get_pairs_by_dataset(self, dataset_info, class_name):
        '''
        generate list of n_pairs (sentence) of a dataset
        sentence: [slice1_path, slice2_path, ..., class_name] contains n+1 elements
        pairs_list: [ sentence1, sentence2,...]
        Returns: 
            list, contains all n_pairs (sentence) of a dataset
        '''

        pairs_list = []
        if self.is_train == 'train':
            mode = 'train_image'
        else:
            mode = 'val_image'
        
        class_num = len(dataset_info[mode])
        class_idx = -1
        for idx in range(class_num):
            if dataset_info[mode][idx]['class_name'] == class_name:
                class_idx = idx
                break
        if class_idx == -1:
            raise ValueError(f"Class name not found: {class_name}")
        
        image_num = len(dataset_info[mode][class_idx]['slice_path'])
        sentence_num = image_num
        image_idx = list(range(image_num))
        random.shuffle(image_idx)

        if self.eval_all:
            for i in range(sentence_num):
                sentence = []
                new_idx = image_idx[:i]+image_idx[i+1:]
                prompt_idx = random.sample(new_idx, self.n_pairs-1)
                prompt_idx.append(image_idx[i])
                for j in range(self.n_pairs):
                    image_info = dataset_info[mode][class_idx]['slice_path'][prompt_idx[j]]
                    sentence.append(image_info)
                sentence.append(class_idx)
                sentence.append(class_name)
                pairs_list.append(sentence)
        else:
            sentence_num = image_num // self.n_pairs
            for i in range(sentence_num):
                sentence = []
                prompt_idx = image_idx[i*self.n_pairs:(i+1)*self.n_pairs]
                for j in range(self.n_pairs):
                    image_info = dataset_info[mode][class_idx]['slice_path'][prompt_idx[j]]
                    sentence.append(image_info)
                sentence.append(class_idx)
                sentence.append(class_name)
                pairs_list.append(sentence)

        return pairs_list
    
    def _get_sentences_list(self):
        '''
        Returns: 
            list, contains all n_pairs (sentence) of all dataset and all specific classes
        '''
        sentences_list = []
        exclusive_classes = self.exclusive_classes if self.exclusive_classes is not None else []
        for idx, dataset_info in enumerate(self.dataset_info_list):
            dataset_name = self.dataset_list[idx].split('--')[1]
            for class_name in self.class_dict[dataset_name]:
                if class_name not in exclusive_classes:
                    sentences_list.extend(self.get_pairs_by_dataset(dataset_info, class_name))
        return sentences_list

   
    def hdf5_reader(self, hdf5_path):
        '''
        read hdf5 file
        Args:
            hdf5_path: str, hdf5 file path
        Returns:
            image: np.array, image data
            label: np.array, label data
        '''
        import h5py
        with h5py.File(hdf5_path, 'r') as f:
            image = np.array(f['image'])
            label = np.array(f['label'])
        return image, label
    
    def __getitem__(self, index):
        '''
        get one sentence
        Returns: 
            images : list, contains all images of a sentence 
                n_pairs, c, h, w
            labels : list, contains all labels of a sentence
                n_pairs, 1, h, w
        '''
        sentence = self.sentences_list[index]
        images = []
        labels = []
        class_idx = sentence[-2]
        class_name = sentence[-1]
        for i in range(self.n_pairs):
            
            image, label = self.hdf5_reader(sentence[i])

            # binary label
            label = (label==class_idx).astype(np.uint8)
            # scale HU value to [0,1]
            if self.scale_dict is not None:
                trunc_low = self.scale_dict[class_name][0]
                trunc_high = self.scale_dict[class_name][1]
                image = np.clip(image, trunc_low, trunc_high)
                image = (image - trunc_low) / (trunc_high - trunc_low) 

            
            if self.use_monai:
                # change hwc to chw 
                # change label from 0,255 to 0,1
                if image.ndim == 2:
                    image = np.expand_dims(image, axis=2)
                if label.ndim == 2:
                    label = np.expand_dims(label, axis=2)
                if image.max() > 1:
                    image = image / 255
                if label.max() > 1:
                    label = label / 255
                image = np.transpose(image, (2,0,1))
                label = np.transpose(label, (2,0,1))
                if self.transform is not None:
                    data_dict = self.transform({'image':image, 'label':label})
                    image = data_dict['image']
                    label = data_dict['label']
                else:
                    transform = mtr.Compose([
                        mtr.Resized(['image','label'],(self.input_size, self.input_size)),
                        mtr.ToTensord(['image','label'],track_meta=False),
                    ])
                    data_dict = transform({'image':image, 'label':label})
                    image = data_dict['image']
                    label = data_dict['label']
                    label = (label>0.5).float()

            else:
                label = label * 255
                image = Image.fromarray(image)
                label = Image.fromarray(label)
                
                if self.transform is not None:
                    image = self.transform(image)
                else:
                    transform = tr.Compose([
                        tr.Resize((self.input_size, self.input_size)),
                        tr.ToTensor(),
                    ])
                    image = transform(image)

                if self.target_transform is not None:
                    label = self.target_transform(label)
                else:
                    target_transform = tr.Compose([
                        tr.Resize((self.input_size, self.input_size),interpolation=Image.NEAREST),
                        tr.ToTensor(),
                    ])
                    label = target_transform(label)
                    label = (label>0.5).float()

                if image.shape[0] == 1 and self.image_channel == 3:
                    image = image.repeat(3,1,1)
                
            images.append(image)
            labels.append(label)

        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)
        return images, labels, (class_idx,index), class_name
    
    def __len__(self):
        return len(self.sentences_list)
    

    # def set_epoch(self, epoch):
    #     '''
    #     set epoch
    #     '''
    #     self.epoch = epoch
    
    # def get_epoch(self):
    #     '''
    #     get epoch
    #     '''
    #     return self.epoch

    def reset_sentences_list(self):
        '''
        reset sentences_list
        '''
        self.sentences_list = self._get_sentences_list()

    # def reset_sentences_list_auto(self):
    #     '''
    #     reset sentences_list
    #     '''
    #     self.sentences_list = self._get_sentences_list()
    #     self.epoch += 1
    
    def get_sentence_list(self):
        '''
        get sentences_list
        '''
        return self.sentences_list



