import os
import functools
import glob
import random
import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Literal
import PIL
import PIL.Image
import torch
from torch.utils.data import Dataset
import numpy as np
import datasets

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None



class MacirDataset(Dataset):
    """
   MACIR dataset class for PyTorch dataloader.
   The dataset can be used in 'query' or 'database' mode:
        - In 'database' mode the dataset yield a dict with keys ['image', 'image_name']
        - In 'query' mode the dataset yield dict with keys:
            - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_caption', 'composition_type','condition_type']
    """

    def __init__(self, dataset_path: Union[Path, str], split,
                 mode: Literal['query', 'database'], preprocess: callable, eval_type: str,eval_level: str, test_mode=False, no_duplicates: Optional[bool] = False):
        """
        :param dataset_path: path to the MACIR dataset
        :param split: dataset split, should be in ['add','remove','replace']
        :param preprocess: function which preprocesses the image
        :param no_duplicates: if True, the dataset will not yield duplicate images in query mode, does not affect database mode
        """
        dataset_path = Path(dataset_path)
        self.dataset_path = dataset_path
        self.preprocess = preprocess
        self.mode = mode
        self.split = split
        self.no_duplicates = no_duplicates
        self.eval_type=eval_type
        self.eval_level=eval_level
        self.test_mode=False

        if mode not in ['query', 'database']:
            raise ValueError("mode should be in ['query', 'database']")
        
        if self.eval_level=="restricted":
            with open(dataset_path / "meta_data" / f'macir_meta_{split}.json') as f:
                self.triplets = json.load(f)    
        else:
            with open(dataset_path / "meta_data" / 'macir_meta_full.json') as f:
                self.triplets = json.load(f)

        if self.eval_level=="restricted":
            with open(dataset_path / "meta_data" / f"image_ids_{split}.json") as f:
                self.image_names = json.load(f)
        else:
            with open(dataset_path / "meta_data" / f"image_ids.json") as f:
                 self.image_names = json.load(f)   
        
 
        

    def __getitem__(self, index) -> dict:
            if self.mode == 'query':
                reference_name = self.triplets[index]['reference_image']
                relative_caption = self.triplets[index]['relative_caption']
                condition_type= self.triplets[index]['condition_type']
                composition_type= self.triplets[index]['composition_type']
                reference_image_path = self.dataset_path / "images" / reference_name
                target_name = self.triplets[index]['target_image']
                target_image_path = self.dataset_path /"images" / target_name
                
            
                reference_image = self.preprocess(PIL.Image.open(reference_image_path), return_tensors='pt')['pixel_values'][0]
                target_image = self.preprocess(PIL.Image.open(target_image_path), return_tensors='pt')['pixel_values'][0]
                
                return {
                    'reference_image': reference_image,
                    'reference_name': reference_name,
                    'target_image': target_image,
                    'target_name': target_name,
                    'relative_caption': relative_caption,
                    'composition_type': composition_type,
                    'condition_type': condition_type,
                }

            elif self.mode == 'database':
                image_name = self.image_names[index]               
                image_path = self.dataset_path / "images" / image_name    
                im = PIL.Image.open(image_path)
                
                image = self.preprocess(im, return_tensors='pt')['pixel_values'][0]
                return {
                    'image': image,
                    'image_name': image_name
                }
            else:
                raise ValueError("mode should be in ['query', 'database']")


    def __len__(self):
        if self.mode == 'query':
            return len(self.triplets)
        elif self.mode == 'database':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['query', 'database']")
