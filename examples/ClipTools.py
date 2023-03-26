import sys
sys.path.insert(0, '/content/kcg-ml/clip_linear_probe_pipeline/')
sys.path.insert(0, '/content/kcg-ml/')
from classify.classify_helper_functions import * 
import open_clip
from typing import Union
import numpy as np
import os
from zipfile import ZipFile
from PIL import Image
import torch
import hashlib
import json
import time

class ClipModel:
    ''' ClipModel class to get all clip model , preprocess and device '''
    def __init__(self, clip_model: str = 'ViT-L-14', pretrained:str = 'laion2b_s32b_b82k'):
        
        self.clip_model = clip_model
        self.pretrained = pretrained
        self.model , self.preprocess , self.device = get_clip(self.clip_model, self.pretrained)

    def download_model(self, model_name: str, pretrained: str):
        """ dowload specifc clip model to the machine. """
        if model_name is None or pretrained is None:
            raise ValueError("[ERROR] please enter the model type.")
        
        open_clip.create_model(model_name = model_name, pretrained= pretrained)
        print("[INFO] Model downloaded succesfully")
    

    def encode_image_from_image_file(self,image_file_path: str):
        """ encodes image with CLIP and returns ndArray of image features. """
        return clip_image_features(image_file_path, self.model ,self.preprocess,self.device)

    def encode_image_from_image_data(self, image_data: Union[bytes,bytearray] ):
        """ enconding image data with CLIP and returns ndArray of image features """
        return clip_image_features(image_data,self.model ,self.preprocess,self.device)
    
    
    def encode_image_list(self,image_list: Union[List[str], List[bytes], List[bytearray]]):
        """encoding a list of images with CLIP and returns a ndArray of all of their embeddings"""
        return np.stack((clip_image_features(image,self.model,self.preprocess,self.device) for image in image_list), axis=0)

    def empty_dirs_check(self, dir_path):
        """ Checking for empty directory and print out warning if any"""
        for dir in os.listdir(dir_path):
            sub_dir = os.path.join(dir_path, dir)
            # Check for directory only
            if os.path.isdir(sub_dir):
                if len(os.listdir(sub_dir)) == 0:
                    # Empty folder
                    print(f'[WARNING] Empty folder found. Ignoring it: {sub_dir}')
                    continue

    def get_clip_vector(self, img, image_file_name : str):
        with torch.no_grad():
            
            if image_file_name.lower().endswith('.gif'): 
                try:
                    img.seek(0)
                except:
                    print [f'[WARNING] Failed to convert {image_file_name} image.']
                    return
            else:
                # Image files other than gif
                img_obj = img

            image = self.preprocess(img_obj).unsqueeze(0).to(self.device)
            return self.model.encode_image(image).detach().numpy()

    def data_gen(self, data_file):
        '''Image generator for data_file'''
        if data_file.endswith('.zip'):
            # Selected data_dir is a zip archive
            
            with ZipFile(data_file) as archive:

                # Getting archive details
                entries = archive.infolist()

                for entry in entries:
                    # Do for every content in the zip file
                    if not entry.is_dir():
                        
                        with archive.open(entry) as file:

                            if entry.filename.lower().endswith(('.zip')):
                                # Another zip file found in the content. Process the content of the zip file
                                with ZipFile(file) as sub_archive:

                                    '''Getting archive details'''
                                    # Check the number of content
                                    sub_entries = sub_archive.infolist()

                                    for sub_entry in sub_entries:
                                        with sub_archive.open(sub_entry) as sub_file:
                                            try:
                                                img = Image.open(sub_file)
                                                yield (os.path.join(data_file, sub_entry.filename),img) # Changed to tuple (filename, img_obj)
                                            except Exception as e:
                                                print (f'[WWARNING] Failed to fetch {os.path.join(data_file, sub_entry.filename)}; {e}')
                                                continue
                            else:
                                # Should be image file. Read it.
                                try:
                                    img = Image.open(file)
                                    yield (os.path.join(data_file, entry.filename), img)
                                except Exception as e:
                                    print (f'[WARNING] Failed to fetch {entry.filename};  {e}')
                                    continue
        else:
            # Should be image file. Read it.
            try:
                img = Image.open(data_file)
                print (f' Fetching: {data_file}')
                yield (img)
            except:
                print (f'[WARNING] Failed to fetch {data_file}')

    def compute_hash(self, img, img_file_name):
        '''Compute image file to hash'''
        try:
            return hashlib.blake2b(img.tobytes()).hexdigest()
        except Exception as e:
            print(f"[ERROR] {e}:  cannot compute hash for {img_file_name}")
            return None 

    def encode_data_directory(self, data_dir: str):
        """encoding images in a specific zip file or in a directory
        or a dictory of zip files.
        """
        # Placeholder for data file names
        files_list = []
        
        if not os.path.isfile(data_dir):
            '''For normal directory'''
            # Check for empty dirs
            self.empty_dirs_check(data_dir)
            # Walking thru files
            for root, _, files in os.walk(data_dir):
                for file in files:
                    files_list.append(os.path.join(root, file))
        else:
            '''Single file (could be a zip archive or image)'''
            files_list = [data_dir]

        list_of_info = [] # List contains information dict. about each image.
        for i, file in enumerate(files_list):
            '''Fetching images'''
            start_time = time.time() # The start of the zip files
            img_counter = 0  # Counting the number of images per zip file/directory.
            for img_path, img in self.data_gen(file):
                # print(f'[INFO] Calculating CLIP vector for {img_path}...')
                # Compute clip vector
                clip_vector = self.get_clip_vector(img, file)
                # Insert image to cache
                list_of_info.append({
                    "type": "@ClipScore",
                    "image-path": img_path,
                    "image-hash": self.compute_hash(img, file), 
                    "clip-model": self.clip_model,
                    "pretrained": self.pretrained,
                    "clip-vector" : clip_vector.tolist()
                })
                img_counter += 1
            print(f"[INFO] ZIP {i+1} OF {len(files_list)}, TIME : {time.time() - start_time:.2f} SECS, {img_counter} IMAGES")
        # Dump json into a clip-scores.json in output directory.
        with open("clip-scores.json", "w") as outfile:
            json.dump(list_of_info, outfile, indent=4)
        outfile.close()
