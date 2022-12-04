

### Pipeline Colab Notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kk-digital/image-tagging-tools/blob/main/jupyter-notebook-example.ipynb)

You can run the pipeline on google colab using the following [link](https://colab.research.google.com/github/kk-digital/image-tagging-tools/blob/main/jupyter-notebook-example.ipynb)

## Installation
All what's needed to start using the pipeline locally is to have python 3.9+ then run the following command

```sh
pip install -r ./requirements.txt
```

# Stage 1: Image Dataset Processor

> A standalone tool for for processing dataset of tagged images and calculating its metadata and the CLIP embeddings. 

## Tool Description

process a directory of images (paths to directory of images or an archived dataset), and computes the images metadata along with its CLIP embeddings and writes the result into a JSON file into `output_folder`

## Example Usage

* For a tagged dataset and save the output into `output` folder in the root directory. 

```sh
python ./stage1/ImageDatasetProcessor.py --input_folder='./my-dataset' 
```

* For a non-tagged dataset and save the output into `output/clip-cache` folder.

```sh
python ./stage1/ImageDatasetProcessor.py --input_folder='./my-dataset' --tagged_dataset=False
```

The tool will immediately starts working, and output any warnings or error into the std while working. 

## CLI Parameters

* `input_folder` _[string]_ - _[required]_ - path to the directory containing sub-folders of each tag.
* `output_folder` _[string]_ - _[optional]_ - path to the directory where to save the files into it.
* `tagged_dataset` _[bool]_ - _[optional]_ - the dataset to process is a tagged dataset such that each each parent folder name is the tag of the images contained within it, default is `True`

* `clip_model` _[str]_ - _[optional]_ CLIP model to be used, default is `'ViT-B-32'`

* `pretrained` _[str]_ - _[optional]_ -  the pre-trained model to be used for CLIP, default is `'openai'`

* `batch_size` _[int]_ - _[optional]_ -  number of images to process at a time, default is `32`. 
* `num_threads` _[int]_ - _[optional]_ - the number to be used in this process, default is `4`

* `device` _[str]_ - _[optional]_ -  the device to be used in computing the CLIP embeddings, if `None` is provided then `cuda` will be used if available, default is `None`


# Stage 2: Train Classifier


> A script for training classification models based on `input-tag-to-image-hash-list.json` file and `input-metadata.json` file.

## Tool Description

Given a `metadata` json file containing embeddings for images and `tag-to-image-hash` json file containing images' hash with tags, the script start to make for every tag two binary classification models and save it in output folder.

## Example Usage

```
python ./stage2/train.py --metadata_json  './output/input-metadata.json' --tag_to_hash_json './output/input-tag-to-image-hash-list.json'
```

> Note that if the `output` is not created the script automatically creates it for you. 


> Note that if the `test_per` is not created the script will make test set ~= 10% of the dataset.

Also you may call `--help` to see the options and their defaults in the cli. 

## CLI Parameters

* `metadata_json` _[string]_ - _[required]_ - The path to the metadata json file. 
* `tag_to_hash_json` _[string]_ - _[required]_ - The path to tag-to-hash json file. 

* `output` _[string]_ - _[optional]_ - The path to the output directory.
* `test_per` _[float]_ - _[optional]_ - The percentage of the test images from the dataset, default = 0.1 


# Stage 3: Classify Dataset

> A script for classification models inference given images' `directory` or .zip file and `metadata_json` .json file.




## Tool Description

Given a `metadata_json` json file containing embeddings for images and `directory` of images' folder or .zip file, the script start to loop over every image and make the classification for it using every binary classification model.

## Example Usage

```
python ./stage3/classify.py --metadata_json './output/input-metadata.json' --directory ‘./output/images_directory’
```

```
python ./stage3/classify.py --metadata_json './input-metadata.json' \
                            --directory ‘/src/to/dir/images_directory’
                            --output ‘./classification_output’
                            --output_bins 10
                            --model ‘./output/models’

```



> Note that if the `output` is not created the script automatically creates it for you. 

> Note that if the `model` is not created the script automatically uses models in [outputs/models](outputs/models/)

Also you may call `--help` to see the options and their defaults in the cli. 

## CLI Parameters

* `directory` _[string]_ - _[required]_ - The path to the images' folder or images' .zip file. 
* `metadata_json` _[string]_ - _[required]_ - The path to the metadata json file for CLIP embeddings. 
* `output` _[string]_ - _[optional]_ - The path to the output directory for the inference results. 
* `model` _[string]_ - _[optional]_ - The path to the models' .pkl files directory or single .pkl file model.
* `output_bins` _[int]_ - _[optional]_ -  The number of bins of the results for each model.




# Examples

## Binary classification of a single image



```python
import argparse
from classify_helper_functions import *

def main(
        image_path: str, 
        bins_number : int, 
        model_path : str, 
    ):

    image_tagging_folder = "./single_image_classification"
    bins_number = 10
    out_json = {}

    clip_model , preprocess , device = get_clip(clip_model_type= 'ViT-B-32',pretrained= 'openai')
    bins_array = get_bins_array(bins_number) 
    blake2b_hash = file_to_hash(image_path)

    models_dict = create_models_dict(model_path)
    image_features = clip_image_features(image_path,clip_model,preprocess,device) # Calculate image features.

    classes_list = [] # a list of dict for every class 
    for model_name in models_dict:
        image_class_prob     = classify_image_prob(image_features,models_dict[model_name]) # get the probability list
        model_type, tag_name = get_model_tag_name(model_name) 
        tag_bin, other_bin   = find_bin(bins_array , image_class_prob) # get the bins 

        # Find the output folder and create it based on model type , tag name 
        tag_name_out_folder = make_dir([image_tagging_folder, f'{model_type}',f'{tag_name}',tag_bin])

        # Copy the file from source to destination 
        shutil.copy(image_path,tag_name_out_folder)

    classes_list.append({
                        'model_type' : model_type,
                        'tag_name'   : tag_name,
                        'tag_prob'   : image_class_prob[0]}
                        )

                                
    out_json[blake2b_hash] = {
            'hash_id'      : blake2b_hash,
            'file_path'    : image_path, 
            'classes'      : classes_list
            }
        
    save_json(out_json,image_tagging_folder) # save the .json file
    print("[INFO] Finished.")

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path'    , type=str, required=True)
    parser.add_argument('--model_path'        , type=str, required=True)
    parser.add_argument('--output_bins'  , type=int  ,required=False , default=5)

    args = parser.parse_args()

    # Run the main program 
    main(
        image_path    = args.image_path, 
        model_path     = args.model_path,
        bins_number    = args.output_bins
        )


```