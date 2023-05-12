import argparse
import os
import time
from classify_helper_functions import *

def run_image_classification(folder_path: str, 
        output_dir: str,
        json_file_path: str, 
        model_type: str, 
        tag: str
        ):
    
    # Get the output folder path.
    if output_dir is None : 
    # Create base directory for ./output. Create the output directory name with time-stamp.
        image_tagging_folder = create_out_folder(base_dir = './output')
    else :
        image_tagging_folder =  create_out_folder(base_dir = output_dir)
    print(f"[INFO] Output folder {image_tagging_folder}")

    # Classify unzipped data
    classify(folder_path, output_dir, json_file_path, model_type, tag, image_tagging_folder)
    # Classify ZIP archived data
    classify_zip(folder_path, output_dir, json_file_path, model_type, tag, image_tagging_folder)


def classify(
        folder_path: str, 
        output_dir: str,
        json_file_path: str, 
        model_type: str, 
        tag: str,
        image_tagging_folder: str
        ):
    """function to be handle unzipped folder.

    :param folder_path: path to the images' folder or archive file or single image file.
    :type foldr_path: str
    :param output_dir: directory for the classification output, 
    :type output_dir: str
    :param json_file_path: .json file containing the hash , clip features and meta-data of the image.
    :type json_file_path: str
    :param model_path: path to the model's .pkl file or the directory of models' pickle files/
    :type model_path: str
    :rtype: None
    """

    # Check if the data is ZIP archive
    if folder_path.endswith('.zip'): 
        # Data is ZIP archive return and run classify_zip
        return 

    if not os.path.isfile(folder_path):
        # Check for empty dirs
        empty_dirs_check(folder_path)
        # Placeholder for dataset file names
        img_files_list = []
        # Walking thru files
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                img_files_list.append(os.path.join(root, file))
    else:
        img_files_list = [folder_path]
    
    # Checking for zip archive and unsupported file format
    for file in img_files_list:
        if file.lower().endswith(('.zip')):
            # Exclude the zip file at this stage
            print (f'[WARNING] ZIP archive excluded: {file}')
            img_files_list.pop(img_files_list.index(file))
        elif not file.lower().endswith(('.gif','.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')):
            # Exclude file with unsupported image format
            print (f'[WARNING] Unsupported file: {file}')
            img_files_list.pop(img_files_list.index(file))
    
    # Get the output folder path
    image_tagging_subfolder = f'{image_tagging_folder}/tagging_result'
    os.makedirs(image_tagging_subfolder, exist_ok=True)

    # Load the json file
    metadata_json_obj = load_json(json_file_path)
    if metadata_json_obj is None:
        print("[WARNING] No json file loaded, calculating embeddings for every image.")

    # Get CLIP model, to calculate CLIP embeddings if it's not in .json metadata file.
    clip_model , preprocess , device = get_clip(clip_model_type= 'ViT-L-14',pretrained= 'laion2b_s32b_b82k')

    # Getting model
    classifier_model = get_classifier_model(model_type, tag)
    
    # If model not found then return
    if classifier_model=={}:
        print ('[INFO]: Model not found. No classification performed.')
        return


    out_json = {} # a dictionary for classification scores for every model.
        
    # Loop through each image in the folder.
    for img_file in tqdm(img_files_list):

        img_out_dict = classify_to_bin(
                                        img_file,
                                        classifier_model,
                                        metadata_json_obj,
                                        image_tagging_subfolder,
                                        clip_model,
                                        preprocess,
                                        device
                                    )
        if img_out_dict is None:
            continue

        out_json[img_out_dict['hash_id']] = img_out_dict

    save_json(out_json,image_tagging_subfolder) # save the output.json file

 

    #make sure result output path exists 


    # Extracting data from json_result from dataset
    json_keys = list(out_json.keys())
    for key in json_keys:
        file_name = os.path.split(out_json[key]['file_path'])[-1]
        file_path = out_json[key]['file_path']
        hash_id = out_json[key]['hash_id']
        model_outs = out_json[key]['classifiers_output']
        model_name = model_outs['model_name']
        model_type = model_outs['model_type']
        model_train_date = model_outs['model_train_date']
        tag = model_outs['tag']
        tag_score = model_outs['tag_score']
        __insert_data_into_database(
            file_name,
            file_path,
            os.path.splitext(file_name)[-1],
            hash_id,
            model_name,
            model_type,
            model_train_date,
            tag,
            str(tag_score)
            )

    print("[INFO] Finished.")


def classify_zip(
        folder_path: str, 
        output_dir: str,
        json_file_path: str, 
        model_type: str, 
        tag: str,
        image_tagging_folder: str
        ):

    # Place holder for the path of the zip files
    zip_files = []
    # Place holder for the attribute of zip files
    zips_info = []

    """function to handle ZIP archived data (contains folders / images).

    :param folder_path: path to the images' folder or archive file or single image file.
    :type foldr_path: str
    :param output_dir: directory for the classification output, 
    :type output_dir: str
    :param json_file_path: .json file containing the hash , clip features and meta-data of the image.
    :type json_file_path: str

    :param model_path: path to the model's .pkl file or the directory of models' pickle files/
    :type model_path: str
    :rtype: None
    """

    if not os.path.isfile(folder_path):
        # Check for empty dirs
        empty_dirs_check(folder_path)
        # Placeholder for data file names
        img_files_list = []
        # Walking thru files
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                img_files_list.append(os.path.join(root, file))
    else:
        img_files_list = [folder_path]
    
    # Selecting zip files only
    for file in img_files_list:
        if file.lower().endswith(('.zip')):
            zip_files.append(file)
    
    # If there is no zip file then return
    if len(zip_files) == 0:
        return

    # Get the output folder path
    image_tagging_subfolder = f'{image_tagging_folder}/zip_tagging_result'
    os.makedirs(image_tagging_subfolder, exist_ok=True)
    
    # Load the .json file.
    metadata_json_obj = load_json(json_file_path)
    if metadata_json_obj is None:
        print("[WARNING] No .json file loaded, calculating embeddings for every image.")

    # Get CLIP model, to calculate CLIP embeddings if it's not in .json metadata file.
    clip_model , preprocess , device = get_clip(clip_model_type= 'ViT-L-14',pretrained= 'laion2b_s32b_b82k')

    # Getting model
    classifier_model = get_classifier_model(model_type, tag)

    # If model not found then return
    if classifier_model=={}:
        print ('[INFO]: Model not found. No classification performed.')
        return


    out_json = {} # a dictionary for classification scores for every model.
        
    # Loop through each zip file.
    for file in tqdm(zip_files):
        # Generating images
        for img, img_file_name in tqdm(zip_gen(file, zips_info)):
            # Classify
            img_out_dict = classify_to_bin_zip(
                                            img,
                                            img_file_name,
                                            classifier_model,
                                            metadata_json_obj,
                                            image_tagging_subfolder,
                                            clip_model,
                                            preprocess,
                                            device
                                        )
            if img_out_dict is None:
                continue
            
            # Appending zip archive name to file path
            #img_out_dict['file_path'] = f"{file}/{img_out_dict['file_path']}"
            out_json[img_out_dict['hash_id']] = img_out_dict

    # Save to output.json file
    save_json(out_json,image_tagging_subfolder) 




if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory'    , type=str, required=True , help="images directory or image file")
    parser.add_argument('--output'       , type=str, required=False , default=None)
    parser.add_argument('--metadata_json', type=str, required=False , default=None)
    #parser.add_argument('--model'        , type=str, required=False, default=None)
    parser.add_argument('--model_type'  , type=str  ,required=True)
    parser.add_argument('--tag'  , type=str  ,required=True)

    args = parser.parse_args()

    # Run the main program 
    run_image_classification(
        folder_path    = args.directory, 
        output_dir     = args.output, 
        json_file_path = args.metadata_json, 
        model_type = args.model_type, 
        tag = args.tag
        ) 