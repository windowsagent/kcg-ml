import argparse
import os
import sqlite3
import time
from classify_helper_functions import *

def main(folder_path: str, 
        output_dir: str,
        json_file_path: str, 
        bins_number: int, 
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
    classify(folder_path, output_dir, json_file_path, bins_number, model_type, tag, image_tagging_folder)
    # Classify ZIP archived data
    classify_zip(folder_path, output_dir, json_file_path, bins_number, model_type, tag, image_tagging_folder)


def classify(
        folder_path: str, 
        output_dir: str,
        json_file_path: str, 
        bins_number: int, 
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
    :param bins_numbers: number of bins to divide the output into.
    :type bins_number: int
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
    clip_model , preprocess , device = get_clip(clip_model_type= 'ViT-L-14',pretrained= 'openai')

    # Getting model
    classifier_model = get_classifier_model(model_type, tag)
    
    # If model not found then return
    if classifier_model=={}:
        print ('[INFO]: Model not found. No classification performed.')
        return

    # Creating bin
    bins_array  = get_bins_array(bins_number) 

    out_json = {} # a dictionary for classification scores for every model.
        
    # Loop through each image in the folder.
    for img_file in tqdm(img_files_list):

        img_out_dict = classify_to_bin(
                                        img_file,
                                        classifier_model,
                                        metadata_json_obj,
                                        image_tagging_subfolder,
                                        bins_array,
                                        clip_model,
                                        preprocess,
                                        device
                                    )
        if img_out_dict is None:
            continue

        out_json[img_out_dict['hash_id']] = img_out_dict

    save_json(out_json,image_tagging_subfolder) # save the output.json file

    '''
    Database writing
    Creating database and table for writing json_result data from dataset
    '''
    
    db_out_dir = output_dir
    #make sure result output path exists 
    os.makedirs(db_out_dir, exist_ok = True)
    DATABASE_NAME = '/score_cache.sqlite'
    DATABASE_PATH = f'{db_out_dir}/{DATABASE_NAME}'

    def __delete_all_data_in_database():
        __delete_database()
        __create_database()

    def __create_database():
        cmd1 = '''CREATE TABLE score_cache (
        file_name   TEXT    NOT NULL,
        file_path   TEXT    NOT NULL,
        type        TEXT,
        hash_id     TEXT,
        model_name  TEXT,
        model_type  TEXT,
        model_train_date  TEXT,    
        tag    TEXT,
        tag_score    REAL    
        );
        '''
        db = sqlite3.connect(DATABASE_PATH)
        c = db.cursor()
        c.execute('PRAGMA encoding="UTF-8";')
        c.execute(cmd1)
        db.commit()
    
    def __delete_database():
        try:
            if(os.path.exists(DATABASE_PATH)):
                os.remove(DATABASE_PATH)
        except Exception as e:
            print(str(e))
            time.sleep(1)
            __delete_database()

    def __insert_data_into_database(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9):
        try:
            cmd = "insert into score_cache(file_name, file_path, type, hash_id, model_name, model_type, model_train_date, tag, tag_score) values ('"+arg1+"', '"+arg2+"', '"+arg3+"', '"+arg4+"', '"+arg5+"', '"+arg6+"', '"+arg7+"', '"+arg8+"', '"+arg9+"')"
            with sqlite3.connect(DATABASE_PATH) as conn:
                conn.execute(cmd)
                conn.commit()
        except Exception as e:
            if(str(e).find('lock') != -1 or str(e).find('attempt to write a readonly database') != -1):
                time.sleep(1)

    print (f'[INFO] Writing to database table in {DATABASE_PATH}')
    
    __delete_all_data_in_database()

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
        bins_number: int, 
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
    :param bins_numbers: number of bins to divide the output into.
    :type bins_number: int
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
    clip_model , preprocess , device = get_clip(clip_model_type= 'ViT-L-14',pretrained= 'openai')

    # Getting model
    classifier_model = get_classifier_model(model_type, tag)

    # If model not found then return
    if classifier_model=={}:
        print ('[INFO]: Model not found. No classification performed.')
        return

    # Creating bin
    bins_array  = get_bins_array(bins_number) 

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
                                            bins_array,
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


    '''
    Database writing
    Creating database and table for writing file info, model and classification result.
    '''
    
    db_out_dir = output_dir
    #make sure result output path exists 
    os.makedirs(db_out_dir, exist_ok = True)
    DATABASE_NAME = '/zip_score_cache.sqlite'
    DATABASE_PATH = f'{db_out_dir}/{DATABASE_NAME}'
    
    print (DATABASE_PATH)
    def __delete_all_data_in_database():
        __delete_database()
        __create_database()

    def __create_database():
        cmd1 = '''CREATE TABLE zip_score_cache (
        file_name   TEXT    NOT NULL,
        file_path   TEXT            ,
        archive_path    TEXT        ,
        type            TEXT        ,
        n_img_content   INTEGER     ,
        hash_id     TEXT            ,
        model_name  TEXT            ,
        model_type  TEXT            ,
        model_train_date  TEXT      ,
        tag    TEXT                 ,
        tag_score    REAL    
        );
        '''

        db = sqlite3.connect(DATABASE_PATH)
        c = db.cursor()
        c.execute('PRAGMA encoding="UTF-8";')
        c.execute(cmd1)
        db.commit()
    
    def __delete_database():
        try:
            if(os.path.exists(DATABASE_PATH)):
                os.remove(DATABASE_PATH)
        except Exception as e:
            print(str(e))
            time.sleep(1)
            __delete_database()

    def __insert_file_into_database(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9):
        try:
            cmd = "insert into zip_score_cache(file_name, file_path, type, hash_id, model_name, model_type, model_train_date, tag, tag_score) values ('"+arg1+"', '"+arg2+"', '"+arg3+"', '"+arg4+"', '"+arg5+"', '"+arg6+"', '"+arg7+"', '"+arg8+"', '"+arg9+"')"
            with sqlite3.connect(DATABASE_PATH) as conn:
                conn.execute(cmd)
                conn.commit()
        except Exception as e:
            if(str(e).find('lock') != -1 or str(e).find('attempt to write a readonly database') != -1):
                time.sleep(1)

    def __insert_zip_into_database(arg1, arg2, arg3, arg4):
        try:
            cmd = "insert into zip_score_cache(file_name, archive_path, type, n_img_content) values ('"+arg1+"', '"+arg2+"', '"+arg3+"', '"+arg4+"')"
            with sqlite3.connect(DATABASE_PATH) as conn:
                conn.execute(cmd)
                conn.commit()
        except Exception as e:
            if(str(e).find('lock') != -1 or str(e).find('attempt to write a readonly database') != -1):
                time.sleep(1)

    __delete_all_data_in_database()

    print (f'[INFO] Writing to database table in {DATABASE_PATH}')

    '''
    Writing ZIP file info, model and classification result to the table by 
    '''
    for item in zips_info:
        file_name = os.path.basename(item[0])
        arch_path = item[0]
        n_content = item[1]
        __insert_zip_into_database(
            file_name,
            arch_path,
            os.path.splitext(file_name)[-1],
            str(n_content)
            )

    '''
    Writing file info, model and classification result to the table by 
    extracting data from out_json
    '''
    json_keys = list(out_json.keys())
    for key in json_keys:
        file_name = os.path.split(out_json[key]['file_path'])[-1]
        #file_path = os.path.split(out_json[key]['file_path'])[0]
        file_path = out_json[key]['file_path']
        hash_id = out_json[key]['hash_id']
        model_outs = out_json[key]['classifiers_output']
        #for out in model_outs:
        model_name = model_outs['model_name']
        model_type = model_outs['model_type']
        model_train_date = model_outs['model_train_date']
        tag = model_outs ['tag']
        tag_score = model_outs['tag_score']
        __insert_file_into_database(
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


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory'    , type=str, required=True , help="images directory or image file")
    parser.add_argument('--output'       , type=str, required=False , default=None)
    parser.add_argument('--metadata_json', type=str, required=False , default=None)
    #parser.add_argument('--model'        , type=str, required=False, default=None)
    parser.add_argument('--output_bins'  , type=int  ,required=False , default=10)
    parser.add_argument('--model_type'  , type=str  ,required=True)
    parser.add_argument('--tag'  , type=str  ,required=True)

    args = parser.parse_args()

    # Run the main program 
    main(
        folder_path    = args.directory, 
        output_dir     = args.output, 
        json_file_path = args.metadata_json, 
        bins_number    = args.output_bins,
        model_type = args.model_type, 
        tag = args.tag
        ) 