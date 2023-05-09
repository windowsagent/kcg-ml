import sys
sys.path.append('./')
sys.path.append('./image_classifier_pipeline/model_api/')
import argparse
import warnings
import numpy as np
from datetime import datetime
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from logistic_regression_pytorch import LogisticRegressionPytorch
from train_helper_functions import *
from model_api import ModelApi


warnings.filterwarnings('ignore')

def train_classifier(
        metadata_json    : str, 
        tag_to_hash_json : str,
        output_dir : str, 
        test_per : float,
        ):
    """train_classifier function to be running, calls other function for making trained models pickle files,
    and making of mapping files.

    :param metadata_json: path to the metadata json file containing embeddings and tags.
    :type metadata_json: str
    :param tag_to_hash_json: path to tag-to-hash json file containing embeddings and tags.
    :type tag_to_hash_json: str
    :param output_dir: directory for the classification models pickle files and mappings jsons. 
    :type output_dir: str
    :param test_per: percentage of the test embeddings 
    :type test_per: float
    :rtype: None
    """

    # Classifier Model API object
    model_api = ModelApi()

    # load tag to hash json and metadata json.
    metadata_dict    = load_json(metadata_json)
    tag_to_hash_json = load_json(tag_to_hash_json)
    if metadata_dict is None or tag_to_hash_json is None : # Problem happened with the json file loading.
        return

    # Get training start time
    t_start = datetime.now()
    # get the two output folder paths (models and reports) with respect to \
    # the output directory provided in the script.
    report_out_folder , models_out_folder = check_out_folder(output_dir) 

    # other training and other validation embeddings lists.
    other_all_emb_list     = [metadata_dict[hash_id]["embeddings_vector"] for hash_id in tag_to_hash_json['other-training']]
    other_val_all_emb_list = [metadata_dict[hash_id]["embeddings_vector"] for hash_id in tag_to_hash_json['other-validation']]

    # get embeddings from tag_emb_dict and make it ready for training the classifier 
    for tag in tag_to_hash_json:

        # make sure that it's a pixel art class tag. 
        if tag in ['other-training' ,'other-validation']:
            continue

        # get embedding list of tag images.
        tag_all_emb_list = [metadata_dict[hash_id]["embeddings_vector"] for hash_id in tag_to_hash_json[tag]]

        # get train test embeddings and labels.
        train_emb, train_labels, test_emb, test_labels , t_n , o_n = get_train_test(tag_all_emb_list, other_all_emb_list , test_per)

        model_type = 'torch-logistic-regression'

        # Check if classifier model with model_type and tag already exist.
        model = model_api.get_model_by_type_tag(model_type, tag)

        if len(model)>0:
            # Existing classifier model with model_type and tag found. Do not create new one
            print (f'Classifier model for type: {model_type} and tag: {tag} already exist. Training will use existing model')
            classifier = model['classifier']
            print (f"MODEL TYPE {model_type}, {model['model_type']}")
        else:
            # No existing classifier model with model_type and tag. Initialize new classifier model
            print (f'Initialize new classifier model for type: {model_type} and tag: {tag}')
            classifier = LogisticRegressionPytorch(output_dim=1)
        
        classifier = train_loop(model = classifier, train_emb=train_emb, train_labels=train_labels)
        
        test_emb = torch.from_numpy(test_emb.astype(np.float32))
        predictions = classifier(test_emb)
        predictions = predictions.round()
        predictions = predictions.round().view(1,-1).squeeze().detach().numpy()

        # get histogram data.
        in_tag_tagged  = histogram_list(np.array(tag_all_emb_list), classifier, other=False, using_torch=(model_type == 'torch-logistic-regression')) # histogram data for in-tag images 
        out_tag_tagged = histogram_list(np.array(other_val_all_emb_list), classifier,  other=True, using_torch=(model_type == 'torch-logistic-regression')) # histogram data for out-tag images

        # put all lines for text file report in one .
        text_file_lines = [ f"model: {model_type}\n", "task: binary-classification\n",
                            f"tag: [{tag}]\n\n", f"tag-set-image-count:   {len(tag_all_emb_list)} \n",
                            f"other-set-image-count: {len(other_all_emb_list)} \n",
                            f'validation-tag-image-count   : {t_n}  \n',f'validation-other-image-count : {o_n}  \n\n']
        text_file_lines.extend(calc_confusion_matrix(test_labels ,predictions, tag)) 
        text_file_lines.extend(histogram_lines(in_tag_tagged, 'in-distribution'))  
        text_file_lines.extend(histogram_lines(out_tag_tagged,'out-distribution')) 
        # generate report for ovr logistic regression model.
        generate_report(report_out_folder , tag , text_file_lines , model_name=model_type)
        # generate model pickle file.
        generate_model_file(models_out_folder, classifier, model_type, t_start, tag)

    print("[INFO] Finished.")




if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_json',    type=str,   required=True ,  default=None , help="path to the metadata json file")
    parser.add_argument('--tag_to_hash_json', type=str,   required=True ,  default=None , help="path to the tag to hash json file")
    parser.add_argument('--output',           type=str,   required=False , default=None , help="path to the output directory for models/reports.")
    parser.add_argument('--test_per',         type=float, required=False , default=0.1  , help="percentage of the test images from dataset.")

    args = parser.parse_args()

    # Run the main program 
    train_classifier(
        metadata_json = args.metadata_json,
        tag_to_hash_json= args.tag_to_hash_json,
        output_dir  = args.output, 
        test_per = args.test_per, 
        )
     
