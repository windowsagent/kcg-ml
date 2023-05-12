
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



  def generate_report(
                        reports_output_folder : str,
                        tag_name : str, 
                        text_file_lines : List[str], 
                        model_name: str,
                        ):
        """generate text file with text file lines provided, 
        save it in output directory.
        :param reports_output_folder: output folder for saving report file.
        :type reports_output_folder: str
        :param tag_name: name of the classifer tag.
        :type tag_name: str
        :param model_name: name of the model .
        :type  model_name: str
        :rtype: None. 
        """

        model_file_name = f'model-report-{model_name}-tag-{tag_name}'
        text_file_path = os.path.join(reports_output_folder ,f'{model_file_name}.txt' )
        with open( text_file_path ,"w+", encoding="utf-8") as f:
            f.writelines(text_file_lines)
        f.close()

        return



    
    def calc_confusion_matrix(
                            test_labels , 
                            predictions ,
                            tag_name : str 
                            ):
        """calculate accuracy, confusion matrix parts and return them.
        :param test_labels: labels for the test embeddings.
        :type test_labels: NdArray
        :param predictions: prediction from the classifer for the test_labels.
        :type predictions: NdArray
        :returns: accuracy,false positive rate, false negative rate, true positive rate, \
                true negative rate, false positive, false negative, true positive, true negative.
        :rtype: list of strings  
        """
        accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
        confusion_matrix = metrics.confusion_matrix(test_labels, predictions)
        FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
        FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
        TP = np.diag(confusion_matrix)
        TN = confusion_matrix.sum() - (FP + FN + TP)
        ALL_SUM = FP + FN + TP + TN
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        return [ 
                f'false-positive-rate: {FPR[0] :.4f}  \n', 
                f'false-negative-rate: {FNR[0] :.4f}  \n',
                f'true-positive-rate : {TPR[0] :.4f}  \n',
                f'true-negative-rate : {TNR[0] :.4f}  \n\n',
                f'false-positive :  {FP[0]} out of {ALL_SUM[0]}  \n',
                f'false-negative : {FN[0]}  out of {ALL_SUM[0]} \n',
                f'true-positive : {TP[0]} out of {ALL_SUM[0]}  \n',
                f'true-negative : {TN[0]} out of {ALL_SUM[0]}  \n\n',
                f'>Accuracy : {accuracy:.4f}\n\n',
                f"Classification Report : \n\n{metrics.classification_report(test_labels, predictions)}\n\n",
                f"Index 0 is class {tag_name}\n",
                "Index 1 is class other \n\n"
                ]