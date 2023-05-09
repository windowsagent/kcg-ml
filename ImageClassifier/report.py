
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

