# ST-CLIP

This is the official code repository for ST-CLIP. At this stage, we have provided detailed descriptions of the data format for reference and reproducibility. After the paper is accepted, we will release a portion of the anonymized image data.

- **[train.py]** is the entry point to run the model.

- **[clip]** contains the corresponding files for the clip model, which can be called directly from the model.
  
- **[configs]** contains the dataset config file and the trainer config file, which set the corresponding parameters. Different config files need to be specified when training the model.

- **[datasets]** contains the Dataset class.

- **[scripts]** contains the model training scripts with examples.

- **[trainers]** Contains the body of the model, which is defined in stclip.py and supported in the rest of the files.。

- **[data]** contains 4 json files in total, which are *record.json*, *dataset.json*, *link_profile_dict.json* and *link_traj_dict.json*. The contents of each json file are as follows:
  - record.json:
    
    Full dataset. The only one key is "dataset". The value is a list, each item is a data item. Each data item is a list, containing three items: *image_path*, *label_list*, *label_name_list*. 
  
    The label_list and label_name_list are shown as examples: Suppose there are scene, surface, width and accessibilitys four aspects, then the length of label_list and label_name_list is 4; Suppose that through has three categories: *easy*, *hard*, and *extremely hard*, and the true value of the image is extremely hard. Then label_list is [\*, \*, \*, 2], label_name_list[\*, \*, \*, extremely hard].
  - dataset.json:

    Partitioned dataset. It contains three keys: *train*, *val*, and *test*. Each value is a dataset as described above.

  - link_profile_dict.json
   
    Road segment properties. The key is road IDs, and the value is the corresponding attribute feature.

  - link_traj_dict.json

    The trajectory of road segments. The key is the name of images, and the value is the corresponding road segment-based trajectory.