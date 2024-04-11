import os
import pickle
import re
import random
import os.path as osp
from collections import defaultdict
import gdown
import json
import numpy as np

from trainers.dataset_build import DATASET_REGISTRY
from dassl.utils import read_json, mkdir_if_missing, check_isfile


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label_list (list(int)): list of class label.
        classname_list (list(str)): list of class name.
    """

    def __init__(self, impath="", label_list=[0], classname_list=[""], traj_property=[]):
        assert isinstance(impath, str)
        assert check_isfile(impath)

        self._impath = impath
        self._label_list = label_list
        self._classname_list = classname_list
        self._traj_property = traj_property

    @property
    def impath(self):
        return self._impath

    @property
    def label_list(self):
        return self._label_list

    @property
    def classname_list(self):
        return self._classname_list
    
    @property
    def traj_property(self):
        return self._traj_property


class DatasetBase:
    """A unified dataset class for
    1) domain adaptation
    2) domain generalization
    3) semi-supervised learning
    """

    dataset_dir = ""  # the directory where the dataset is stored

    def __init__(self, train_x=None, train_u=None, val=None, test=None):
        self._train_x = train_x  # labeled training data
        self._train_u = train_u  # unlabeled training data (optional)
        self._val = val  # validation data (optional)
        self._test = test  # test data
        self._num_class_list = self.get_num_class_list(train_x)
        self._lab2cname_list, self._classname_list = self.get_lab2cname_list(train_x)

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_u(self):
        return self._train_u

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def lab2cname_list(self):
        return self._lab2cname_list

    @property
    def classname_list(self):
        return self._classname_list

    @property
    def num_class_list(self):
        return self._num_class_list

    @staticmethod
    def get_num_class_list(data_source):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        label_dict = defaultdict(set)
        for item in data_source:
            for index in range(len(item.label_list)):
                label_dict[index].add(item.label_list[index])
        label_list = [max(label_set) + 1 for label_set in list(label_dict.values())]
        return label_list

    @staticmethod
    def get_lab2cname_list(data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        container = defaultdict(set)
        for item in data_source:
            for index in range(len(item.label_list)):
                container[index].add((item.label_list[index], item.classname_list[index]))
        
        mapping_list = list()
        for index in range(len(container)):
            mapping = {label: classname for label, classname in container[index]}
            mapping_list.append(mapping)

        classname_list = []
        for index, mapping in enumerate(mapping_list):
            labels = list(mapping.keys())
            labels.sort()
            classname_list.append([mapping[label] for label in labels])
        return mapping_list, classname_list


    def generate_fewshot_dataset(
        self, *data_sources, num_shots=-1, repeat=False
    ):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a small number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed (default: False).
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f"Creating a {num_shots}-shot dataset")

        output = []

        for data_source in data_sources:
            tracker_list = self.split_dataset_by_label(data_source)
            dataset = []

            for index, tracker in enumerate(tracker_list):
                for label, items in tracker.items():
                    if len(items) >= num_shots:
                        sampled_items = random.sample(items, num_shots)
                    else:
                        if repeat:
                            sampled_items = random.choices(items, k=num_shots)
                        else:
                            sampled_items = items
                    dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output

    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = list()

        for item in data_source:
            for index in range(len(item.label_list)):
                if len(output) < index + 1:
                    output.append(defaultdict(list))
                output[index][item.label_list[index]].append(item)

        return output


@DATASET_REGISTRY.register()
class CHENGDU_Dataset(DatasetBase):

    dataset_dir = "chengdu_dataset"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "dataset_split.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        self.link_profile_dict_path = os.path.join(self.dataset_dir, "link_profile_dict.json")
        self.link_traj_dict_path = os.path.join(self.dataset_dir, "link_traj_dict.json")

        with open(self.link_profile_dict_path, "r") as f:
            link_property_dict = json.load(f)
        with open(self.link_traj_dict_path, "r") as f:
            link_traj_dict = json.load(f)

        train, val, test = self.read_split(self.split_path, link_property_dict, link_traj_dict)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as f:
                    data = pickle.load(f)
                    train, val = data["train"], data["val"]
            
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        super().__init__(train_x=train, val=val, test=test)

    def read_split(self, filepath, link_property_dict, link_traj_dict):
        def _convert(items):
            out = []
            for impath, label_list, classname_list in items:
                img = impath.split("/")[-1]
                label_list = np.array([int(item) for item in label_list])
                link_traj = link_traj_dict[img]
                link_property = np.array(link_property_dict[str(link_traj["link_id"])])
                prev_link_property = np.array(link_property_dict[str(link_traj["prev_link_id"])])
                next_link_property = np.array(link_property_dict[str(link_traj["next_link_id"])])
                traj_property = np.array([prev_link_property, link_property, next_link_property])
                item = Datum(impath=impath, label_list=label_list, classname_list=classname_list, traj_property=traj_property)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test

