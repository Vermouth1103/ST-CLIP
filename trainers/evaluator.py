import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix
import pandas as pd
import torch.nn.functional as F

from trainers.evaluator_build import EVALUATOR_REGISTRY


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname_list=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname_list = lab2cname_list
        self._correct_list = [0] * len(lab2cname_list)
        self._total_list = [0] * len(lab2cname_list)
        self._y_true_list = [[] for _ in range(len(lab2cname_list))]
        self._y_pred = [[] for _ in range(len(lab2cname_list))]
        self.threshold = 0.9

    def reset(self):
        self._correct_list = [0] * len(self._lab2cname_list)
        self._total_list = [0] * len(self._lab2cname_list)
        self._y_true_list = [[] for _ in range(len(self._lab2cname_list))]
        self._y_pred_list = [[] for _ in range(len(self._lab2cname_list))]

    def process(self, logits_list, label_list, device):
        for index in range(label_list.shape[1]):
            logit, pred = F.softmax(logits_list[index], -1).max(dim=1)

            if index == 0: # scene
                new_pred = list()
                for i in range(len(pred)):
                    if pred[i]!=4 and logit[i]>=self.threshold:
                        new_pred.append(pred[i])
                    else:
                        new_pred.append(4)
                pred = torch.tensor(new_pred).to(device)
            elif index == 1 or index == 2: # surface & width
                new_pred = list()
                for i in range(len(pred)):
                    if pred[i]==0 or pred[i]!=3 and logit[i]>=self.threshold:
                        new_pred.append(pred[i])
                    else:
                        new_pred.append(3)
                pred = torch.tensor(new_pred).to(device)
            elif index == 3:
                new_pred = list()
                for i in range(len(pred)):
                    if pred[i]==0 or logit[i]>=self.threshold:
                        new_pred.append(pred[i])
                    else:
                        new_pred.append(3)
                pred = torch.tensor(new_pred).to(device)

            label = label_list[:, index]
            matches = pred.eq(label).float()
            self._correct_list[index] += int(matches.sum().item())
            self._total_list[index] += label_list.shape[0]

            self._y_true_list[index].extend(label.data.cpu().numpy().tolist())
            self._y_pred_list[index].extend(pred.data.cpu().numpy().tolist())

    def evaluate(self):
        kind_list = ["scene", "surface", "width", "through"]
        pred_label_dict = dict()
        for index in range(len(self._y_true_list)):
            pred_label_dict[f"{kind_list[index]}_true"] = self._y_true_list[index]
            pred_label_dict[f"{kind_list[index]}_pred"] = self._y_pred_list[index]

        pred_label_df = pd.DataFrame(pred_label_dict)
        pred_label_df.to_csv(osp.join(self.cfg.OUTPUT_DIR, f"pred_label_{str(self.threshold)}.csv"), index=0)

        results = OrderedDict()
        acc_list = [100.0 * self._correct_list[index] / self._total_list[index] for index in range(len(self._lab2cname_list))]
        macro_f1_list = [100.0 * f1_score(
            self._y_true_list[index],
            self._y_pred_list[index],
            average="macro",
            labels=np.unique(self._y_true_list[index])
            ) for index in range(len(self._lab2cname_list))
        ]

        # The first value will be returned by trainer.test()
        for index in range(len(self._lab2cname_list)):
            results[f"accuracy_{index}"] = acc_list[index]
            results[f"macro_f1_{index}"] = macro_f1_list[index]

        output = "=> result\n"
        for index in range(len(self._lab2cname_list)):
            output += f"* total_{index}: {self._total_list[index]:,}\t"
            output += f"* correct_{index}: {self._correct_list[index]:,}\t"
            output += f"* accuracy_{index}: {acc_list[index]:.2f}%\t"
            output += f"* macro_f1_{index}: {macro_f1_list[index]:.2f}%\n"

        print(output)
        return results