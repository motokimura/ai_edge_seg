import chainer
from chainer import reporter as reporter_module
from chainer.training import extensions
from chainer import function
import numpy as np
import copy
from chainercv.evaluations import eval_semantic_segmentation

class IouEvaluator(extensions.Evaluator):

    def __init__(self, iterator, target, device, label_names=None):
        super(IouEvaluator, self).__init__(
            iterator, target, device=device)
        self.device_id = device
        self.label_names = label_names
    
    def evaluate(self):
        iterator = self._iterators['main']
        model = self._targets['main']
        eval_func = self.eval_func or model

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        labels_all = []
        preds_all = []

        for batch in it:
            observation = {}

            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                with function.no_backprop_mode():
                    if isinstance(in_arrays, tuple):
                        eval_func(*in_arrays)
                    elif isinstance(in_arrays, dict):
                        eval_func(**in_arrays)
                    else:
                        eval_func(in_arrays)
                    
                    _, labels = in_arrays
                    if self.device_id >= 0:
                        labels = chainer.cuda.to_cpu(labels)

                    y = model.y.data
                    if self.device_id >= 0:
                        y = chainer.cuda.to_cpu(y)
                    preds = y.argmax(axis=1)
                    
                    labels_all.append(labels)
                    preds_all.append(preds)

            # print(observation)
            summary.add(observation)
        
        ss_eval = eval_semantic_segmentation(preds_all, labels_all)
        iou = ss_eval['iou'][1:] # Assuming label '0' is assigned for background

        iou_observation = {}
        iou_observation['miou'] = np.nanmean(iou)
        for i, label_name in enumerate(self.label_names):
            iou_observation['iou/{:s}'.format(label_name)] = iou[i]
        summary.add(iou_observation)

        return summary.compute_mean()

    def iou(self, in_arrays):
        model = self._targets['main']

        _, labels = in_arrays
        if self.device_id >= 0:
            labels = chainer.cuda.to_cpu(labels)

        y = model.y.data
        if self.device_id >= 0:
            y = chainer.cuda.to_cpu(y)
        # print(y)
        y = y.argmax(axis=1)

        # print('labels', labels)
        # print('predct', y)
        and_count = (labels & y).sum()
        or_count = (labels | y).sum()
        return and_count, or_count