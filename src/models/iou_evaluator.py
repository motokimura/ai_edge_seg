import chainer
from chainer import reporter as reporter_module
from chainer.training import extensions
from chainer import function
import numpy as np
import copy

class IouEvaluator(extensions.Evaluator):

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

        and_count = 0.
        or_count = 0.

        for batch in it:
            observation = {}

            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                with function.no_backprop_mode():
                    if isinstance(in_arrays, tuple):
                        eval_func(*in_arrays)
                        ac, oc = self.iou(in_arrays)
                    elif isinstance(in_arrays, dict):
                        eval_func(**in_arrays)
                        ac, oc = self.iou(in_arrays)
                    else:
                        eval_func(in_arrays)
                        ac, oc = self.iou(in_arrays)
                    and_count = and_count + ac
                    or_count = or_count + oc

            # print(observation)
            summary.add(observation)

        iou_observation = {}
        if(or_count == 0):
            iou_observation['iou'] = 0.
        else:
            iou_observation['iou'] = float(and_count) / or_count
        summary.add(iou_observation)

        return summary.compute_mean()

    def iou(self, in_arrays):
        model = self._targets['main']

        _, labels = in_arrays
        #if self.device >= 0:
        #    labels = chainer.cuda.to_cpu(labels)
        labels = chainer.cuda.to_cpu(labels)

        y = model.y.data
        #if self.device >= 0:
        #    y = chainer.cuda.to_cpu(y)
        y = chainer.cuda.to_cpu(y)
        # print(y)
        y = y.argmax(axis=1)

        # print('labels', labels)
        # print('predct', y)
        and_count = (labels & y).sum()
        or_count = (labels | y).sum()
        return and_count, or_count