#!/usr/bin/env python

# Implementation of "Deep Residual U-Net"
# arXiv-link: https://arxiv.org/abs/1711.10684

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers

class Block(chainer.Chain):

    def __init__(self, in_ch, n_filters, strides, no_bn1=False, initialW=None, residual=True):
        super(Block, self).__init__()
        if initialW is None:
            initialW = initializers.HeNormal()

        with self.init_scope():
            # Residual path
            if no_bn1:
                self.bn1 = None
            else:
                self.bn1 = L.BatchNormalization(in_ch)
            self.conv1 = L.Convolution2D(
				in_ch, n_filters[0], 3, strides[0], 1, initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(n_filters[0])
            self.conv2 = L.Convolution2D(
				n_filters[0], n_filters[1], 3, strides[1], 1, initialW=initialW, nobias=True)
            
            if residual:
                # Shortcut path
                self.conv3 = L.Convolution2D(
                    in_ch, n_filters[1], 1, strides[0], 0, initialW=initialW, nobias=True)
                self.bn3 = L.BatchNormalization(n_filters[1])
            
        self._residual = residual

    def forward(self, x):
        # Residual path
        if self.bn1 is None:
            h = x
        else:
            h = F.relu(self.bn1(x))
        h = self.conv1(h)
        h = self.conv2(F.relu(self.bn2(h)))

        if self._residual:
            # Shortcut path
            ## In the paper, identity mapping is used for the shortcut path, 
            ## but it seems impossible because input `x` shape is always different from residual `h` shape.
            ## So in the shortcut path, I applied 1x1 convolution to input `x` with the same stride used in the residual path.
            h = h + self.bn3(self.conv3(x))
        
        return h

class UNet(chainer.Chain):

    def __init__(self, class_num, train_wh, test_wh, base_w=64, ignore_label=255, initialW=None, residual=True):
        super(UNet, self).__init__()

        if initialW is None:
            initialW = initializers.HeNormal()

        with self.init_scope():
            # Encoder
            self.e0 = Block(3, [base_w, base_w], [1, 1], True, initialW, residual)
            self.e1 = Block(base_w, [2*base_w, 2*base_w], [2, 1], False, initialW, residual)
            self.e2 = Block(2*base_w, [4*base_w, 4*base_w], [2, 1], False, initialW, residual)
            
            # Bridge
            self.bridge = Block(4*base_w, [8*base_w, 8*base_w], [2, 1], False, initialW, residual)
            
            # Decoder
            self.d2 = Block(12*base_w, [4*base_w, 4*base_w], [1, 1], False, initialW, residual)
            self.d1 = Block((4+2)*base_w, [2*base_w, 2*base_w], [1, 1], False, initialW, residual)
            self.d0 = Block((2+1)*base_w, [base_w, base_w], [1, 1], False, initialW, residual)

            # Classifier
            self.conv = L.Convolution2D(
                base_w, class_num, 1, 1, 0, initialW=initialW, nobias=False)
        
        self._train_wh = train_wh
        self._test_wh = test_wh
        self._ignore_label = ignore_label

    def predict(self, x):
        in_w, in_h = self._train_wh if chainer.config.train else self._test_wh

        # Encoder
        e0_out = self.e0(x)
        e1_out = self.e1(e0_out)
        e2_out = self.e2(e1_out)
        
        # Bridge
        h = self.bridge(e2_out)
        
        # Decoder
        h = F.concat([F.resize_images(h, (in_h//4, in_w//4)), e2_out])
        del e2_out
        h = self.d2(h)

        h = F.concat([F.resize_images(h, (in_h//2, in_w//2)), e1_out])
        del e1_out
        h = self.d1(h)

        h = F.concat([F.resize_images(h, (in_h, in_w)), e0_out])
        del e0_out
        h = self.d0(h)

        # Classifier
        h = self.conv(h)

        return h

    def forward(self, x, t):
        h = self.predict(x)
        
        ## In the original paper (for binary category classification task), 
        ## MSE loss is computed from the sigmoid activation of `h`, 
        ## but I use SCE loss to apply this model for multi-class classification task.
        loss = F.softmax_cross_entropy(h, t, ignore_label=self._ignore_label)
        accuracy = F.accuracy(h, t, ignore_label=self._ignore_label)
        chainer.report({'loss': loss, 'accuracy': accuracy}, self)

        return loss