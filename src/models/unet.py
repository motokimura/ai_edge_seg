#!/usr/bin/env python

# From https://github.com/pfnet/PaintsChainer/blob/master/cgi-bin/paint_x2_unet/unet.py

import chainer
import chainer.functions as F
import chainer.links as L

class UNet(chainer.Chain):

    def __init__(self, class_num, base_width=32, ignore_label=255):
        super(UNet, self).__init__()
        with self.init_scope():
            w = base_width
            self.c0=L.Convolution2D(3, w, 3, 1, 1)
            self.c1=L.Convolution2D(w, 2*w, 4, 2, 1)
            self.c2=L.Convolution2D(2*w, 2*w, 3, 1, 1)
            self.c3=L.Convolution2D(2*w, 4*w, 4, 2, 1)
            self.c4=L.Convolution2D(4*w, 4*w, 3, 1, 1)
            self.c5=L.Convolution2D(4*w, 8*w, 4, 2, 1)
            self.c6=L.Convolution2D(8*w, 8*w, 3, 1, 1)
            self.c7=L.Convolution2D(8*w, 16*w, 4, 2, 1)
            self.c8=L.Convolution2D(16*w, 16*w, 3, 1, 1)

            self.dc8=L.Deconvolution2D(32*w, 16*w, 4, 2, 1)
            self.dc7=L.Convolution2D(16*w, 8*w, 3, 1, 1)
            self.dc6=L.Deconvolution2D(16*w, 8*w, 4, 2, 1)
            self.dc5=L.Convolution2D(8*w, 4*w, 3, 1, 1)
            self.dc4=L.Deconvolution2D(8*w, 4*w, 4, 2, 1)
            self.dc3=L.Convolution2D(4*w, 2*w, 3, 1, 1)
            self.dc2=L.Deconvolution2D(4*w, 2*w, 4, 2, 1)
            self.dc1=L.Convolution2D(2*w, w, 3, 1, 1)
            self.dc0=L.Convolution2D(2*w, class_num, 3, 1, 1)

            self.bnc0=L.BatchNormalization(w)
            self.bnc1=L.BatchNormalization(2*w)
            self.bnc2=L.BatchNormalization(2*w)
            self.bnc3=L.BatchNormalization(4*w)
            self.bnc4=L.BatchNormalization(4*w)
            self.bnc5=L.BatchNormalization(8*w)
            self.bnc6=L.BatchNormalization(8*w)
            self.bnc7=L.BatchNormalization(16*w)
            self.bnc8=L.BatchNormalization(16*w)

            self.bnd8=L.BatchNormalization(16*w)
            self.bnd7=L.BatchNormalization(8*w)
            self.bnd6=L.BatchNormalization(8*w)
            self.bnd5=L.BatchNormalization(4*w)
            self.bnd4=L.BatchNormalization(4*w)
            self.bnd3=L.BatchNormalization(2*w)
            self.bnd2=L.BatchNormalization(2*w)
            self.bnd1=L.BatchNormalization(w)

        self._ignore_label = ignore_label

    def predict(self, x):
        e0 = F.relu(self.bnc0(self.c0(x)))
        e1 = F.relu(self.bnc1(self.c1(e0)))
        e2 = F.relu(self.bnc2(self.c2(e1)))
        del e1
        e3 = F.relu(self.bnc3(self.c3(e2)))
        e4 = F.relu(self.bnc4(self.c4(e3)))
        del e3
        e5 = F.relu(self.bnc5(self.c5(e4)))
        e6 = F.relu(self.bnc6(self.c6(e5)))
        del e5
        e7 = F.relu(self.bnc7(self.c7(e6)))
        e8 = F.relu(self.bnc8(self.c8(e7)))

        d8 = F.relu(self.bnd8(self.dc8(F.concat([e7, e8]))))
        del e7, e8
        d7 = F.relu(self.bnd7(self.dc7(d8)))
        del d8
        d6 = F.relu(self.bnd6(self.dc6(F.concat([e6, d7]))))
        del d7, e6
        d5 = F.relu(self.bnd5(self.dc5(d6)))
        del d6
        d4 = F.relu(self.bnd4(self.dc4(F.concat([e4, d5]))))
        del d5, e4
        d3 = F.relu(self.bnd3(self.dc3(d4)))
        del d4
        d2 = F.relu(self.bnd2(self.dc2(F.concat([e2, d3]))))
        del d3, e2
        d1 = F.relu(self.bnd1(self.dc1(d2)))
        del d2
        d0 = self.dc0(F.concat([e0, d1]))

        self.y = d0
        return d0

    def forward(self, x, t):
        h = self.predict(x)
        
        loss = F.softmax_cross_entropy(h, t, ignore_label=self._ignore_label)
        accuracy = F.accuracy(h, t, ignore_label=self._ignore_label)
        
        chainer.report({'loss': loss, 'accuracy': accuracy}, self)
        
        return loss
