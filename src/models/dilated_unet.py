#!/usr/bin/env python

# From https://github.com/lyakaap/Kaggle-Carvana-3rd-Place-Solution/blob/master/model_pytorch.py

import chainer
import chainer.functions as F
import chainer.links as L

class DilatedUNet(chainer.Chain):

    def __init__(self, class_num, base_width=44, bn=False, ignore_label=255):
        super(DilatedUNet, self).__init__()
        with self.init_scope():
            w = base_width

            # Encoder's conv layers
            self.e0 = L.Convolution2D(3, w, 3, 1, 1)
            self.e1 = L.Convolution2D(w, w, 3, 1, 1)
            ## pooling x 1/2
            self.e2 = L.Convolution2D(w, 2*w, 3, 1, 1)
            self.e3 = L.Convolution2D(2*w, 2*w, 3, 1, 1)
            ## pooling x 1/2
            self.e4 = L.Convolution2D(2*w, 4*w, 3, 1, 1)
            self.e5 = L.Convolution2D(4*w, 4*w, 3, 1, 1)
            ## pooling x 1/2

            if bn:
                self.bne0 = L.BatchNormalization(w)
                self.bne1 = L.BatchNormalization(w)
                self.bne2 = L.BatchNormalization(2*w)
                self.bne3 = L.BatchNormalization(2*w)
                self.bne4 = L.BatchNormalization(4*w)
                self.bne5 = L.BatchNormalization(4*w)

            # Bottleneck's dilated conv layers
            self.b0 = L.DilatedConvolution2D(4*w, 8*w, 3, 1, pad=1,  dilate=1 )
            self.b1 = L.DilatedConvolution2D(8*w, 8*w, 3, 1, pad=2,  dilate=2 )
            self.b2 = L.DilatedConvolution2D(8*w, 8*w, 3, 1, pad=4,  dilate=4 )
            self.b3 = L.DilatedConvolution2D(8*w, 8*w, 3, 1, pad=8,  dilate=8 )
            self.b4 = L.DilatedConvolution2D(8*w, 8*w, 3, 1, pad=16, dilate=16)
            self.b5 = L.DilatedConvolution2D(8*w, 8*w, 3, 1, pad=32, dilate=32)

            if bn:
                self.bnb0 = L.BatchNormalization(8*w)
                self.bnb1 = L.BatchNormalization(8*w)
                self.bnb2 = L.BatchNormalization(8*w)
                self.bnb3 = L.BatchNormalization(8*w)
                self.bnb4 = L.BatchNormalization(8*w)
                self.bnb5 = L.BatchNormalization(8*w)

            # Decoder's conv layers
            self.up6 = L.Deconvolution2D(8*w, 4*w, 4, 2, 1)
            ## concat
            self.d5 = L.Convolution2D(8*w, 4*w, 3, 1, 1)
            self.d4 = L.Convolution2D(4*w, 4*w, 3, 1, 1)
            self.up4 = L.Deconvolution2D(4*w, 2*w, 4, 2, 1)
            ## concat
            self.d3 = L.Convolution2D(4*w, 2*w, 3, 1, 1)
            self.d2 = L.Convolution2D(2*w, 2*w, 3, 1, 1)
            self.up2 = L.Deconvolution2D(2*w, w, 4, 2, 1)
            ## concat
            self.d1 = L.Convolution2D(2*w, w, 3, 1, 1)
            self.d0 = L.Convolution2D(w, w, 3, 1, 1)

            if bn:
                self.bnup6 = L.BatchNormalization(4*w)
                self.bnup4 = L.BatchNormalization(2*w)
                self.bnup2 = L.BatchNormalization(w)

                self.bnd5 = L.BatchNormalization(4*w)
                self.bnd4 = L.BatchNormalization(4*w)
                self.bnd3 = L.BatchNormalization(2*w)
                self.bnd2 = L.BatchNormalization(2*w)
                self.bnd1 = L.BatchNormalization(w)
                self.bnd0 = L.BatchNormalization(w)

            # Classifier
            self.last = L.Convolution2D(w, class_num, 3, 1, 1)
        
        self._bn = bn
        self._ignore_label = ignore_label

    def predict(self, x):

        if self._bn:
            # Encoder
            e0_out = F.relu(self.bne0(self.e0(x)))
            e1_out = F.relu(self.bne1(self.e1(e0_out)))
            del e0_out
            e1_out_pool = F.max_pooling_2d(e1_out, 2)

            e2_out = F.relu(self.bne2(self.e2(e1_out_pool)))
            del e1_out_pool
            e3_out = F.relu(self.bne3(self.e3(e2_out)))
            e3_out_pool = F.max_pooling_2d(e3_out, 2)

            e4_out = F.relu(self.bne4(self.e4(e3_out_pool)))
            del e3_out_pool
            e5_out = F.relu(self.bne5(self.e5(e4_out)))
            e5_out_pool = F.max_pooling_2d(e5_out, 2)

            # Bottleneck
            b0_out = F.relu(self.bnb0(self.b0(e5_out_pool)))
            del e5_out_pool
            b_out = b0_out
            b1_out = F.relu(self.bnb1(self.b1(b0_out)))
            del b0_out
            b_out += b1_out
            b2_out = F.relu(self.bnb2(self.b2(b1_out)))
            del b1_out
            b_out += b2_out
            b3_out = F.relu(self.bnb3(self.b3(b2_out)))
            del b2_out
            b_out += b3_out
            b4_out = F.relu(self.bnb4(self.b4(b3_out)))
            del b3_out
            b_out += b4_out
            b5_out = F.relu(self.bnb5(self.b5(b4_out)))
            del b4_out
            b_out += b5_out

            # Decoder
            up6_out = F.relu(self.bnup6(self.up6(b_out)))
            del b_out
            d5_out = F.relu(self.bnd5(self.d5(F.concat([e5_out, up6_out]))))
            del e5_out, up6_out
            d4_out = F.relu(self.bnd4(self.d4(d5_out)))
            del d5_out
            up4_out = F.relu(self.bnup4(self.up4(d4_out)))
            del d4_out
            d3_out = F.relu(self.bnd3(self.d3(F.concat([e3_out, up4_out]))))
            del e3_out, up4_out
            d2_out = F.relu(self.bnd2(self.d2(d3_out)))
            del d3_out
            up2_out = F.relu(self.bnup2(self.up2(d2_out)))
            del d2_out
            d1_out = F.relu(self.bnd1(self.d1(F.concat([e1_out, up2_out]))))
            del e1_out, up2_out
            d0_out = F.relu(self.bnd0(self.d0(d1_out)))
            del d1_out

        else:
            # Encoder
            e0_out = F.relu(self.e0(x))
            e1_out = F.relu(self.e1(e0_out))
            del e0_out
            e1_out_pool = F.max_pooling_2d(e1_out, 2)

            e2_out = F.relu(self.e2(e1_out_pool))
            del e1_out_pool
            e3_out = F.relu(self.e3(e2_out))
            e3_out_pool = F.max_pooling_2d(e3_out, 2)

            e4_out = F.relu(self.e4(e3_out_pool))
            del e3_out_pool
            e5_out = F.relu(self.e5(e4_out))
            e5_out_pool = F.max_pooling_2d(e5_out, 2)

            # Bottleneck
            b0_out = F.relu(self.b0(e5_out_pool))
            del e5_out_pool
            b_out = b0_out
            b1_out = F.relu(self.b1(b0_out))
            del b0_out
            b_out += b1_out
            b2_out = F.relu(self.b2(b1_out))
            del b1_out
            b_out += b2_out
            b3_out = F.relu(self.b3(b2_out))
            del b2_out
            b_out += b3_out
            b4_out = F.relu(self.b4(b3_out))
            del b3_out
            b_out += b4_out
            b5_out = F.relu(self.b5(b4_out))
            del b4_out
            b_out += b5_out

            # Decoder
            up6_out = self.up6(b_out) # No activation???
            del b_out
            d5_out = F.relu(self.d5(F.concat([e5_out, up6_out])))
            del e5_out, up6_out
            d4_out = F.relu(self.d4(d5_out))
            del d5_out
            up4_out = self.up4(d4_out) # No activation???
            del d4_out
            d3_out = F.relu(self.d3(F.concat([e3_out, up4_out])))
            del e3_out, up4_out
            d2_out = F.relu(self.d2(d3_out))
            del d3_out
            up2_out = self.up2(d2_out) # No activation???
            del d2_out
            d1_out = F.relu(self.d1(F.concat([e1_out, up2_out])))
            del e1_out, up2_out
            d0_out = F.relu(self.d0(d1_out))
            del d1_out

        last_out = self.last(d0_out)
        del d0_out
        
        self.y = last_out
        return last_out

    def forward(self, x, t):
        h = self.predict(x)
        
        loss = F.softmax_cross_entropy(h, t, ignore_label=self._ignore_label)
        accuracy = F.accuracy(h, t, ignore_label=self._ignore_label)
        
        chainer.report({'loss': loss, 'accuracy': accuracy}, self)
        
        return loss
