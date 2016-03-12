import os
os.environ['CHAINER_TYPE_CHECK'] = "0"

import cPickle, time, glob, sys, shutil, logging, copy
import operator, numpy as np, chainer, chainer.links as L
import chainer.functions as F
from chainer import cuda, Chain, Variable, FunctionSet, optimizers, serializers
from chainer.functions.pooling import pooling_2d

###############################################################################
# Result Directory and log ####################################################
###############################################################################
result_dir = 'result_' + time.strftime('%Y-%m-%d_%H-%M-%S\\')
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

my_fname = os.path.basename(__file__)
shutil.copy(my_fname, result_dir)

log_fname = result_dir+'log.txt'
stream_log = logging.StreamHandler()
stream_log.setLevel(logging.INFO)
file_log = logging.FileHandler(filename=log_fname)
file_log.setLevel(logging.INFO)
file_log.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logging.getLogger().addHandler(stream_log)
logging.getLogger().addHandler(file_log)
logging.getLogger().setLevel(logging.INFO)


###############################################################################
# Data setup ##################################################################
###############################################################################
def load(fname):
    with open(fname, 'rb') as f:
        dic =  cPickle.load(f)
    return dic['data'], dic['labels']

train_data, train_labels = zip(*[load(f) for f in glob.glob("cifar-10-batches-py/data_batch_*")])
train_data = np.reshape(train_data, (50000, 3, 32, 32))
train_labels = np.reshape(train_labels, (50000))

test_data, test_labels = load("cifar-10-batches-py/test_batch")
test_data = np.reshape(test_data, (10000, 3, 32, 32))
test_labels = np.reshape(test_labels, (10000))


###############################################################################
# Model description ###########################################################
###############################################################################
class Dataset(object):
    def __init__(self, x, t):
        self.x, self.t = x, t
        self._size = len(self.x)
        self.perm()
        self._pos = 0

    def perm(self):
        perm = np.random.permutation(self._size)
        self.x, self.t = self.x[perm], self.t[perm]

    def feed(self, num):
        self._pos += num
        if self._pos > self._size:
            self.perm()
            self._pos = num
        return self.x[self._pos-num:self._pos].astype(np.float32), self.t[self._pos-num:self._pos]

class ModelChain(Chain):
    def __init__(self):
        super(ModelChain, self).__init__()
        class Train(): pass
        self.link_name_func_arg = []
        self.link_arg = []
        self.train = Train()
        self.clear()

    def add_Chain(self, link_name, link_func, link_arg=[], reg=True):
        link_name = link_name+"_{}".format(len(self.link_name_func_arg)+1)
        self.link_name_func_arg += [(link_name, link_func, link_arg)]
        if isinstance(link_func, chainer.link.Link):
            self.add_link(link_name, link_func)

    def clear(self):
        self.accum_count = 0
        self.accum_loss = 0.0
        self.accum_acc = 0.0

    def __getitem__(self, name):
        if name == 'loss':
            return float(self.loss.data)
        elif name == 'acc':
            return float(self.acc.data)
        elif name == 'accum_loss':
            return float(self.accum_loss)
        elif name == 'accum_acc':
            return float(self.accum_acc)

    def __call__(self, x, t, train=False, gpu=False):
        if gpu is True:
            x, t = cuda.to_gpu(x), cuda.to_gpu(t)
        x, t = Variable(x, volatile=not train), Variable(t, volatile=not train)

        for n, f, a in self.link_name_func_arg:
            arg = copy.copy(a)
            if self.train in a:
                arg[arg.index(self.train)] = train
            #print n, f, x.data.shape, a
            x = f(*([x]+arg))

        self.loss = F.softmax_cross_entropy(x, t)
        self.acc = F.accuracy(x, t)

        self.accum_count += 1
        self.accum_loss = self.accum_loss*(1.0-1.0/self.accum_count) + self.loss.data*(1.0/self.accum_count)
        self.accum_acc = self.accum_acc*(1.0-1.0/self.accum_count) + self.acc.data*(1.0/self.accum_count)
        return self.loss

class ResNet(ModelChain):
    def __init__(self, n=18):
        super(ResNet, self).__init__()

        class ResBlk(Chain):
            def __init__(self, n_in, n_out, stride=1, ksize=1):
                super(ResBlk, self).__init__(
                    conv1=L.Convolution2D(n_in, n_out, 3, stride=stride, pad=1, wscale=1.0/1.41421356),
                    bn1=L.BatchNormalization(n_out),
                    conv2=L.Convolution2D(n_out, n_out, 3, stride=1, pad=1, wscale=1.0/1.41421356),
                    bn2=L.BatchNormalization(n_out),
                )

            def __call__(self, x, train):
                h = F.relu(self.bn1(self.conv1(x), test=not train))
                h = self.bn2(self.conv2(h), test=not train)
                if x.data.shape != h.data.shape:
                    xp = cuda.get_array_module(x.data)
                    n, c, hh, ww = x.data.shape
                    pad_c = h.data.shape[1] - c
                    p = xp.zeros((n, pad_c, hh, ww), dtype=xp.float32)
                    p = Variable(p, volatile=not train)
                    x = F.concat((p, x))
                    if x.data.shape[2:] != h.data.shape[2:]:
                        x = F.average_pooling_2d(x, 1, 2)
                return F.relu(h + x)

        self.add_Chain('conv', L.Convolution2D(3, 16, 3, stride=1, pad=1, wscale=1.41421356))
        self.add_Chain('bn  ', L.BatchNormalization(16))
        [self.add_Chain('resn', ResBlk(16, 16), [self.train]) for i in range(n)]
        self.add_Chain('resn', ResBlk(16, 32, stride=2), [self.train])
        [self.add_Chain('resn', ResBlk(32, 32), [self.train]) for i in range(n)]
        self.add_Chain('resn', ResBlk(32, 64, stride=2), [self.train])
        [self.add_Chain('resn', ResBlk(64, 64), [self.train]) for i in range(n)]
        self.add_Chain('avgp', F.average_pooling_2d, [8], reg=False)
        self.add_Chain('fc  ', L.Linear(64, 10))

class VGG(ModelChain):
    def __init__(self):
        super(VGG, self).__init__()
        self.add_Chain('conv', L.Convolution2D(3, 64, 3, pad=1))
        self.add_Chain('bn  ', L.BatchNormalization(64))
        self.add_Chain('relu', F.relu, reg=False)
        self.add_Chain('drop', F.dropout, [0.3, self.train], reg=False)
        self.add_Chain('conv', L.Convolution2D(64, 64, 3, pad=1))
        self.add_Chain('maxp', F.max_pooling_2d, [2, 2], reg=False)

        self.add_Chain('conv', L.Convolution2D(64, 128, 3, pad=1))
        self.add_Chain('bn  ', L.BatchNormalization(128))
        self.add_Chain('relu', F.relu, reg=False)
        self.add_Chain('drop', F.dropout, [0.4, self.train], reg=False)
        self.add_Chain('conv', L.Convolution2D(128, 128, 3, pad=1))
        self.add_Chain('maxp', F.max_pooling_2d, [2, 2], reg=False)

        self.add_Chain('conv', L.Convolution2D(128, 256, 3, pad=1))
        self.add_Chain('bn  ', L.BatchNormalization(256))
        self.add_Chain('relu', F.relu, reg=False)
        self.add_Chain('drop', F.dropout, [0.4, self.train], reg=False)
        self.add_Chain('conv', L.Convolution2D(256, 256, 3, pad=1))
        self.add_Chain('bn  ', L.BatchNormalization(256))
        self.add_Chain('relu', F.relu, reg=False)
        self.add_Chain('drop', F.dropout, [0.4, self.train], reg=False)
        self.add_Chain('conv', L.Convolution2D(256, 256, 3, pad=1))
        self.add_Chain('bn  ', L.BatchNormalization(256))
        self.add_Chain('relu', F.relu, reg=False)
        self.add_Chain('maxp', F.max_pooling_2d, [2, 2], reg=False)

        self.add_Chain('conv', L.Convolution2D(256, 512, 3, pad=1))
        self.add_Chain('bn  ', L.BatchNormalization(512))
        self.add_Chain('relu', F.relu, reg=False)
        self.add_Chain('drop', F.dropout, [0.4, self.train], reg=False)
        self.add_Chain('conv', L.Convolution2D(512, 512, 3, pad=1))
        self.add_Chain('bn  ', L.BatchNormalization(512))
        self.add_Chain('relu', F.relu, reg=False)
        self.add_Chain('drop', F.dropout, [0.4, self.train], reg=False)
        self.add_Chain('conv', L.Convolution2D(512, 512, 3, pad=1))
        self.add_Chain('bn  ', L.BatchNormalization(512))
        self.add_Chain('relu', F.relu, reg=False)
        self.add_Chain('maxp', F.max_pooling_2d, [2, 2], reg=False)

        self.add_Chain('conv', L.Convolution2D(512, 512, 3, pad=1))
        self.add_Chain('bn  ', L.BatchNormalization(512))
        self.add_Chain('relu', F.relu, reg=False)
        self.add_Chain('drop', F.dropout, [0.4, self.train], reg=False)
        self.add_Chain('conv', L.Convolution2D(512, 512, 3, pad=1))
        self.add_Chain('bn  ', L.BatchNormalization(512))
        self.add_Chain('relu', F.relu, reg=False)
        self.add_Chain('drop', F.dropout, [0.4, self.train], reg=False)
        self.add_Chain('conv', L.Convolution2D(512, 512, 3, pad=1))
        self.add_Chain('bn  ', L.BatchNormalization(512))
        self.add_Chain('relu', F.relu, reg=False)
        self.add_Chain('maxp', F.max_pooling_2d, [2, 2], reg=False)

        self.add_Chain('conv', L.Convolution2D(512, 512, 3, pad=1))
        self.add_Chain('bn  ', L.BatchNormalization(512))
        self.add_Chain('relu', F.relu, reg=False)
        self.add_Chain('drop', F.dropout, [0.4, self.train], reg=False)
        self.add_Chain('conv', L.Convolution2D(512, 512, 3, pad=1))
        self.add_Chain('bn  ', L.BatchNormalization(512))
        self.add_Chain('relu', F.relu, reg=False)
        self.add_Chain('drop', F.dropout, [0.5, self.train], reg=False)
        self.add_Chain('conv', L.Convolution2D(512, 512, 3, pad=1))
        self.add_Chain('bn  ', L.BatchNormalization(512))
        self.add_Chain('relu', F.relu, reg=False)
        self.add_Chain('maxp', F.max_pooling_2d, [2, 2], reg=False)

        self.add_Chain('fc  ', F.Linear(512, 512))
        self.add_Chain('bn  ', L.BatchNormalization(512))
        self.add_Chain('relu', F.relu, reg=False)
        self.add_Chain('drop', F.dropout, [0.5, self.train], reg=False)

        self.add_Chain('fc  ', F.Linear(512, 10))


class Cifar10(ModelChain):
    def __init__(self):
        super(Cifar10, self).__init__()
        self.add_Chain('conv', L.Convolution2D(3, 32, 5, stride=1, pad=2))
        self.add_Chain('relu', F.relu, reg=False)
        self.add_Chain('maxpool', F.max_pooling_2d, [3, 2], reg=False)
        self.add_Chain('conv', L.Convolution2D(32, 32, 5, stride=1, pad=2))
        self.add_Chain('relu', F.relu, reg=False)
        self.add_Chain('maxpool', F.max_pooling_2d, [3, 2], reg=False)
        self.add_Chain('conv', L.Convolution2D(32, 64, 5, stride=1, pad=2))
        self.add_Chain('relu', F.relu, reg=False)
        self.add_Chain('sppool', F.spatial_pyramid_pooling_2d, [3, F.MaxPooling2D], reg=False)
        self.add_Chain('fc', F.Linear(1344, 4096))
        self.add_Chain('relu', F.relu, reg=False)
        self.add_Chain('drop', F.dropout, [0.5, self.train], reg=False)
        self.add_Chain('fc', F.Linear(4096, 10))


###############################################################################
# Configuration ###############################################################
###############################################################################
n_epoch = 500
batchsize = 128
gpu_Enable = True

model = ResNet(9)
#model = VGG()
#model = Cifar10()
if gpu_Enable:
    model.to_gpu()


model_fname = result_dir+"model_e%d.hdf5"
optimizer_fname = result_dir+"opt_e%d.hdf5"

N = 50000
N_test = len(test_data)
optimizer = optimizers.MomentumSGD(lr=0.1, momentum=0.9)
#optimizer = optimizers.RMSprop(0.001)
#optimizer = optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

train_set = Dataset(train_data, train_labels)
test_set = Dataset(test_data, test_labels)


###############################################################################
# Run training ################################################################
###############################################################################
logging.info("Start training")

for epoch in xrange(1, n_epoch+1):
    logging.info('----- epoch:{}/{} ({}-{}itr) batchsize:{} -----'.format(
    epoch, n_epoch, (epoch-1)*N / batchsize, epoch*N / batchsize, batchsize))

    # training #
    model.clear()
    start_time = time.time()
    for i in xrange(0, N, batchsize):
        x, t = train_set.feed(batchsize)
        optimizer.update(model, x, t, train=True, gpu=True)

        if i > 0:
            print "Now...{:2.2f}% time:{:.1f}s elapse:{:.1f}s duration:{:.2f}s acc{:.2f}%\r".format(
            100.0 * i / N,
            time.time() - start_time,
            (time.time() - start_time) / (float(i) / N),
            (time.time() - start_time) / (i/batchsize),
            model['acc'] * 100.0
            ),

    epoch_time = time.time() - start_time
    print ""
    logging.info('train loss:{:.3f} accuracy:{:.3f} epoch time:{:.1f} duration:{:.3f}'.format(
        model['accum_loss'], model['accum_acc'], float(epoch_time), float(epoch_time) / (N/batchsize)))

    # Save #
    serializers.save_hdf5(model_fname%epoch, model)
    serializers.save_hdf5(optimizer_fname%epoch, optimizer)

    # testing #
    model.clear()
    for i in xrange(0, N_test, batchsize):
        x, t = test_set.feed(batchsize)
        model(x, t, train=False, gpu=True)

    logging.info('test loss:{:.3f} accuracy:{:.3f}'.format(model['accum_loss'], model['accum_acc']))


