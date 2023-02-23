import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter

_momentum_update_parameter = ops.MultitypeFuncGraph("momentum_update_parameter")


@_momentum_update_parameter.register("Tensor", "Tensor", "Tensor")
def momentum_update_parameter(m, encoder_k, encoder_q):
    value = ops.assign(encoder_k, encoder_k * m + encoder_q * (1.0 - m))
    return value


class MoCo(nn.Cell):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, rank_id=0, device_num=1):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.classifier.weight.shape[1]
            self.encoder_q.classifier = nn.SequentialCell([
                nn.Dense(dim_mlp, dim_mlp),  # fixme: weight init???
                nn.ReLU(),
                self.encoder_q.classifier
            ])
            self.encoder_k.classifier = nn.SequentialCell([
                nn.Dense(dim_mlp, dim_mlp),
                nn.ReLU(),
                self.encoder_k.classifier
            ])

        # `get_parameters` invoke `yield` which is not supported in `construct`, we have to cache parameters of encoder
        self.params_q = list(self.encoder_q.get_parameters())
        self.params_k = list(self.encoder_k.get_parameters())
        for param_q, param_k in zip(self.params_q, self.params_k):
            param_k.set_data(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        queue = np.random.randn(K, dim)  # diff from torch version!
        queue = queue / np.linalg.norm(queue, axis=1, keepdims=True)
        self.queue = Parameter(queue.astype(np.float32), requires_grad=False)
        self.queue_ptr = Parameter(Tensor(0, dtype=ms.float32), requires_grad=False)

        # necessary operations
        self.normalize = ops.L2Normalize(axis=1, epsilon=1e-12)
        self.hyper_map = ops.HyperMap()
        self.partial = ops.Partial()
        self.bmm = ops.BatchMatMul()
        self.mm = ops.MatMul(transpose_b=True)
        self.rank_id = rank_id
        self.device_num = device_num
        if self.device_num == 1:
            self.all_gather = lambda x: x
            self.broadcast = lambda x: x
        else:
            self.all_gather = ops.AllGather()
            self.broadcast = ops.Broadcast(0)

    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        success = self.hyper_map(self.partial(_momentum_update_parameter, Tensor(self.m, ms.float32)),
                                 self.params_k, self.params_q)
        return success

    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = self.all_gather(keys)

        batch_size = keys.shape[0]
        # assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        index = ops.linspace(self.queue_ptr, self.queue_ptr + batch_size - 1, batch_size)
        index = ops.cast(index, ms.int64)
        self.queue = ops.scatter_update(self.queue, index, keys)
        self.queue_ptr = ops.depend(self.queue_ptr, self.queue)
        self.queue_ptr = (self.queue_ptr + batch_size) % self.K  # move pointer
        return self.queue_ptr

    def _batch_shuffle_ddp(self, x, idx_shuffle):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        x_gather = self.all_gather(x)

        # random shuffle index
        # move random shuffle index to dataset for convenience!

        # broadcast to all gpus
        idx_shuffle = self.broadcast((idx_shuffle,))[0]

        # index for restoring
        # move unshuffle index to dataset for convenience!

        # shuffled index for this gpu
        idx_this = idx_shuffle.view(self.device_num, -1)[self.rank_id]

        return x_gather[idx_this]

    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        x_gather = self.all_gather(x)

        # broadcast to all gpus
        # we also need to do broadcast for idx_unshuffle...
        idx_unshuffle = self.broadcast((idx_unshuffle,))[0]

        # restored index for this gpu
        idx_this = idx_unshuffle.view(self.device_num, -1)[self.rank_id]

        return x_gather[idx_this]

    def construct(self, im_q, k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = self.normalize(q)

        # compute key features
        # move to MoCoTrainStep to support torch.no_grad?

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = ops.squeeze(self.bmm(ops.expand_dims(q, 1), ops.expand_dims(k, 2)), -1)  # q.view(N,1,C) @ k.view(N,C,1)
        # negative logits: NxK
        # MatMul in fp32 is too slow!
        l_neg = ops.cast(self.mm(ops.cast(q, ms.float16), ops.cast(self.queue, ms.float16)), ms.float32)

        # logits: Nx(1+K)
        logits = ops.concat([l_pos, l_neg], axis=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        # move label construction to dataset for convenience!

        # dequeue and enqueue
        # move to MoCoTrainStep to support torch.no_grad?

        return logits


class WithLossCell(nn.WithLossCell):
    def construct(self, im_q, k, label):
        out = self._backbone(im_q, k)
        return self._loss_fn(out, label)


class MoCoTrainStep(nn.TrainOneStepCell):
    def construct(self, im_q, im_k, label, idx_shuffle, idx_unshuffle):
        # check_inputs(im_q, im_k)
        # -------------------------------- MoCo Stuff -------------------------------- #
        # compute key features
        # here we update the key encoder and compute key features before compute query features, but it's ok.
        success = self.network._backbone._momentum_update_key_encoder()  # update the key encoder
        im_k = ops.depend(im_k, success)
        im_k = self.network._backbone._batch_shuffle_ddp(im_k, idx_shuffle.view(-1))  # shuffle for making use of BN
        k = self.network._backbone.encoder_k(im_k)  # keys: NxC
        k = self.network._backbone.normalize(k)
        k = self.network._backbone._batch_unshuffle_ddp(k, idx_unshuffle.view(-1))  # undo shuffle

        loss = self.network(im_q, k, label)
        k = ops.depend(k, loss)

        # dequeue and enqueue
        queue_ptr = self.network._backbone._dequeue_and_enqueue(k)
        loss = ops.depend(loss, queue_ptr)
        # -------------------------------- MoCo Stuff -------------------------------- #

        sens = ops.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, self.weights)(im_q, k, label, sens)
        grads = self.grad_reducer(grads)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss


def check_inputs(im_q, im_k):
    def t2i(t):
        mean = np.expand_dims(np.array([0.485, 0.456, 0.406]), (1, 2))
        std = np.expand_dims(np.array([0.229, 0.224, 0.225]), (1, 2))
        t = 255 * (t * std + mean)
        t = np.transpose(t, (1, 2, 0))
        t = t.astype(np.uint8)
        return t

    import cv2
    im_q = im_q.asnumpy()
    im_k = im_k.asnumpy()
    ims = []
    for i in range(len(im_q)):
        q = t2i(im_q[i])
        k = t2i(im_k[i])
        im = cv2.hconcat((q, k))
        cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        ims.append(im)
    ims = cv2.vconcat(ims)
    cv2.imwrite("input.jpg", ims)
