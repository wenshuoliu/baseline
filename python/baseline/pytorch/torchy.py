import torch
import numpy as np
from baseline.utils import lookup_sentence, get_version
import torch.autograd
import torch.nn as nn


PYT_MAJOR_VERSION = get_version(torch)


def classify_bt(model, batch_time):
    tensor = torch.from_numpy(batch_time) if type(batch_time) == np.ndarray else batch_time
    probs = model(torch.autograd.Variable(tensor, requires_grad=False).cuda()).exp().data
    probs.div_(torch.sum(probs))
    results = []
    batchsz = probs.size(0)
    for b in range(batchsz):
        outcomes = [(model.labels[id_i], prob_i) for id_i, prob_i in enumerate(probs[b])]
        results.append(outcomes)
    return results


def predict_seq_bt(model, x, xch, lengths):
    x_t = torch.from_numpy(x) if type(x) == np.ndarray else x
    xch_t = torch.from_numpy(xch) if type(xch) == np.ndarray else xch
    len_v = torch.from_numpy(lengths) if type(lengths) == np.ndarray else lengths
    x_v = torch.autograd.Variable(x_t, requires_grad=False).cuda()
    xch_v = torch.autograd.Variable(xch_t, requires_grad=False).cuda()
    #len_v = torch.autograd.Variable(len_t, requires_grad=False)
    results = model((x_v, xch_v, len_v))
    #print(results)
    #if type(x) == np.ndarray:
    #    # results = results.cpu().numpy()
    #    # Fix this to not be greedy
    #    results = np.argmax(results, -1)

    return results


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class SequenceCriterion(nn.Module):

    def __init__(self, LossFn=nn.NLLLoss):
        super(SequenceCriterion, self).__init__()
        self.crit = LossFn(ignore_index=0, size_average=False)

    def forward(self, inputs, targets):
        # This is BxT, which is what we want!
        total_sz = targets.nelement()
        loss = self.crit(inputs.view(total_sz, -1), targets.view(total_sz))
        return loss


class StackedLSTMCell(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTMCell, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size=input_size, hidden_size=rnn_size, bias=False))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        hs, cs = [], []
        for i, layer in enumerate(self.layers):
            h_i, c_i = layer(input, (h_0[i], c_0[i]))
            input = h_i
            if i != self.num_layers - 1:
                input = self.dropout(input)
            hs += [h_i]
            cs += [c_i]

        hs = torch.stack(hs)
        cs = torch.stack(cs)

        return input, (hs, cs)


class StackedGRUCell(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRUCell, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size=input_size, hidden_size=rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0 = hidden
        hs = []
        for i, layer in enumerate(self.layers):
            h_i = layer(input, (h_0[i]))
            input = h_i
            if i != self.num_layers:
                input = self.dropout(input)
            hs += [h_i]

        hs = torch.stack(hs)

        return input, hs


def pytorch_rnn_cell(insz, hsz, rnntype, nlayers, dropout):

    if rnntype == 'gru':
        rnn = StackedGRUCell(nlayers, insz, hsz, dropout)
    else:
        rnn = StackedLSTMCell(nlayers, insz, hsz, dropout)
    return rnn


def pytorch_embedding(x2vec, finetune=True):
    dsz = x2vec.dsz
    lut = nn.Embedding(x2vec.vsz + 1, dsz, padding_idx=0)
    del lut.weight
    lut.weight = nn.Parameter(torch.FloatTensor(x2vec.weights),
                              requires_grad=finetune)
    return lut


def pytorch_activation(name="relu"):
    if name == "tanh":
        return nn.Tanh()
    if name == "prelu":
        return nn.PReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "log_sigmoid":
        return nn.LogSigmoid()
    return nn.ReLU()


def pytorch_conv1d(in_channels, out_channels, fsz, unif=0, padding=0):
    c = nn.Conv1d(in_channels, out_channels, fsz, padding=padding)
    if unif > 0:
        c.weight.data.uniform_(-unif, unif)
    else:
        torch.nn.init.xavier_uniform(c.weight)
    return c


def pytorch_linear(in_sz, out_sz, unif=0):
    l = nn.Linear(in_sz, out_sz)
    if unif > 0:
        l.weight.data.uniform_(-unif, unif)
    else:
        torch.nn.init.xavier_uniform(l.weight)
    l.bias.data.zero_()
    return l


def pytorch_rnn(insz, hsz, rnntype, nlayers, dropout):

    if rnntype == 'gru':
        rnn = torch.nn.GRU(insz, hsz, nlayers, dropout=dropout)
    else:
        rnn = torch.nn.LSTM(insz, hsz, nlayers, dropout=dropout)

    return rnn


class Highway(nn.Module):

    def __init__(self,
                 input_size):
        super(Highway, self).__init__()
        self.proj = nn.Linear(input_size, input_size)
        self.transform = nn.Linear(input_size, input_size)
        self.transform.bias.data.fill_(-2.0)

    def forward(self, input):
        proj_result = nn.functional.relu(self.proj(input))
        proj_gate = nn.functional.sigmoid(self.transform(input))
        gated = (proj_gate * proj_result) + ((1 - proj_gate) * input)
        return gated


def pytorch_lstm(insz, hsz, rnntype, nlayers, dropout, unif=0, batch_first=False):
    ndir = 2 if rnntype.startswith('b') else 1
    #print('ndir: %d, rnntype: %s, nlayers: %d, dropout: %.2f, unif: %.2f' % (ndir, rnntype, nlayers, dropout, unif))
    rnn = torch.nn.LSTM(insz, hsz, nlayers, dropout=dropout, bidirectional=True if ndir > 1 else False, batch_first=batch_first)#, bias=False)
    if unif > 0:
        for weight in rnn.parameters():
            weight.data.uniform_(-unif, unif)
    else:
        nn.init.xavier_uniform(rnn.weight_hh_l0)
        nn.init.xavier_uniform(rnn.weight_ih_l0)

    return rnn, ndir*hsz

# https://github.com/pytorch/pytorch/pull/2016/files
class CyclicLR(object):
    """Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.
    Cyclical learning rate policy changes the learning rate after every batch.
    `batch_step` should be called after a batch has been used for training.
    To resume training, save `last_batch_iteration` and use it to instantiate `CycleLR`.
    This class has three built-in policies, as put forth in the paper:
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    This implementation was adapted from the github repo: `bckenstler/CLR`_
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for eachparam groups.
            Default: 0.001
        max_lr (float or list): Upper boundaries in the cycle for
            each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function. Default: 0.006
        step_size (int): Number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch. Default: 2000
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        last_batch_iteration (int): The index of the last batch. Default: -1
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.CyclicLR(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         scheduler.batch_step()
        >>>         train_batch(...)
    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs

def pytorch_prepare_optimizer(model, **kwargs):

    mom = kwargs.get('mom', 0.9)
    optim = kwargs.get('optim', 'sgd')
    eta = kwargs.get('eta', kwargs.get('lr', 0.01))
    decay_rate = float(kwargs.get('decay_rate', 0.0))
    decay_type = kwargs.get('decay_type', None)
    bounds = kwargs.get('bounds', 2000)
    if optim == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=eta)
    elif optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=eta)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=eta)
    elif optim == 'asgd':
        optimizer = torch.optim.ASGD(model.parameters(), lr=eta)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=eta, momentum=mom)

    scheduler = None
    if decay_rate > 0.0 and decay_type is not None:
        if decay_type == 'invtime':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=decay_rate)

    if decay_type == "clr":
        print('Step size', bounds)
        scheduler = CyclicLR(optimizer, max_lr=2*eta, step_size=bounds)

    return optimizer, scheduler


def append2seq(seq, modules):

    for i, module in enumerate(modules):
        seq.add_module('%s-%d' % (str(module), i), module)


def tensor_max(tensor):
    return tensor.max()


def tensor_shape(tensor):
    return tensor.size()


def tensor_reverse_2nd(tensor):
    idx = torch.LongTensor([i for i in range(tensor.size(1)-1, -1, -1)])
    return tensor.index_select(1, idx)


def long_0_tensor_alloc(dims, dtype=None):
    lt = long_tensor_alloc(dims)
    lt.zero_()
    return lt


def long_tensor_alloc(dims, dtype=None):
    if type(dims) == int or len(dims) == 1:
        return torch.LongTensor(dims)
    return torch.LongTensor(*dims)


# Mashed together from code using numpy only, hacked for th Tensors
# This function should never be used for decoding.  It exists only so that the training model can greedily decode
# It is super slow and doesnt use maintain a beam of hypotheses
def show_examples_pytorch(model, es, rlut1, rlut2, embed2, mxlen, sample, prob_clip, max_examples, reverse):
    si = np.random.randint(0, len(es))

    batch_dict = es[si]

    src_array = batch_dict['src']
    tgt_array = batch_dict['dst']
    src_len = batch_dict['src_len']
    #src_array, tgt_array, src_len, _ = es[si]

    if max_examples > 0:
        max_examples = min(max_examples, src_array.size(0))
        src_array = src_array[0:max_examples]
        tgt_array = tgt_array[0:max_examples]
        src_len = src_len[0:max_examples]

    GO = embed2.vocab['<GO>']
    EOS = embed2.vocab['<EOS>']

    # TODO: fix this, check for GPU first
    src_array = src_array.cuda()
    
    for src_len,src_i,tgt_i in zip(src_len, src_array, tgt_array):

        print('========================================================================')

        sent = lookup_sentence(rlut1, src_i.cpu().numpy(), reverse=reverse)
        print('[OP] %s' % sent)
        sent = lookup_sentence(rlut2, tgt_i)
        print('[Actual] %s' % sent)
        dst_i = torch.zeros(1, mxlen).long()
        #if use_gpu:
        dst_i = dst_i.cuda()

        next_value = GO
        src_i = src_i.view(1, -1)
        for j in range(mxlen):
            dst_i[0,j] = next_value
            probv = model((torch.autograd.Variable(src_i), torch.autograd.Variable(dst_i)))
            output = probv.squeeze()[j]
            if sample is False:
                _, next_value = torch.max(output, 0)
                next_value = int(next_value.data[0])
            else:
                probs = output.data.exp()
                # This is going to zero out low prob. events so they are not
                # sampled from
                best, ids = probs.topk(prob_clip, 0, largest=True, sorted=True)
                probs.zero_()
                probs.index_copy_(0, ids, best)
                probs.div_(torch.sum(probs))
                fv = torch.multinomial(probs, 1)[0]
                next_value = fv

            if next_value == EOS:
                break

        sent = lookup_sentence(rlut2, dst_i.squeeze())
        print('Guess: %s' % sent)
        print('------------------------------------------------------------------------')
