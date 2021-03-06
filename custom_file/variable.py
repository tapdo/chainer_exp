import heapq

import numpy

from chainer import cuda
from chainer import flag


class Variable(object):

    """Array with a structure to keep track of computation.

    Every variable holds a data array of type either :class:`numpy.ndarray` or
    :class:`cupy.ndarray`.

    A Variable object may be constructed in two ways: by the user or by some
    function. When a variable is created by some function as one of its
    outputs, the variable holds a reference to that function. This reference is
    used in error backpropagation (a.k.a. backprop). It is also used in
    *backward unchaining*. A variable that does not hold a reference to its
    creator is called a *root* variable. A variable is root if it is created by
    the user, or if the reference is deleted by :meth:`unchain_backward`.

    Users can disable this chaining behavior by setting the volatile flag for
    the initial variables. When a function gets volatile variables as its
    inputs, the output variables do not hold references to the function. This
    acts like unchaining on every function application.

    Args:
        data (array): Initial data array.
        volatile (~chainer.Flag): Volatility flag. String ('on', 'off', or
            'auto') or boolean values can be used, too.
        name (str): Name of the variable.

    Attributes:
        data: Data array of type either :class:`numpy.ndarray` or
            :class:`cupy.ndarray`.
        grad: Gradient array. It is ``None`` until backprop reaches this
            variable.
        creator: The function who creates this variable. It is ``None`` if the
            variable is not created by any function.
        volatile: Ternary :class:`~chainer.Flag` object. If ON, the variable
            does not keep track of any function applications. See
            :class:`~chainer.Flag` for the detail of ternary flags.

    """
    def __init__(self, data, volatile=flag.OFF, name=None):
        assert isinstance(data, (numpy.ndarray, cuda.ndarray))

        self.data = data
        self.rank = 0
        self._volatile = flag.Flag(volatile)

        self._grad = None
        self.creator = None

        self.name = name

    def __reduce__(self):
        return (Variable, (self.data, self.volatile, self.name))

    def __repr__(self):
        if self.name:
            return '<variable %s>' % self.name
        else:
            return '<variable at 0x%x>' % id(self)

    def __str__(self):
        return self.name or ('<var@%x>' % id(self))

    def __pos__(self):
        return self

    def __len__(self):
        """Returns the number of elements of the data array.

        Returns:
            int: the number of elements of the data array.

        """
        return self.data.size

    @property
    def volatile(self):
        return self._volatile

    @volatile.setter
    def volatile(self, v):
        self._volatile = flag.Flag(v)

    @property
    def label(self):
        """Short text that represents the function."""
        if self.data.shape == ():
            return str(self.data.dtype)
        return '(%s), %s' % (', '.join(map(str, self.data.shape)),
                             str(self.data.dtype))

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, g):
        error_msg = '''
This error is occured in two cases. The first case is when the user manually
sets the Variable.grad incorrectly. The second case is when some Function
implementation has a bug. If you do not manually set the Variable.grad in your
script, please report this error to the issue tracker with the stack trace,
the information of your environment, and your script:
https://github.com/pfnet/chainer/issues/new.
'''
        if g is not None:
            if not isinstance(g, type(self.data)):
                raise TypeError('Type of data and grad mismatch: %s != %s%s'
                                % (type(self.data), type(g), error_msg))
            if g.dtype != self.data.dtype:
                raise TypeError('Dtype of data and grad mismatch: %s != %s%s'
                                % (self.data.dtype, g.dtype, error_msg))
            if g.shape != self.data.shape:
                raise ValueError('Shape of data and grad mismatch: %s != %s%s'
                                 % (self.data.shape, g.shape, error_msg))
        self._grad = g

    def to_cpu(self):
        """Copies the data and gradient arrays to CPU."""
        self.data = cuda.to_cpu(self.data)
        if self._grad is not None:
            self._grad = cuda.to_cpu(self._grad)

    def to_gpu(self, device=None):
        """Copies the data and gradient arrays to specified GPU.

        Args:
            device: Target device specifier. If omitted, the current device is
                used.

        """
        #with cuda.get_device(device):
        self.data = cuda.to_gpu(self.data)
        if self._grad is not None:
            self._grad = cuda.to_gpu(self._grad)

    def zerograd(self):
        """Initializes the gradient array by zeros."""
        #with cuda.get_device(self.data) as dev:
        dev = 0
        if self._grad is None:
            xp = numpy if int(dev) == -1 else cuda.cupy
            self._grad = xp.zeros_like(self.data)
        else:
            self._grad.fill(0)

    def copydata(self, var):
        """Copies the data array from given source variable.

        This method just copies the data attribute from given variable to this
        variable, except that the copy is even done across the host and
        different devices.

        Args:
            var (Variable): Source variable.

        """
        src = var.data
        dst = self.data
        src_xp = cuda.get_array_module(src)
        dst_xp = cuda.get_array_module(dst)
        if dst_xp is src_xp:
            dst_xp.copyto(dst, src)
        elif dst_xp is numpy:
            dst_xp.copyto(dst, src.get())
        else:
            dst.set(src)

    def addgrad(self, var):
        """Accumulates the gradient array from given source variable.

        This method just runs ``self.grad += var.grad``, except that the
        accumulation is even done across the host and different devices.

        Args:
            var (Variable): Source variable.

        """
        src = var._grad
        dst = self._grad
        if src is None:
            raise ValueError('Source gradient is not set.')
        if dst is None:
            raise ValueError('Target graidient is not set.')

        xp = cuda.get_array_module(dst)
        if xp is numpy:
            dst += cuda.to_cpu(src)
        elif isinstance(src, numpy.ndarray):
            dst += cuda.to_gpu(src, device=dst)
        else:
            dst_dev = dst.device
            if dst_dev == src.device:
                dst += src
            else:
                with dst_dev:
                    dst += xp.copy(src)

    def set_creator(self, gen_func):
        """Notifies the variable that the given function is its creator.

        Args:
            gen_func (Function): Function object that creates this variable as
                one of its outputs.

        """
        self.creator = gen_func
        self.rank = gen_func.rank + 1

    def backward(self, retain_grad=False):
        """Runs error backpropagation (a.k.a. backprop) from this variable.

        On backprop, :meth:`Function.backward` is called on each
        :class:`Function` object appearing in the backward graph starting from
        this variable. The backward graph is represented by backward references
        from variables to their creators, and from functions to their inputs.
        The backprop stops at all root variables. Some functions set ``None``
        as gradients of some inputs, where further backprop does not take place
        at such input variables.

        This method uses :data:`grad` as the initial error array. User can
        manually set a gradient array before calling this method. If
        :data:`data` contains only one element (i.e., it is scalar) and
        :data:`grad` is None, then this method automatically complements 1.0 as
        the initial error. This is useful on starting backprop from some scalar
        loss value.

        Args:
            retain_grad (bool): If True, the gradient arrays of all
                intermediate variables are kept. Otherwise, :data:`grad` of the
                intermediate variables are set to ``None`` on appropriate
                timing, which may reduce the maximum memory consumption.

                In most cases of training some model, the purpose of backprop
                is to compute gradients of parameters, not of variables, so it
                is recommended to set this flag False.

        """
        if self.creator is None:
            return

        cand_funcs = []
        seen_set = set()
        seen_vars = set()
        need_copy = set()

        # Initilize error by 1, if this is a loss variable
        if self.data.size == 1 and self.grad is None:
           #with cuda.get_device(self.data) as device:
            device = 0
            if device is cuda.DummyDevice:
                self.grad = numpy.ones_like(self.data)
            else:
                self.grad = cuda.cupy.ones_like(self.data)

        def add_cand(cand):
            if cand not in seen_set:
                # Negate since heapq is min-heap
                heapq.heappush(cand_funcs, (-cand.rank, len(seen_set), cand))
                seen_set.add(cand)

        if self.creator is not None:
            add_cand(self.creator)

        while cand_funcs:
            _, _, func = heapq.heappop(cand_funcs)
            outputs = tuple(y() for y in func.outputs)  # access via weak ref

            in_data = tuple(x.data for x in func.inputs)
            out_grad = tuple(None if y is None else y.grad for y in outputs)
            #with cuda.get_device(*(in_data + out_grad)):
            gxs = func.backward(in_data, out_grad)
            assert len(gxs) == len(in_data)

            if not retain_grad:
                for y in outputs:
                    if y is not None and y is not self:
                        y.grad = None
            for x, gx in zip(func.inputs, gxs):
                if gx is None:
                    continue
                # Accumulate the graident to x. It is a bit tricky to handle
                # branches and parameter gradient accumulation correctly.
                #with cuda.get_device(gx):
                id_x = id(x)
                if x.creator is None:  # leaf
                    if x._grad is None:
                        x.grad = gx
                        need_copy.add(id_x)
                    elif id_x in need_copy:
                        x.grad = x.grad + gx  # copy
                        need_copy.remove(id_x)
                    else:
                        x._grad += gx
                else:  # not a leaf
                    add_cand(x.creator)
                    if id_x not in seen_vars:  # 1st visit
                        x.grad = gx
                        seen_vars.add(id_x)
                        need_copy.add(id_x)
                    elif id_x in need_copy:  # 2nd visit
                        x._grad = gx + x._grad  # copied
                        need_copy.remove(id_x)
                    else:  # 3rd or later visit
                        x._grad += gx

    def unchain_backward(self):
        """Deletes references between variables and functions backward.

        After this method completes, intermediate variables and functions that
        are not referenced from anywhere are deallocated by reference
        count GC. Also this variable itself deletes the reference to its
        creator function, i.e. this variable becomes root in the computation
        graph. It indicates that backprop after unchaining stops at this
        variable. This behavior is useful to implement truncated BPTT.

        """
        cand_funcs = []
        seen_set = set()

        def add_cand(cand):
            if cand is not None and cand not in seen_set:
                cand_funcs.append(cand)
                seen_set.add(cand)

        add_cand(self.creator)

        while cand_funcs:
            func = cand_funcs.pop()
            for var in func.inputs:
                add_cand(var.creator)
            func.unchain()

    def __lt__(self, other):
        raise NotImplementedError()

    def __le__(self, other):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()

    def __ne__(self, other):
        raise NotImplementedError()

    def __gt__(self, other):
        raise NotImplementedError()

    def __ge__(self, other):
        raise NotImplementedError()

    def __nonzero__(self):
        raise NotImplementedError()

    def __bool__(self):
        raise NotImplementedError()

    def __hash__(self):
        return super(Variable, self).__hash__()

    __array_priority__ = 200
