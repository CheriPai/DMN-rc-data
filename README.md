# Dynamic memory networks trained on rc-data

Forked from https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano

## Repository contents

| file | description |
| --- | --- |
| `main.py` | the main entry point to train and test available network architectures on bAbI-like tasks |
| `dmn_basic.py` | our baseline implementation. It is as close to the original as we could understand the paper, except the number of steps in the main memory GRU is fixed. Attention module uses `T.abs_` function as a distance between two vectors which causes gradients to become `NaN` randomly.  The results reported in [this blog post](http://yerevann.github.io/2016/02/05/implementing-dynamic-memory-networks/) are based on this network |
| `dmn_smooth.py` | uses the square of the Euclidean distance instead of `abs` in the attention module. Training is very stable. Performance on bAbI is slightly better |
| `dmn_batch.py` | `dmn_smooth` with minibatch training support. The batch size cannot be set to `1` because of the [Theano bug](https://github.com/Theano/Theano/issues/1772) | 
| `dmn_qa_draft.py` | draft version of a DMN designed for answering multiple choice questions | 
| `utils.py` | tools for working with bAbI tasks and GloVe vectors |
| `nn_utils.py` | helper functions on top of Theano and Lasagne |
| `fetch_glove_data.sh` | shell script to fetch GloVe vectors (by [5vision](https://github.com/5vision/kaggle_allen)) |
| `server/` | contains Flask-based restful api server |


## Usage

This implementation is based on Theano and Lasagne. One way to install them is:

    pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
    pip install https://github.com/Lasagne/Lasagne/archive/master.zip

The following bash scripts will download GloVe vectors.

    ./fetch_glove_data.sh

Use `main.py` to train a network:

    python main.py --network dmn_basic --babi_id 1

The states of the network will be saved in `states/` folder. 
There is one pretrained state on the 1st bAbI task. It should give 100% accuracy on the test set:

    python main.py --network dmn_basic --mode test --babi_id 1 --load_state states/dmn_basic.mh5.n40.babi1.epoch4.test0.00033.state


## Roadmap

* Mini-batch training ([done](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/blob/master/dmn_batch.py), 08/02/2016)
* Web interface ([done](https://github.com/YerevaNN/dmn-ui), 08/23/2016)
* Visualization of episodic memory module ([done](https://github.com/YerevaNN/dmn-ui), 08/23/2016)
* Regularization (work in progress, L2 doesn't help at all, dropout and batch normalization help a little)
* Support for multiple-choice questions ([work in progress](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/blob/master/dmn_qa_draft.py))
* Evaluation on more complex datasets
* Import some ideas from [Neural Reasoner](http://arxiv.org/abs/1508.05508)
