from flask import Flask, jsonify, render_template
from numpy import argmax
from random import choice
import dmn_smooth
import os
import utils


app = Flask(__name__)
word_vector_size = 50
dim = 64
mode = 'predict'
answer_module = 'feedforward'
input_mask_mode = 'sentence'
memory_hops = 5
l2 = 0
normalize_attention = False
batch_norm = False
dropout = 0.05
learning_rate = 0.0001
state = 'states/dmn_smooth.mh5.n64.bs10.d0.05.cnn.epoch0.test2.66246.state'
validation_path = 'data/cnn/questions/validation/'
test_path = 'data/cnn/questions/test/'
dmn = None


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/random/<dataset>')
def random(dataset):
    if dataset == 'test':
        path = test_path
    else:
        path = validation_path

    random_file = choice(os.listdir(path))
    data = utils.init_file(path + random_file)

    context = data[0]["C"]
    question = data[0]["Q"]
    correct_answer = data[0]["A"]

    probabilities, attentions = dmn.predict(data)
    prediction = "@entity" + str(argmax(probabilities))
    
    return jsonify(
        context=context,
        question=question,
        correct_answer=correct_answer,
        prediction=prediction
    )


if __name__ == '__main__':
    babi_train_raw, babi_test_raw = utils.get_babi_raw('cnn', '')
    word2vec = utils.load_glove(word_vector_size)
    dmn = dmn_smooth.DMN_smooth(
        None, babi_test_raw, word2vec, word_vector_size, dim,
        mode, answer_module, input_mask_mode, memory_hops, l2,
        normalize_attention, batch_norm, dropout, learning_rate
    )
    dmn.load_state(state)


    print "==> running server"
    app.run(debug=True, use_reloader=False)
