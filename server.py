from flask import Flask
import dmn_smooth
import numpy as np
import os
import random
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
dmn = None


@app.route('/')
def main():
    random_file = random.choice(os.listdir('data/cnn/questions/validation/'))
    data = utils.init_file('data/cnn/questions/validation/' + random_file)
    correct_answer = data[0]["A"]
    probabilities, attentions = dmn.predict(data)
    print probabilities[0:50]
    return "Prediction: @entity" + \
        str(np.argmax(probabilities)) + "    " + \
        "Answer: " + correct_answer


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
