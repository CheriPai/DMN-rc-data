from flask import Flask, jsonify, render_template
from numpy import argmax
from random import choice
import dmn_smooth
import os
import utils

app = Flask(__name__)
word_vector_size = 50
dim = 64
mode = "predict"
answer_module = "feedforward"
input_mask_mode = "sentence"
memory_hops = 5
l2 = 0
normalize_attention = False
batch_norm = False
dropout = 0.05
learning_rate = 0.0001
state = "states/dmn_smooth.mh5.n64.bs10.d0.05.cnn.epoch0.test2.54527.state"
validation_path = "data/cnn/questions/validation/"
test_path = "data/cnn/questions/test/"
dmn = None


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/random/<dataset>")
def random(dataset):
    if dataset == "test":
        path = test_path
    else:
        path = validation_path

    random_file = choice(os.listdir(path))
    data = utils.init_file(path + random_file)
    correct_answer = data[0]["A"]
    probabilities, attentions = dmn.predict(data)
    data = data[0]
    data["P"] = "@entity" + str(argmax(probabilities))
    data["A"] = correct_answer  # The predict function overwrites this

    data = replace_entities(data)

    return jsonify(context=data["C"],
                   question=data["Q"],
                   correct_answer=data["A"],
                   prediction=data["P"])


def replace_entities(data):
    """ Replaces @entityN with actual string
    """
    context = data["C"]
    question = data["Q"]
    correct_answer = data["A"]
    prediction = data["P"]
    split_context = context.split()
    split_question = question.split()

    for k, v in data["E"].iteritems():
        for i, w in enumerate(split_context):
            if w == k:
                split_context[i] = v

        for i, w in enumerate(split_question):
            if w == k:
                split_question[i] = v

        if (correct_answer == k):
            data["A"] = v
        if (prediction == k):
            data["P"] = v

    data["C"] = " ".join(split_context)
    data["Q"] = " ".join(split_question).replace("@placeholder", "________")

    return data


if __name__ == "__main__":
    babi_train_raw, babi_test_raw = utils.get_babi_raw("cnn", "")
    word2vec = utils.load_glove(word_vector_size)
    dmn = dmn_smooth.DMN_smooth(
        None, babi_test_raw, word2vec, word_vector_size, dim, mode,
        answer_module, input_mask_mode, memory_hops, l2, normalize_attention,
        batch_norm, dropout, learning_rate)
    dmn.load_state(state)

    print "==> running server"
    app.run(debug=True, use_reloader=False)
