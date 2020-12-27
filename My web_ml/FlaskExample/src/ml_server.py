import os
import numpy as np
import ensembles
import pandas as pd
import matplotlib.pyplot as plt

from flask import Flask, request, url_for
from flask import render_template, redirect
from flask import send_from_directory


app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
app.config['CACHE_TYPE'] = 'null'


class UserData:
    def __init__(self, ident):
        self.ident = ident
        self.model_name = None
        self.model_params = dict()
        self.model = None
        self.train_data = None
        self.target_data = None
        self.test_data = None
        self.predicted_data = None
        self.train_res = None


glob_ident = 0
users = dict()


def get_train_res(ident):
    cur_us = users[ident]
    if cur_us.model_name == 'Random Forest':
        cur_us.model = ensembles.RandomForestMSE(**cur_us.model_params)
    else:
        cur_us.model = ensembles.GradientBoostingMSE(**cur_us.model_params)
    X = pd.read_csv(cur_us.train_data)
    y = pd.read_csv(cur_us.target_data)
    X = X.sample(frac=1)
    y = y.iloc[X.index]
    X_arr = X.values
    y_arr = y.values.reshape(-1)
    iters, log_train = cur_us.model.fit(X_arr, y_arr, return_log=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_ylim(0, 500)
    ax.plot(iters, log_train, label='train RMSE')
    ax.set_xlabel('Номер итерации')
    ax.set_ylabel('Rooted Mean Squared Error')
    ax.set_title('Ход обучения алгоритма ' + cur_us.model_name)
    ax.legend()
    ax.grid(True, alpha=0.5)
    return fig, iters[0: len(iters): len(iters)//15 + 1], log_train[0: len(iters): len(iters)//15 + 1]


@app.route('/', methods=['GET', 'POST'])
def index():
    global glob_ident
    users[glob_ident] = UserData(glob_ident)
    glob_ident += 1
    return render_template('index.html', ident=glob_ident-1)


@app.route('/upload_file/<int:ident>', methods=['GET', 'POST'])
def upload_file(ident):
    return ''


@app.route('/train_res/<int:ident>', methods=['GET', 'POST'])
def train_res(ident):
    fig, iters, log_train = get_train_res(ident)
    cur_us = users[ident]
    cur_us.train_res = 'train_res_'+str(cur_us.ident)+'.png'
    fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'], cur_us.train_res))
    users[ident] = cur_us
    noise = "".join((np.random.randn(3) + 4).astype(str))
    return render_template('train_res.html', ident=ident, noise=noise, iters=iters, log_train=log_train, log_len=len(log_train))


@app.route('/train_res/<int:ident>/figure/<string:noise>')
def figure(ident, noise):
    cur_us = users[ident]
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename=cur_us.train_res, mimetype='image/png')


@app.route('/model_settings/<int:ident>', methods=['GET', 'POST'])
def model_settings(ident):
    return ''


@app.route('/predict/<int:ident>', methods=['GET', 'POST'])
def predict(ident):
    return ''


@app.route('/clean_data/<int:ident>')
def clean_data(ident):
    cur_us = users[ident]
    if isinstance(cur_us.train_data, str):
        os.remove(cur_us.train_data)

    if isinstance(cur_us.target_data, str):
        os.remove(cur_us.target_data)

    if isinstance(cur_us.train_res, str):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], cur_us.train_res))

    return redirect(url_for('upload_file', ident=ident))
