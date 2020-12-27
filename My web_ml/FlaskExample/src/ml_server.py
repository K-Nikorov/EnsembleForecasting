import os
import numpy as np
import ensembles
import pandas as pd
import matplotlib.pyplot as plt

from flask import Flask, request, url_for
from flask import render_template, redirect
from flask import send_from_directory


UPLOAD_FOLDER = './user_data'
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CACHE_TYPE'] = 'null'


russion_params = {'n_estimators': 'Количество деревьев',
                  'max_depth': 'Максимальная глубина дерева',
                  'learning_rate': 'Темп обучения',
                  'feature_subsample_size': 'Размерность подвыборки признаков'
                  }


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


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    global glob_ident
    users[glob_ident] = UserData(glob_ident)
    glob_ident += 1
    return render_template('index.html', ident=glob_ident-1)


@app.route('/upload_file/<int:ident>', methods=['GET', 'POST'])
def upload_file(ident):
    cur_us = users[ident]
    cur_us.model_params = dict()
    cur_us.model_name = None
    cur_us.model_params = dict()
    cur_us.model = None
    cur_us.train_res = None

    if request.method == 'POST':
        X_train, y_train = request.files['train'], request.files['target']
        if (X_train and y_train) and (allowed_file(X_train.filename) and allowed_file(y_train.filename)):
            cur_us.model_name = request.form['model']
            cur_us.train_data = os.path.join(app.config['UPLOAD_FOLDER'], 'train_data_'+str(cur_us.ident)+'.csv')
            cur_us.target_data = os.path.join(app.config['UPLOAD_FOLDER'], 'target_data_' + str(cur_us.ident) + '.csv')
            X_train.save(cur_us.train_data)
            y_train.save(cur_us.target_data)
            users[cur_us.ident] = cur_us
            return redirect(url_for('model_settings', ident=cur_us.ident))
    return render_template('upload_file.html')


@app.route('/model_settings/<int:ident>', methods=['GET', 'POST'])
def model_settings(ident):
    cur_us = users[ident]
    cur_us.model_params = dict()

    if request.method == 'POST':
        flag_ok = 1
        if request.form['n_estimators'] == '':
            cur_us.model_params['n_estimators'] = 100
        else:
            try:
                cur_us.model_params['n_estimators'] = int(request.form['n_estimators'])
            except:
                cur_us.model_params['n_estimators'] = (request.form['n_estimators']
                                                       + ' -- ERROR: должно быть положительным целочисленным')
                flag_ok = 0

        if request.form['max_depth'] == '':
            cur_us.model_params['max_depth'] = 5
        else:
            try:
                cur_us.model_params['max_depth'] = int(request.form['max_depth'])
            except:
                cur_us.model_params['max_depth'] = (request.form['max_depth']
                                                    + ' -- ERROR: должно быть положительным целочисленным ')
                flag_ok = 0

        if request.form['feature_subsample_size'] == '':
            cur_us.model_params['feature_subsample_size'] = 1.0
        else:
            try:
                cur_us.model_params['feature_subsample_size'] = float(request.form['feature_subsample_size'])
                if not(0 < cur_us.model_params['feature_subsample_size'] <= 1):
                    raise ValueError
            except:
                cur_us.model_params['feature_subsample_size'] = (request.form['feature_subsample_size']
                                                                 + ' -- ERROR: должно быть вещественным в диапазоне (0, 1]')
                flag_ok = 0

        if cur_us.model_name == 'Gradient Boosting':
            if request.form['learning_rate'] == '':
                cur_us.model_params['learning_rate'] = 0.1
            else:
                try:
                    cur_us.model_params['learning_rate'] = float(request.form['learning_rate'])
                except:
                    cur_us.model_params['learning_rate'] = (request.form['learning_rate']
                                                        + ' -- ERROR: должно быть положительным вещественным ')
                    flag_ok = 0
        users[ident] = cur_us
        return redirect(url_for('model_info', ident=cur_us.ident, correct_params=flag_ok))
    return render_template('model_settings.html', model_name=cur_us.model_name)


@app.route('/model_info/<int:ident>/<int:correct_params>', methods=['GET', 'POST'])
def model_info(ident, correct_params):
    cur_us = users[ident]
    if request.method == 'POST':
        pass
    return render_template('model_info.html', ident=ident, model_name=cur_us.model_name, model_params=cur_us.model_params,
                           correct_params=correct_params, russion_params=russion_params)


@app.route('/train_res/<int:ident>', methods=['GET', 'POST'])
def train_res(ident):
    return ''