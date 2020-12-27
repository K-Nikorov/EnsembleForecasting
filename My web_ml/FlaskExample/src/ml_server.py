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


def get_prediction(X_test, ident):
    model = users[ident].model
    filename = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction_' + str(ident) + '.csv')
    pred = model.predict(X_test.values)
    pred = pd.Series(data=pred, name='Prediction', index=X_test.index)
    pred.to_csv(filename)
    return 'prediction_' + str(ident) + '.csv'


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
    cur_us.model = None
    cur_us.train_res = None
    cur_us.train_data = None
    cur_us.target_data = None
    cur_us.test_data = None
    cur_us.predicted_data = None

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


@app.route('/predict/<int:ident>', methods=['GET', 'POST'])
def predict(ident):
    cur_us = users[ident]
    if request.method == 'POST':
        X_test = request.files['test']
        if X_test and allowed_file(X_test.filename):
            cur_us.test_data = os.path.join(app.config['UPLOAD_FOLDER'], 'test_data_' + str(cur_us.ident) + '.csv')
            X_test.save(cur_us.test_data)
            cur_us.predicted_data = get_prediction(pd.read_csv(cur_us.test_data), ident)
            users[ident] = cur_us
            return redirect(url_for('uploaded_file',
                                    ident=ident))
    return render_template('predict.html', ident=ident)


@app.route('/uploads/<int:ident>')
def uploaded_file(ident):
    cur_us = users[ident]
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               cur_us.predicted_data, as_attachment=True, cache_timeout=0,
                               attachment_filename='prediction.csv')


@app.route('/clean_data/<int:ident>')
def clean_data(ident):
    cur_us = users[ident]
    if isinstance(cur_us.train_data, str):
        os.remove(cur_us.train_data)

    if isinstance(cur_us.target_data, str):
        os.remove(cur_us.target_data)

    if isinstance(cur_us.test_data, str):
        os.remove(cur_us.test_data)

    if isinstance(cur_us.train_res, str):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], cur_us.train_res))

    if isinstance(cur_us.predicted_data, str):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], cur_us.predicted_data))

    return redirect(url_for('upload_file', ident=ident))
