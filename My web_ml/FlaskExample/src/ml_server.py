import ensembles


from flask import Flask
from flask import render_template

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


@app.route('/', methods=['GET', 'POST'])
def index():
    global glob_ident
    users[glob_ident] = UserData(glob_ident)
    glob_ident += 1
    return render_template('index.html', ident=glob_ident-1)


@app.route('/upload_file/<int:ident>', methods=['GET', 'POST'])
def upload_file(ident):
    return ''
