from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_experiment')
def submit_experiment():
    return 'Experiment submitted!'

@app.route('/get_experiment_results')
def get_experiment_results():
    return 'Experiment results'

