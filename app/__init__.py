import os
from flask import Flask, render_template, request, redirect, url_for

from . import db, experiment
from app.db import get_db

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'mnist.sqlite')
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    db.init_app(app)

    app.register_blueprint(experiment.bp)
    app.add_url_rule('/', endpoint='index')


    @app.route('/')
    def index():
        db = get_db()
        jobs = db.execute('SELECT * FROM job ORDER BY accuracy DESC').fetchall()

        return render_template('experiment/result.html', jobs=jobs)


    @app.route('/add_job', methods=['POST'])
    def add_job():
        batch_size = request.form['batch-size']
        learning_rate = request.form['learning-rate']
        epochs = request.form['number-of-epochs']

        #TODO: check duplicated jobs

        db = get_db()
        db.execute('INSERT INTO job (batch_size, learning_rate, epochs) VALUES (?,?,?)', (batch_size, learning_rate, epochs))
        db.commit()

        return redirect(url_for('index'))
    
    return app
        
