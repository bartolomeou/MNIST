import os
from flask import Flask, render_template, request, redirect, url_for

from . import db, experiment


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, static_folder='../client/static', template_folder='../client/templates')
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

    return app