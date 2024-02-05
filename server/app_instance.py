import os
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__, static_folder='../client/static', template_folder='../client/templates')
app.config.from_mapping(
    SECRET_KEY='dev',
    DATABASE=os.path.join(app.instance_path, 'mnist.sqlite')
)