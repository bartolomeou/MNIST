from flask import (Blueprint, flash, g, redirect, render_template, request, url_for, jsonify)
from werkzeug.exceptions import abort

from server.db import get_db
from server.job_thread import JobThread

bp = Blueprint('experiment', __name__, static_folder='../client/static', template_folder='../client/templates')
job_threads = dict()
job_progress = dict()


def progress_callback(job_id, progress):
    job_progress[job_id] = progress


@bp.route('/')
def index():
    db = get_db()
    jobs = db.execute('SELECT * FROM job ORDER BY accuracy ASC').fetchall()

    return render_template('result.html', jobs=jobs)


@bp.route('/add_job', methods=['POST'])
def add_job():
    batch_size = int(request.form['batch-size'])
    learning_rate = float(request.form['learning-rate'])
    epochs = int(request.form['number-of-epochs'])

    #TODO: check duplicated jobs

    db = get_db()
    cursor = db.execute('INSERT INTO job (batch_size, learning_rate, epochs) VALUES (?,?,?)', (batch_size, learning_rate, epochs))
    job_id = cursor.lastrowid
    db.commit()

    job_threads[job_id] = JobThread(job_id, batch_size, learning_rate, epochs, progress_callback)
    job_threads[job_id].start()

    return jsonify({'job_id': job_id})


@bp.route('/progress/<int:job_id>')
def progress(job_id):
    return jsonify({'progress': job_progress.get(job_id, 0)})
    