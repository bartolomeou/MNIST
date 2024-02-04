from flask import (Blueprint, flash, g, redirect, render_template, request, url_for)
from werkzeug.exceptions import abort

from app.db import get_db
from app.job_thread import JobThread

bp = Blueprint('experiment', __name__)
job_threads = {}


@bp.route('/')
def index():
    db = get_db()
    jobs = db.execute('SELECT * FROM job ORDER BY accuracy ASC').fetchall()

    return render_template('result.html', jobs=jobs)


@bp.route('/add_job', methods=['POST'])
def add_job():
    batch_size = request.form['batch-size']
    learning_rate = request.form['learning-rate']
    epochs = request.form['number-of-epochs']

    #TODO: check duplicated jobs

    db = get_db()
    cursor = db.execute('INSERT INTO job (batch_size, learning_rate, epochs) VALUES (?,?,?)', (batch_size, learning_rate, epochs))
    job_id = cursor.lastrowid
    db.commit()

    # process_job(batch_size, learning_rate, epochs, job_id)
    job_threads[job_id] = JobThread()
    job_threads[job_id].start()

    return redirect(url_for('experiment.index'))
    # return 'job id: #%s' % job_id
    