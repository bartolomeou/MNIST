function check_progress(job_id, progress_bar) {
	function worker() {
		$.get(`/progress/${job_id}`, function (data) {
			var progress = parseInt(data.progress);
			console.log(progress);
			if (progress < 100) {
				progress_bar.css('width', progress + '%');
				setTimeout(worker, 1000);
			} else {
				console.log('Job completed!');
				location.reload();
			}
		});
	}

	// start the progress-checking worker
	worker();
}

$(document).ready(function () {
	$('#newJob').submit(function (event) {
		event.preventDefault();
		// submit the form to create a new job
		$.post('/add_job', $('#newJob').serialize(), function (data) {
			// extract the generated job_id from the response
			var job_id = parseInt(data.job_id);
			console.log(job_id);
			if (!isNaN(job_id)) {
				var progress_bar = $('#progress-bar');
				// trigger check_progress immediately after creating the job
				check_progress(job_id, progress_bar);
			} else {
				alert('Failed to retrieve job ID.');
			}
		});
	});
});
