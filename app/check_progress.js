function check_progress(job_id, progress_bar) {
	function worker() {
		$.get(`/progress/${job_id}`, function (data) {
			if (progress < 100) {
				progress_bar.set_progress(progress);
				setTimeout(worker, 1000);
			}
		});
	}

	worker();
}
