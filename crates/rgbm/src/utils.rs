pub fn build_thread_pool(n_jobs: isize) -> Option<rayon::ThreadPool> {
    let n_threads = match n_jobs {
        n if n <= 0 => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        n => n as usize,
    };
    if n_threads > 1 {
        Some(rayon::ThreadPoolBuilder::new().num_threads(n_threads).build().unwrap())
    } else {
        None
    }
}
