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

#[inline(always)]
pub(crate) fn prefetch<T>(ptr: *const T) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // pldl1keep: prefetch for read, L1 cache, temporal locality.
        // Inline asm used because stdarch_aarch64_prefetch is still unstable.
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) ptr, options(nostack, readonly));
    }

    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }
}