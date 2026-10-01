use super::*;

#[test]
fn profiler_reports_a_bounded_region() {
    let profiler = ProcessProfiler::start(Duration::from_millis(1));
    std::thread::sleep(Duration::from_millis(3));
    let profile = profiler.finish();

    assert!(profile.wall_time_ms >= 2.0);
    #[cfg(target_os = "linux")]
    assert!(profile.sample_count >= 2);
    #[cfg(not(target_os = "linux"))]
    assert_eq!(profile.sample_count, 0);
    if let (Some(baseline), Some(peak)) = (profile.baseline_rss_bytes, profile.peak_rss_bytes) {
        assert!(peak >= baseline);
    }
}
