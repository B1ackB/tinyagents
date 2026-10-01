use super::rand01;

#[test]
fn rand01_stays_inside_the_unit_interval() {
    for _ in 0..10_000 {
        let value = rand01();
        assert!((0.0..1.0).contains(&value), "out of range: {value}");
    }
}

#[test]
fn rand01_is_not_a_constant() {
    let first = rand01();
    // A stuck generator (the bug this module exists to avoid) would return
    // the same value forever.
    assert!(
        (0..64).any(|_| rand01() != first),
        "generator produced a constant sequence"
    );
}

#[test]
fn rand01_covers_both_halves_of_the_interval() {
    let mut low = false;
    let mut high = false;
    for _ in 0..1_000 {
        if rand01() < 0.5 {
            low = true;
        } else {
            high = true;
        }
    }
    assert!(low && high, "generator never crossed the midpoint");
}
