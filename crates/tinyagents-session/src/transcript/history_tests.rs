use super::*;

#[test]
fn parent_components_never_pop_the_filesystem_root() {
    assert_eq!(
        absolute_normalized_path(Path::new("/a/../..")),
        PathBuf::from("/")
    );
}

#[test]
fn parent_components_collapse_normal_segments() {
    assert_eq!(
        absolute_normalized_path(Path::new("/a/b/../c/./d")),
        PathBuf::from("/a/c/d")
    );
}
