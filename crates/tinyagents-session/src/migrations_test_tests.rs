use super::*;

/// The index of a migration **is** its version, so the list may only ever
/// be appended to. This pins the current length: bumping it is the moment
/// to re-read the module docs and confirm nothing was reordered.
#[test]
fn migration_list_is_append_only() {
    assert_eq!(
        MIGRATIONS.len(),
        8,
        "MIGRATIONS is append-only — adding one is fine, reordering or \
             deleting one silently re-numbers every later migration"
    );
}

#[test]
fn apply_is_idempotent_and_records_the_version() {
    let conn = Connection::open_in_memory().expect("open");
    apply(&conn).expect("first apply");
    let version: i64 = conn
        .query_row("SELECT version FROM schema_version WHERE id = 1", [], |r| {
            r.get(0)
        })
        .expect("read version");
    assert_eq!(version, MIGRATIONS.len() as i64 - 1);
    // Second run is a no-op and must not error.
    apply(&conn).expect("second apply");
}

/// A database created by the pre-migration DDL has the tables but no
/// version marker. Applying migrations must bring it forward rather than
/// failing on already-existing objects.
#[test]
fn apply_upgrades_a_pre_migration_database() {
    let conn = Connection::open_in_memory().expect("open");
    // Simulate the old world: migrations 0..=2 executed with no marker.
    for sql in &MIGRATIONS[..3] {
        conn.execute_batch(sql).expect("legacy ddl");
    }
    apply(&conn).expect("upgrade");
    // The version-3 index only exists because the migration ran.
    let exists: bool = conn
        .prepare(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_agent_teams_updated'",
        )
        .expect("prepare")
        .exists([])
        .expect("exists");
    assert!(
        exists,
        "migration 3 added the agent_teams(updated_at) index"
    );
}
