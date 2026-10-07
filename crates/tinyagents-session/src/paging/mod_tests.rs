//! Unit tests for [`PagedQuery`].

use super::*;
use rusqlite::Connection;

fn table() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE items (id INTEGER, kind TEXT, rank INTEGER);
         INSERT INTO items VALUES (1, 'a', 3), (2, 'b', 1), (3, 'a', 2), (4, 'a', 5);",
    )
    .unwrap();
    conn
}

fn ids(query: &mut PagedQuery, limit: i64, offset: i64) -> (Vec<i64>, u64) {
    query
        .page(limit, offset)
        .fetch(&table(), "items", "id", "rank DESC", |row| row.get(0))
        .unwrap()
}

#[test]
fn counts_every_match_and_returns_one_page_in_order() {
    let mut query = PagedQuery::default();
    query.eq("kind", Some("a"));

    assert_eq!(ids(&mut query, 2, 0), (vec![4, 1], 3));
}

#[test]
fn the_offset_is_bound_after_the_filters() {
    let mut query = PagedQuery::default();
    query.eq("kind", Some("a"));

    assert_eq!(ids(&mut query, 2, 2), (vec![3], 3));
}

#[test]
fn absent_and_blank_filters_add_no_clause() {
    let mut query = PagedQuery::default();
    query
        .eq("kind", None::<String>)
        .eq_nonblank("kind", Some("  "))
        .eq_nonblank("kind", None);

    assert_eq!(ids(&mut query, 10, 0), (vec![4, 1, 3, 2], 4));
}

#[test]
fn custom_clauses_use_their_placeholder_number() {
    let mut query = PagedQuery::default();
    query
        .eq_nonblank("kind", Some("a"))
        .push(2_i64, |n| format!("rank > ?{n}"));

    assert_eq!(ids(&mut query, 10, 0), (vec![4, 1], 2));
}

#[test]
fn repeated_fetch_does_not_accumulate_pagination_parameters() {
    let conn = table();
    let mut query = PagedQuery::default();
    query.eq("kind", Some("a"));

    query.page(2, 0);
    assert_eq!(
        query
            .fetch(&conn, "items", "id", "rank DESC", |row| row.get(0))
            .unwrap(),
        (vec![4, 1], 3)
    );

    query.page(1, 2);
    assert_eq!(
        query
            .fetch(&conn, "items", "id", "rank DESC", |row| row.get(0))
            .unwrap(),
        (vec![3], 3)
    );
}
