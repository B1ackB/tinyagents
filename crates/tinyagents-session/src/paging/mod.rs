//! One page of a filtered `SELECT`, plus the total number of matching rows.
//!
//! Every list endpoint over the session store does the same four things:
//! collect optional `WHERE` clauses with their bound values, count the matching
//! rows, bind `LIMIT`/`OFFSET` after the filter values, and map the page.
//! [`PagedQuery`] is that sequence once, so the placeholder numbering — the
//! part that silently breaks when one copy is edited and the others are not —
//! lives in one place.

use rusqlite::Connection;
use rusqlite::types::ToSql;

use tinyagents_harness::error::{Result, TinyAgentsError};

#[derive(Default)]
/// Filters for a paged query, each clause numbered after the values before it.
pub(crate) struct PagedQuery {
    clauses: Vec<String>,
    values: Vec<Box<dyn ToSql>>,
    limit: i64,
    offset: i64,
}

impl PagedQuery {
    /// Binds `value` and adds the clause `clause` builds from its placeholder
    /// number, for filters that are more than `column = value`.
    pub(crate) fn push(
        &mut self,
        value: impl ToSql + 'static,
        clause: impl FnOnce(usize) -> String,
    ) -> &mut Self {
        self.values.push(Box::new(value));
        self.clauses.push(clause(self.values.len()));
        self
    }

    /// Filters on `column = value` when `value` is `Some`.
    pub(crate) fn eq(&mut self, column: &str, value: Option<impl ToSql + 'static>) -> &mut Self {
        if let Some(value) = value {
            self.push(value, |n| format!("{column} = ?{n}"));
        }
        self
    }

    /// Filters on `column = value` when `value` is present and not blank.
    pub(crate) fn eq_nonblank(&mut self, column: &str, value: Option<&str>) -> &mut Self {
        self.eq(
            column,
            value
                .filter(|value| !value.trim().is_empty())
                .map(str::to_owned),
        )
    }

    /// Sets the page: at most `limit` rows, skipping the first `offset`.
    pub(crate) fn page(&mut self, limit: i64, offset: i64) -> Result<&mut Self> {
        if limit < 0 {
            return Err(TinyAgentsError::Storage(format!(
                "pagination limit must be non-negative: {limit}"
            )));
        }
        if offset < 0 {
            return Err(TinyAgentsError::Storage(format!(
                "pagination offset must be non-negative: {offset}"
            )));
        }
        self.limit = limit;
        self.offset = offset;
        Ok(self)
    }

    /// Counts every row of `from` matching the filters, then maps `columns`
    /// for the page of them in `order_by` order. Returns the page and the
    /// total count.
    pub(crate) fn fetch<T>(
        &mut self,
        conn: &Connection,
        from: &str,
        columns: &str,
        order_by: &str,
        map: impl FnMut(&rusqlite::Row<'_>) -> rusqlite::Result<T>,
    ) -> Result<(Vec<T>, u64)> {
        let where_sql = if self.clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", self.clauses.join(" AND "))
        };
        let total = conn.query_row(
            &format!("SELECT COUNT(*) FROM {from} {where_sql}"),
            self.params().as_slice(),
            |row| row.get::<_, i64>(0),
        )? as u64;
        let limit_idx = self.values.len() + 1;
        let offset_idx = self.values.len() + 2;
        let mut page_params = self.params();
        page_params.push(&self.limit);
        page_params.push(&self.offset);
        let mut stmt = conn.prepare(&format!(
            "SELECT {columns}
             FROM {from} {where_sql}
             ORDER BY {order_by}
             LIMIT ?{limit_idx} OFFSET ?{offset_idx}"
        ))?;
        let rows = stmt
            .query_map(page_params.as_slice(), map)?
            .collect::<rusqlite::Result<Vec<T>>>()?;
        Ok((rows, total))
    }

    fn params(&self) -> Vec<&dyn ToSql> {
        self.values.iter().map(|value| value.as_ref()).collect()
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
