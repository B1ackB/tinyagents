# Archive inspection

`inspect_archive` lists ZIP, TAR and TAR.GZ entries from supplied bytes. Entry
names are untrusted metadata; no filesystem objects are created and links are
never followed. The public API is re-exported through `multimodal::archive` and
`multimodal` for compatibility.

| File | Responsibility |
| --- | --- |
| `mod.rs` | Module wiring and deliberate public exports |
| `types.rs` | Formats, caller budgets, entries, errors and truncation |
| `ops.rs` | Listing, CRC validation and bounded decompression |
| `zip_admission.rs` | Shared allocation-free ZIP preflight and admitted reader |
| `ops_tests.rs` | Synthetic malformed, hostile and bounded archive fixtures |

ZIP metadata preflight occurs before the eager parser allocates its index.
Office MIME probing uses the same internal admission path. Input, metadata,
entry, name and expansion ceilings remain enforced; oversized archives report
explicit truncation. TAR.GZ inflation is bounded once, and GNU/PAX records are
listed rather than followed into unbounded names. A partial listing validates
only the processed prefix. See the parent README for detailed limits and the
conservative rejection of ZIP footer signatures in metadata/comments.
