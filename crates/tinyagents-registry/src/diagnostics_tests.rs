use super::*;
use crate::component::ComponentId;

#[test]
fn to_dot_escapes_quotes_and_backslashes_in_component_ids() {
    let snapshot = RegistrySnapshot {
        components: vec![ComponentMetadata {
            id: ComponentId(r#"weird"name\with\backslashes"#.to_string()),
            kind: ComponentKind::Tool,
            description: None,
            tags: Vec::new(),
            aliases: Vec::new(),
        }],
        aliases: Vec::new(),
    };

    let dot = snapshot.to_dot();

    // The raw id must never appear unescaped inside the DOT output, or a
    // malicious/unlucky component name could break out of its quoted
    // string and inject arbitrary DOT syntax.
    assert!(!dot.contains("\"weird\"name"));
    assert!(dot.contains(r#"weird\"name\\with\\backslashes"#));
}
