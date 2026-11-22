//! Reading and writing graph6, the one-line-per-graph format that nauty's
//! geng writes.
//!
//! A line is the node count N(n), then the upper triangle of the adjacency
//! matrix, column by column ((0,1), (0,2), (1,2), (0,3), ...), packed six bits
//! per byte, each byte offset by 63 so the line is printable ASCII. See
//! nauty's formats.txt. An optional `>>graph6<<` header is skipped.

use std::fmt;

use crate::ball::{Edge, Node};

/// Why a line is not valid graph6.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Graph6Error {
    Empty,
    /// A byte outside 63..=126, at this position.
    BadByte(usize),
    /// sparse6 (`:`) or digraph6 (`&`), which this reader does not handle.
    OtherFormat,
    /// The line has `found` data bytes where n needs `expected`.
    WrongLength {
        expected: usize,
        found: usize,
    },
    /// More nodes than a `Node` can number.
    TooManyNodes(u64),
}

impl fmt::Display for Graph6Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Graph6Error::Empty => write!(f, "empty line"),
            Graph6Error::BadByte(i) => write!(f, "byte {i} is not in the graph6 range 63..=126"),
            Graph6Error::OtherFormat => write!(f, "sparse6 or digraph6, not graph6"),
            Graph6Error::WrongLength { expected, found } => {
                write!(f, "expected {expected} data bytes, found {found}")
            }
            Graph6Error::TooManyNodes(n) => write!(f, "{n} nodes is too many"),
        }
    }
}

impl std::error::Error for Graph6Error {}

/// The node count and edges of one graph6 line, without its line break.
/// Edges come out normalized (smaller node first), sorted by larger node.
pub fn parse_graph6(line: &[u8]) -> Result<(usize, Vec<Edge>), Graph6Error> {
    let line = line.strip_prefix(b">>graph6<<").unwrap_or(line);
    match line.first() {
        None => return Err(Graph6Error::Empty),
        Some(b':' | b'&') => return Err(Graph6Error::OtherFormat),
        Some(_) => {}
    }
    if let Some(i) = line.iter().position(|&b| !(63..=126).contains(&b)) {
        return Err(Graph6Error::BadByte(i));
    }
    let value = |bytes: &[u8]| {
        bytes
            .iter()
            .fold(0u64, |acc, &b| acc << 6 | u64::from(b - 63))
    };
    let short = |len: usize| Graph6Error::WrongLength {
        expected: len,
        found: line.len(),
    };
    let (n, data) = if line[0] < 126 {
        (u64::from(line[0] - 63), &line[1..])
    } else if line.len() >= 4 && line[1] < 126 {
        (value(&line[1..4]), &line[4..])
    } else if line.len() >= 8 {
        (value(&line[2..8]), &line[8..])
    } else {
        return Err(short(8));
    };
    if n > u64::from(Node::MAX) {
        return Err(Graph6Error::TooManyNodes(n));
    }
    let n = n as usize;
    let bits = n * n.saturating_sub(1) / 2;
    let expected = bits.div_ceil(6);
    if data.len() != expected {
        return Err(Graph6Error::WrongLength {
            expected,
            found: data.len(),
        });
    }
    let mut edges = Vec::new();
    let mut k = 0;
    for j in 1..n {
        for i in 0..j {
            if (data[k / 6] - 63) >> (5 - k % 6) & 1 == 1 {
                edges.push((i as Node, j as Node));
            }
            k += 1;
        }
    }
    Ok((n, edges))
}

/// One graph6 line, without a line break, for the graph on `0..n` with these
/// edges (either orientation, no repeats or self-loops).
///
/// graph6 stores one bit per node pair, about n^2/12 bytes, so this is for
/// small graphs.
///
/// # Panics
///
/// If an edge is a self-loop or has a node outside `0..n`, or n is too large
/// for graph6 (2^36 or more).
pub fn write_graph6(n: usize, edges: &[Edge]) -> String {
    assert!(n < 1 << 36, "{n} nodes is too many for graph6");
    let mut out: Vec<u8> = Vec::new();
    if n < 63 {
        out.push(n as u8 + 63);
    } else if n < 258_048 {
        out.push(126);
        out.extend((0..3).rev().map(|i| (n >> (6 * i) & 63) as u8 + 63));
    } else {
        out.extend([126, 126]);
        out.extend((0..6).rev().map(|i| (n >> (6 * i) & 63) as u8 + 63));
    }
    let header = out.len();
    let bits = n * n.saturating_sub(1) / 2;
    out.resize(header + bits.div_ceil(6), 0);
    for &(u, v) in edges {
        let (i, j) = (u.min(v) as usize, u.max(v) as usize);
        assert!(
            i != j && j < n,
            "edge ({u}, {v}) is not valid for {n} nodes"
        );
        let k = j * (j - 1) / 2 + i;
        out[header + k / 6] |= 1 << (5 - k % 6);
    }
    for byte in &mut out[header..] {
        *byte += 63;
    }
    String::from_utf8(out).expect("graph6 is ASCII")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_graphs() {
        // Examples from nauty's formats.txt and NetworkX's to_graph6_bytes.
        assert_eq!(parse_graph6(b"@"), Ok((1, vec![])));
        assert_eq!(parse_graph6(b"Bw"), Ok((3, vec![(0, 1), (0, 2), (1, 2)])));
        assert_eq!(parse_graph6(b"Ch"), Ok((4, vec![(0, 1), (1, 2), (2, 3)])));
        assert_eq!(
            parse_graph6(b">>graph6<<Bw"),
            Ok((3, vec![(0, 1), (0, 2), (1, 2)]))
        );
        assert_eq!(parse_graph6(b"?"), Ok((0, vec![])));
    }

    #[test]
    fn rejects_bad_lines() {
        assert_eq!(parse_graph6(b""), Err(Graph6Error::Empty));
        assert_eq!(parse_graph6(b":Fa@x^"), Err(Graph6Error::OtherFormat));
        assert_eq!(parse_graph6(b"B w"), Err(Graph6Error::BadByte(1)));
        assert_eq!(
            parse_graph6(b"Bww"),
            Err(Graph6Error::WrongLength {
                expected: 1,
                found: 2
            })
        );
        assert_eq!(
            parse_graph6(b"C"),
            Err(Graph6Error::WrongLength {
                expected: 1,
                found: 0
            })
        );
    }

    #[test]
    fn writes_what_it_reads() {
        assert_eq!(write_graph6(3, &[(0, 1), (2, 0), (1, 2)]), "Bw");
        assert_eq!(write_graph6(4, &[(0, 1), (1, 2), (2, 3)]), "Ch");
        assert_eq!(write_graph6(1, &[]), "@");
        assert_eq!(write_graph6(0, &[]), "?");
        // Round trips, including the 4-byte node count past 62 nodes.
        let mut state = 9u64;
        for n in [2usize, 5, 17, 62, 63, 100] {
            let mut edges = Vec::new();
            for j in 1..n {
                for i in 0..j {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    if state >> 62 == 0 {
                        edges.push((i as Node, j as Node));
                    }
                }
            }
            let line = write_graph6(n, &edges);
            let (m, mut parsed) = parse_graph6(line.as_bytes()).expect("valid");
            parsed.sort_unstable();
            edges.sort_unstable();
            assert_eq!((m, parsed), (n, edges), "n = {n}");
        }
    }

    #[test]
    fn long_form_node_count() {
        // 63 nodes needs the 4-byte form: 126 then 18 bits.
        let mut line = vec![126, 63, 63, 63 + 63];
        line.extend(std::iter::repeat_n(63u8, (63 * 62 / 2usize).div_ceil(6)));
        let (n, edges) = parse_graph6(&line).expect("valid");
        assert_eq!((n, edges.len()), (63, 0));
    }
}
