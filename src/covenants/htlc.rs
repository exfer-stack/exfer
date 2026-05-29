//! Hash Time-Locked Contract (HTLC) for atomic swaps.
//!
//! Two spending paths:
//! 1. **Hash path**: receiver reveals preimage + signs (before timeout)
//! 2. **Timeout path**: sender reclaims after block height exceeds timeout

use serde::{Deserialize, Serialize};

use super::builder::ScriptBuilder;
use crate::script::ast::{Combinator, Program};
use crate::script::serialize::{deserialize_program, serialize_program};
use crate::script::value::Value;
use crate::types::hash::Hash256;

/// Create an HTLC script.
///
/// **Hash path** witness: `[Left(Unit)][preimage_bytes][msg][sig_receiver]`
/// **Timeout path** witness: `[Right(Unit)][msg][sig_sender]`
pub fn htlc(
    sender_key: &[u8; 32],
    receiver_key: &[u8; 32],
    hash_lock: &Hash256,
    timeout_height: u64,
) -> Program {
    let mut b = ScriptBuilder::new();

    // Hash path: hash_eq(lock) AND sig_check(receiver)
    let hash_check = b.hash_eq(hash_lock);
    let receiver_check = b.sig_check(receiver_key);
    let hash_path = b.and(hash_check, receiver_check);

    // Timeout path: height_gt(timeout) AND sig_check(sender)
    let timeout_check = b.height_gt(timeout_height);
    let sender_check = b.sig_check(sender_key);
    let timeout_path = b.and(timeout_check, sender_check);

    // Dispatch based on witness selector
    let selector = b.witness();
    let case_node = b.case(hash_path, timeout_path);
    let _root = b.comp(selector, case_node);
    b.build()
}

/// The four parameters that uniquely identify an HTLC template instance —
/// the return shape of [`try_parse_htlc`], the inverse of [`htlc`].
///
/// Binary `[u8; 32]` fields serialize as lowercase hex strings to match the
/// convention used elsewhere in the JSON-RPC surface (see `src/rpc.rs`). All
/// four fields are protocol-bound widths, so they are genuinely fixed-size.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct HtlcParams {
    #[serde(with = "hex_bytes32_serde")]
    pub sender: [u8; 32],
    #[serde(with = "hex_bytes32_serde")]
    pub receiver: [u8; 32],
    #[serde(with = "hex_bytes32_serde")]
    pub hash_lock: [u8; 32],
    pub timeout_height: u64,
}

/// Attempts to decode an output script as an HTLC produced by [`htlc`].
///
/// Returns the original four parameters if the script structurally matches
/// the canonical template. The strategy:
///
/// 1. Deserialize the script to a [`Program`] AST.
/// 2. Walk every node and collect candidate constants: exactly two 32-byte
///    `Bytes` values (the pubkeys), one `Hash` (the hashlock), one `U64`
///    (the timeout).
/// 3. For each of the two possible (sender, receiver) orderings,
///    reconstruct what [`htlc`] would produce and compare the resulting
///    serialised bytes to the input.
///
/// Equality of the serialised bytes is sufficient because both
/// [`ScriptBuilder::build`] and [`serialize_program`] are deterministic,
/// so two structurally identical programs always serialise to identical
/// byte sequences. This makes [`try_parse_htlc`] robust against scripts
/// that happen to embed the right constants in the wrong structural
/// positions — they will fail the byte comparison.
///
/// Returns `None` if the script is malformed, contains the wrong number
/// or types of constants, or its structure does not match the template.
pub fn try_parse_htlc(script_bytes: &[u8]) -> Option<HtlcParams> {
    let program = deserialize_program(script_bytes).ok()?;

    // Collect candidate constants.
    let mut pubkeys: Vec<[u8; 32]> = Vec::new();
    let mut hash_lock_candidate: Option<Hash256> = None;
    let mut timeout_candidate: Option<u64> = None;

    for node in &program.nodes {
        let Combinator::Const(v) = node else { continue };
        match v {
            Value::Bytes(b) if b.len() == 32 => {
                if pubkeys.len() >= 2 {
                    // Template only contains two 32-byte Bytes constants.
                    return None;
                }
                let mut arr = [0u8; 32];
                arr.copy_from_slice(b);
                pubkeys.push(arr);
            }
            Value::Hash(h) => {
                if hash_lock_candidate.is_some() {
                    return None;
                }
                hash_lock_candidate = Some(*h);
            }
            Value::U64(n) => {
                if timeout_candidate.is_some() {
                    return None;
                }
                timeout_candidate = Some(*n);
            }
            // Other constants (e.g. Value::Bool(false) from boolean and())
            // are part of the template and ignored here. Disallowing them
            // would break this check the moment the template adds another
            // bool short-circuit; we rely on the final byte-compare to
            // reject anything that isn't the exact template.
            _ => {}
        }
    }

    if pubkeys.len() != 2 {
        return None;
    }
    let hash_lock = hash_lock_candidate?;
    let timeout_height = timeout_candidate?;

    let make_candidate = |sender: [u8; 32], receiver: [u8; 32]| HtlcParams {
        sender,
        receiver,
        hash_lock: hash_lock.0,
        timeout_height,
    };

    let try_candidate = |c: &HtlcParams| -> bool {
        let reconstructed = htlc(
            &c.sender,
            &c.receiver,
            &Hash256(c.hash_lock),
            c.timeout_height,
        );
        serialize_program(&reconstructed) == script_bytes
    };

    let candidate_a = make_candidate(pubkeys[0], pubkeys[1]);
    if try_candidate(&candidate_a) {
        return Some(candidate_a);
    }
    let candidate_b = make_candidate(pubkeys[1], pubkeys[0]);
    if try_candidate(&candidate_b) {
        return Some(candidate_b);
    }
    None
}

// ---------------------------------------------------------------------------
// Serde helper: `[u8; 32]` <-> lowercase hex string.
// ---------------------------------------------------------------------------

mod hex_bytes32_serde {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8; 32], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&hex::encode(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 32], D::Error> {
        let s = String::deserialize(d)?;
        let v = hex::decode(s).map_err(serde::de::Error::custom)?;
        if v.len() != 32 {
            return Err(serde::de::Error::custom(format!(
                "expected 32-byte hex value, got {} bytes",
                v.len()
            )));
        }
        let mut out = [0u8; 32];
        out.copy_from_slice(&v);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed_pubkey(seed: u8) -> [u8; 32] {
        let mut k = [0u8; 32];
        for (i, b) in k.iter_mut().enumerate() {
            *b = (i as u8).wrapping_add(seed).wrapping_mul(7);
        }
        k
    }

    fn fixed_hash(seed: u8) -> Hash256 {
        let mut h = [0u8; 32];
        for (i, b) in h.iter_mut().enumerate() {
            *b = (i as u8).wrapping_add(seed).wrapping_mul(13);
        }
        Hash256(h)
    }

    #[test]
    fn try_parse_htlc_roundtrip_recovers_all_params() {
        let sender = fixed_pubkey(1);
        let receiver = fixed_pubkey(2);
        let hash_lock = fixed_hash(3);
        let timeout = 1_234_567u64;

        let program = htlc(&sender, &receiver, &hash_lock, timeout);
        let bytes = serialize_program(&program);

        let parsed = try_parse_htlc(&bytes).expect("template script must parse");
        assert_eq!(parsed.sender, sender);
        assert_eq!(parsed.receiver, receiver);
        assert_eq!(parsed.hash_lock, hash_lock.0);
        assert_eq!(parsed.timeout_height, timeout);
    }

    #[test]
    fn try_parse_htlc_distinguishes_sender_from_receiver() {
        // Sender and receiver play structurally different roles. Swapping
        // them produces a different script — and the parsed result must
        // reflect the actual on-chain assignment, not be ambiguous.
        let a = fixed_pubkey(10);
        let b = fixed_pubkey(20);
        let lock = fixed_hash(30);

        let prog_ab = htlc(&a, &b, &lock, 1000);
        let bytes_ab = serialize_program(&prog_ab);
        let prog_ba = htlc(&b, &a, &lock, 1000);
        let bytes_ba = serialize_program(&prog_ba);

        assert_ne!(
            bytes_ab, bytes_ba,
            "HTLC is asymmetric in sender/receiver"
        );

        let parsed_ab = try_parse_htlc(&bytes_ab).unwrap();
        assert_eq!(parsed_ab.sender, a);
        assert_eq!(parsed_ab.receiver, b);

        let parsed_ba = try_parse_htlc(&bytes_ba).unwrap();
        assert_eq!(parsed_ba.sender, b);
        assert_eq!(parsed_ba.receiver, a);
    }

    #[test]
    fn try_parse_htlc_rejects_empty_input() {
        assert_eq!(try_parse_htlc(&[]), None);
    }

    #[test]
    fn try_parse_htlc_rejects_random_bytes() {
        let garbage = [0xAAu8; 64];
        assert_eq!(try_parse_htlc(&garbage), None);
    }

    #[test]
    fn try_parse_htlc_rejects_unrelated_program() {
        let prog = Program::single(Combinator::Unit);
        let bytes = serialize_program(&prog);
        assert_eq!(try_parse_htlc(&bytes), None);
    }

    #[test]
    fn try_parse_htlc_rejects_tampered_byte() {
        let sender = fixed_pubkey(5);
        let receiver = fixed_pubkey(6);
        let lock = fixed_hash(7);
        let prog = htlc(&sender, &receiver, &lock, 500);
        let mut bytes = serialize_program(&prog);

        // Flip the high bit of a byte in the middle of the payload. Either
        // the script will fail to deserialize, or it will deserialize to a
        // structure that the byte-compare step rejects, or — if the flipped
        // byte happens to land inside one of the parameter constants —
        // we'll get back a *different* HtlcParams. In none of those cases
        // should the original four-tuple be returned.
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0x80;

        match try_parse_htlc(&bytes) {
            None => {}
            Some(parsed) => {
                let unchanged = parsed.sender == sender
                    && parsed.receiver == receiver
                    && parsed.hash_lock == lock.0
                    && parsed.timeout_height == 500;
                assert!(
                    !unchanged,
                    "tampered script must not parse to the original parameters"
                );
            }
        }
    }

    #[test]
    fn htlc_params_serde_rejects_short_pubkey() {
        let bad_json = r#"{
            "sender": "abcd",
            "receiver": "0000000000000000000000000000000000000000000000000000000000000001",
            "hash_lock": "0000000000000000000000000000000000000000000000000000000000000002",
            "timeout_height": 100
        }"#;
        assert!(serde_json::from_str::<HtlcParams>(bad_json).is_err());
    }
}
