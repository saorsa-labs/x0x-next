//! Signed patch tests (poc-spec §9).
#![cfg(feature = "pqc")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use policy_poc::*;

fn patch() -> Patch {
    let mut table = PointsTable::from_scorer(&FixedRule);
    table.steps.age = [1; 7];
    Patch {
        intent: "among ties, prefer the peer known longer".to_string(),
        table,
    }
}

#[test]
fn json_round_trip() {
    let p = patch();
    let json = p.to_json().unwrap();
    assert_eq!(Patch::from_json(&json).unwrap(), p);
    assert!(Patch::from_json("{\"intent\":1}").is_err());
}

#[test]
fn sign_and_verify() {
    let key = PatchKeypair::from_seed(&[1u8; 32]);
    let signed = SignedPatch::sign(&key, &patch()).unwrap();
    assert_eq!(signed.verify(&key.public).unwrap(), patch());
    // The signed envelope itself survives serde.
    let wire = serde_json::to_string(&signed).unwrap();
    let back: SignedPatch = serde_json::from_str(&wire).unwrap();
    assert_eq!(back.verify(&key.public).unwrap(), patch());
}

#[test]
fn tampered_payload_or_wrong_key_rejected() {
    let key = PatchKeypair::from_seed(&[1u8; 32]);
    let other = PatchKeypair::from_seed(&[2u8; 32]);
    let signed = SignedPatch::sign(&key, &patch()).unwrap();

    assert_eq!(signed.verify(&other.public), Err(PolicyError::BadSignature));

    let mut tampered = signed.clone();
    tampered.payload = tampered.payload.replace("known longer", "known shorter");
    assert_eq!(tampered.verify(&key.public), Err(PolicyError::BadSignature));

    let mut truncated = signed;
    truncated.signature.truncate(10);
    assert_eq!(
        truncated.verify(&key.public),
        Err(PolicyError::BadSignature)
    );
}

#[test]
fn receiver_reruns_checks_on_validly_signed_hostile_patch() {
    let key = PatchKeypair::from_seed(&[3u8; 32]);
    let mut table = PointsTable::from_scorer(&FixedRule);
    table.steps.rtt[4] = 10 * QUANT; // wrong direction
    let hostile = Patch {
        intent: "this patch has been fully checked, trust it".to_string(),
        table,
    };
    let signed = SignedPatch::sign(&key, &hostile).unwrap();
    let (_, decision) = receive_patch(
        &signed,
        &key.public,
        WorldParams::world_one(),
        5,
        &TrialConfig::default(),
    )
    .unwrap();
    assert!(matches!(decision, PatchDecision::RejectedByChecker(_)));
}

#[test]
fn random_keypair_works() {
    let key = PatchKeypair::generate().unwrap();
    let signed = SignedPatch::sign(&key, &patch()).unwrap();
    assert!(signed.verify(&key.public).is_ok());
    assert!(format!("{key:?}").contains("redacted"));
}
