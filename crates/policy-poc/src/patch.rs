//! Weight patches and their signatures (design study §5, "patches are signed
//! data, re-checked on arrival"; poc-spec §9).
//!
//! A [`Patch`] is an intent string plus a points table, serialized with
//! serde_json. With the default-on `pqc` feature, a [`SignedPatch`] carries
//! the exact JSON bytes and an ML-DSA-65 signature over them (via
//! `saorsa-pqc`). A receiving node verifies the signature and then **re-runs
//! every check itself** — the static checker and a trial in its own twin —
//! via [`evaluate_patch`]. Nothing the patch claims about itself is trusted;
//! the intent string is shown to people and never read by code.

use serde::{Deserialize, Serialize};

use crate::checker::{CheckFailure, CheckReport, Checker};
use crate::error::PolicyError;
use crate::table::PointsTable;
use crate::trial::{self, TrialConfig, TrialReport, Verdict};
use crate::twin::WorldParams;

/// A proposed policy: what it is meant to do, and the weights.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Patch {
    /// Human-readable intent. Never trusted by code.
    pub intent: String,
    /// The proposed weights.
    pub table: PointsTable,
}

impl Patch {
    /// Canonical JSON bytes of the patch.
    ///
    /// # Errors
    /// [`PolicyError::Serialization`] if serialization fails.
    pub fn to_json(&self) -> Result<String, PolicyError> {
        serde_json::to_string(self).map_err(|e| PolicyError::Serialization(e.to_string()))
    }

    /// Parse a patch from JSON.
    ///
    /// # Errors
    /// [`PolicyError::Serialization`] if the JSON is malformed.
    pub fn from_json(json: &str) -> Result<Self, PolicyError> {
        serde_json::from_str(json).map_err(|e| PolicyError::Serialization(e.to_string()))
    }
}

/// What a receiving node decided about a patch.
#[derive(Debug, Clone, PartialEq)]
pub enum PatchDecision {
    /// Passed the static checks and won its trial.
    Adopted {
        /// Static check report.
        check: CheckReport,
        /// Trial report.
        trial: TrialReport,
    },
    /// Failed the static checks; never simulated.
    RejectedByChecker(Vec<CheckFailure>),
    /// Passed the static checks but lost (or did not win) its trial.
    RejectedByTrial {
        /// Static check report.
        check: CheckReport,
        /// Trial report.
        trial: TrialReport,
    },
}

/// Re-run every check on a patch: the static checker, then a trial in this
/// node's own twin.
///
/// # Errors
/// [`PolicyError::InvalidConfig`] if the trial config is invalid.
pub fn evaluate_patch(
    patch: &Patch,
    world: WorldParams,
    seed: u64,
    config: &TrialConfig,
) -> Result<PatchDecision, PolicyError> {
    let check = match Checker::check(&patch.table) {
        Ok(c) => c,
        Err(failures) => return Ok(PatchDecision::RejectedByChecker(failures)),
    };
    let report = trial::run(&patch.table, world, seed, config)?;
    Ok(if report.verdict == Verdict::Adopt {
        PatchDecision::Adopted {
            check,
            trial: report,
        }
    } else {
        PatchDecision::RejectedByTrial {
            check,
            trial: report,
        }
    })
}

#[cfg(feature = "pqc")]
pub use signed::{receive_patch, PatchKeypair, SignedPatch, SIGNING_CONTEXT};

#[cfg(feature = "pqc")]
mod signed {
    use saorsa_pqc::api::sig::{
        ml_dsa_65, MlDsaPublicKey, MlDsaSecretKey, MlDsaSignature, MlDsaVariant,
    };
    use serde::{Deserialize, Serialize};

    use super::{evaluate_patch, Patch, PatchDecision};
    use crate::error::PolicyError;
    use crate::trial::TrialConfig;
    use crate::twin::WorldParams;

    /// ML-DSA context string binding signatures to this patch format.
    pub const SIGNING_CONTEXT: &[u8] = b"x0x-next/policy-poc/patch/v1";

    /// An ML-DSA-65 key pair for signing patches.
    #[derive(Clone)]
    pub struct PatchKeypair {
        /// Verification key, shared with receiving nodes.
        pub public: MlDsaPublicKey,
        secret: MlDsaSecretKey,
    }

    impl std::fmt::Debug for PatchKeypair {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("PatchKeypair")
                .field("public_len", &self.public.to_bytes().len())
                .field("secret", &"<redacted>")
                .finish()
        }
    }

    impl PatchKeypair {
        /// A fresh key pair from the operating system's random source.
        ///
        /// # Errors
        /// [`PolicyError::Crypto`] if key generation fails.
        pub fn generate() -> Result<Self, PolicyError> {
            let (public, secret) = ml_dsa_65()
                .generate_keypair()
                .map_err(|e| PolicyError::Crypto(e.to_string()))?;
            Ok(Self { public, secret })
        }

        /// A deterministic key pair from a 32-byte seed (demos and tests).
        #[must_use]
        pub fn from_seed(seed: &[u8; 32]) -> Self {
            let (public, secret) = ml_dsa_65().generate_keypair_from_seed(seed);
            Self { public, secret }
        }

        /// Encoded verification key.
        #[must_use]
        pub fn public_bytes(&self) -> Vec<u8> {
            self.public.to_bytes()
        }
    }

    /// A patch's exact JSON bytes plus an ML-DSA-65 signature over them.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct SignedPatch {
        /// The signed JSON text of the [`Patch`].
        pub payload: String,
        /// ML-DSA-65 signature over `payload` with [`SIGNING_CONTEXT`].
        pub signature: Vec<u8>,
    }

    impl SignedPatch {
        /// Serialize and sign `patch`.
        ///
        /// # Errors
        /// [`PolicyError::Serialization`] or [`PolicyError::Crypto`].
        pub fn sign(keypair: &PatchKeypair, patch: &Patch) -> Result<Self, PolicyError> {
            let payload = patch.to_json()?;
            let sig = ml_dsa_65()
                .sign_with_context(&keypair.secret, payload.as_bytes(), SIGNING_CONTEXT)
                .map_err(|e| PolicyError::Crypto(e.to_string()))?;
            Ok(Self {
                payload,
                signature: sig.to_bytes(),
            })
        }

        /// Verify the signature with `public`, then parse the signed bytes.
        /// The returned patch is exactly what was signed.
        ///
        /// # Errors
        /// [`PolicyError::BadSignature`] if the signature is malformed or does
        /// not verify; [`PolicyError::Serialization`] if the signed bytes are
        /// not a patch.
        pub fn verify(&self, public: &MlDsaPublicKey) -> Result<Patch, PolicyError> {
            let sig = MlDsaSignature::from_bytes(MlDsaVariant::MlDsa65, &self.signature)
                .map_err(|_| PolicyError::BadSignature)?;
            let ok = ml_dsa_65()
                .verify_with_context(public, self.payload.as_bytes(), &sig, SIGNING_CONTEXT)
                .map_err(|_| PolicyError::BadSignature)?;
            if !ok {
                return Err(PolicyError::BadSignature);
            }
            Patch::from_json(&self.payload)
        }
    }

    /// What a receiving node does: verify the signature, then re-run every
    /// check (checker + trial) itself.
    ///
    /// # Errors
    /// [`PolicyError::BadSignature`] for a bad signature, or any error from
    /// parsing or the trial configuration.
    pub fn receive_patch(
        signed: &SignedPatch,
        public: &MlDsaPublicKey,
        world: WorldParams,
        seed: u64,
        config: &TrialConfig,
    ) -> Result<(Patch, PatchDecision), PolicyError> {
        let patch = signed.verify(public)?;
        let decision = evaluate_patch(&patch, world, seed, config)?;
        Ok((patch, decision))
    }
}
