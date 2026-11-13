#!/usr/bin/env python3
"""
PROTOTYPE / EXPERIMENT -- NOT PRODUCTION CODE, NOT PART OF x0x.

Times ML-DSA-65, ML-KEM-768 and SHA-256 on one core, for comparison with the
decision costs printed by decision_cost.c. Uses the PQClean reference build
shipped in the `pqcrypto` Python package, so optimised builds will be faster.
Read the ratios, not the absolute numbers. Nothing here was run on a Raspberry
Pi or a phone.

    python3 pq_timing.py        (prints the five lines appended to decision_cost.txt)
"""
import hashlib, os, time
from pqcrypto.sign import ml_dsa_65
from pqcrypto.kem import ml_kem_768


def bench(fn, n):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


msg = os.urandom(200)
pk, sk = ml_dsa_65.keygen()
sig = ml_dsa_65.sign(sk, msg)
ek, dk = ml_kem_768.keygen()
ct, ss = ml_kem_768.encaps(ek)
kib = os.urandom(1024)

print(f"mldsa65_sign_us {1e6 * bench(lambda: ml_dsa_65.sign(sk, msg), 200):.0f}")
print(f"mldsa65_verify_us {1e6 * bench(lambda: ml_dsa_65.verify(pk, msg, sig), 2000):.0f}   (pk {len(pk)} B, sig {len(sig)} B)")
print(f"mlkem768_encaps_us {1e6 * bench(lambda: ml_kem_768.encaps(ek), 2000):.0f}")
print(f"mlkem768_decaps_us {1e6 * bench(lambda: ml_kem_768.decaps(dk, ct), 2000):.0f}   (ek {len(ek)} B, ct {len(ct)} B)")
print(f"sha256_1KiB_us {1e6 * bench(lambda: hashlib.sha256(kib).digest(), 200000):.2f}")
