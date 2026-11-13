#!/usr/bin/env python3
"""
PROTOTYPE / EXPERIMENT -- NOT PRODUCTION CODE, NOT PART OF x0x.

Arithmetic, not simulation: what post-quantum-only cryptography costs in bytes,
and what that means on a thin radio link. Sizes are the published parameter
sizes (FIPS 203, 204, 205; Falcon/FN-DSA draft). LoRa airtime uses the standard
Semtech SX127x formula.
"""
import json, math

SIZE = dict(
    mlkem768_ek=1184, mlkem768_ct=1088,
    mldsa65_pk=1952, mldsa65_sig=3309,
    mldsa44_pk=1312, mldsa44_sig=2420,
    slhdsa128s_pk=32, slhdsa128s_sig=7856,
    slhdsa192s_pk=48, slhdsa192s_sig=16224,   # category 3, to match ML-DSA-65 and ML-KEM-768
    fndsa512_pk=897, fndsa512_sig=666,   # FIPS 206 still a draft in Nov 2026
    hash=32, aead_tag=16, x25519=32, ed25519_sig=64,
)
S = SIZE


def lora_airtime_s(payload_bytes, sf, bw_hz=125_000, cr=1, preamble=8, crc=1, header=1):
    """Semtech AN1200.13 time-on-air."""
    tsym = (2 ** sf) / bw_hz
    de = 1 if (sf >= 11 and bw_hz == 125_000) else 0
    ih = 0 if header else 1
    n_payload = 8 + max(math.ceil((8 * payload_bytes - 4 * sf + 28 + 16 * crc - 20 * ih) / (4 * (sf - 2 * de))) * (cr + 4), 0)
    return (preamble + 4.25) * tsym + n_payload * tsym


def frames(total_bytes, mtu):
    return math.ceil(total_bytes / mtu)


out = {}

# --- 1. session handshakes (both sides authenticate; identities not yet cached)
hs = {
    "Signature-authenticated (ML-KEM-768 + 2 x ML-DSA-65 sig + 2 x pk)":
        S["mlkem768_ek"] + S["mlkem768_ct"] + 2 * S["mldsa65_sig"] + 2 * S["mldsa65_pk"],
    "Signature-authenticated, keys already cached":
        S["mlkem768_ek"] + S["mlkem768_ct"] + 2 * S["mldsa65_sig"],
    "KEM-authenticated, static KEM keys cached (3 x ML-KEM-768 exchange)":
        S["mlkem768_ek"] + 3 * S["mlkem768_ct"],
    "KEM-authenticated, keys not yet known (adds 2 static KEM keys)":
        S["mlkem768_ek"] + 3 * S["mlkem768_ct"] + 2 * S["mlkem768_ek"],
    "Resumption from a shared secret (two nonces + tags)":
        2 * (32 + S["aead_tag"]),
    "Classical reference (X25519 + 2 x Ed25519 sig + 2 x pk)":
        2 * S["x25519"] + 2 * S["ed25519_sig"] + 2 * 32,
}
out["handshake_bytes"] = hs

# --- 2. per-message authentication overhead
batch = 256
per_msg = {
    "Bare ML-DSA-65 signature on every message": S["mldsa65_sig"],
    "Bare ML-DSA-65 signature + public key (sender unknown)": S["mldsa65_sig"] + S["mldsa65_pk"],
    f"Merkle-batched signature, batch of {batch} (proof + amortised root sig)":
        S["hash"] * int(math.log2(batch)) + S["mldsa65_sig"] / batch,
    "Session MAC (AEAD tag) on an authenticated session": S["aead_tag"],
    "FN-DSA-512 signature (if and when FIPS 206 is final)": S["fndsa512_sig"],
}
out["per_message_auth_bytes"] = per_msg

# --- 3. a relayed message through h gossip hops, 200-byte payload
payload = 200
for hops in (1, 4, 8):
    out[f"bytes_on_wire_{hops}_hops_200B_payload"] = {
        "bare signature + key each hop": hops * (payload + S["mldsa65_sig"] + S["mldsa65_pk"]),
        "bare signature each hop": hops * (payload + S["mldsa65_sig"]),
        "session MAC each hop, batched end-to-end proof": hops * (payload + S["aead_tag"] + S["hash"] * 8),
        "session MAC each hop only": hops * (payload + S["aead_tag"]),
    }

# --- 4. capability chain of depth 3 (principal -> device -> agent)
out["capability_chain_depth3_bytes"] = {
    "first presentation (3 sigs + 3 keys)": 3 * (S["mldsa65_sig"] + S["mldsa65_pk"]),
    "first presentation, dual-family root (adds one SLH-DSA-128s sig)": 3 * (S["mldsa65_sig"] + S["mldsa65_pk"]) + S["slhdsa128s_sig"],
    "first presentation, dual-family root at category 3 (adds one SLH-DSA-192s sig and key)": 3 * (S["mldsa65_sig"] + S["mldsa65_pk"]) + S["slhdsa192s_sig"] + S["slhdsa192s_pk"],
    "later reference by hash": S["hash"],
}

# --- 4b. a signed weight patch for one decision point
# body: decision-point id (4) + hash of the weights it applies to (32) + hash of the result (32)
#       + 44 step weights as 16-bit integers (88) + a short statement of intent (~200 bytes of text)
patch_body = 4 + S["hash"] + S["hash"] + 44 * 2 + 200
out["weight_patch_bytes"] = {
    "patch body (ids, two hashes, 44 step weights, statement of intent)": patch_body,
    "patch with one ML-DSA-65 signature, author key referred to by hash": patch_body + S["mldsa65_sig"] + S["hash"],
    "patch with signature and the author's public key (author unknown)": patch_body + S["mldsa65_sig"] + S["mldsa65_pk"],
    "the same patch, FN-DSA-512 signature (if and when FIPS 206 is final)": patch_body + S["fndsa512_sig"] + S["hash"],
}

# --- 5. thin bearer: LoRa, EU868-style 1% duty cycle
# Frame sizes are bytes on air (the LoRa PHY payload). A LoRaWAN frame adds a 13-byte MAC
# header and integrity code, so "51 bytes of application payload at DR0" is 64 bytes on air.
lora = {}
for name, sf, mtu in (("SF7 / 125 kHz, 222-byte frames", 7, 222), ("SF9 / 125 kHz, 115-byte frames", 9, 115), ("SF12 / 125 kHz, 51-byte frames", 12, 51)):
    air = lora_airtime_s(mtu, sf)
    row = {"airtime_per_full_frame_s": round(air, 3)}
    for label, total in (("KEM-authenticated handshake", hs["KEM-authenticated, static KEM keys cached (3 x ML-KEM-768 exchange)"]),
                         ("KEM-authenticated first contact, keys not yet known", hs["KEM-authenticated, keys not yet known (adds 2 static KEM keys)"]),
                         ("signature-authenticated handshake, keys unknown", hs["Signature-authenticated (ML-KEM-768 + 2 x ML-DSA-65 sig + 2 x pk)"]),
                         ("resumption handshake", hs["Resumption from a shared secret (two nonces + tags)"]),
                         ("one 200-byte message with session MAC", payload + S["aead_tag"]),
                         ("one 200-byte message with bare ML-DSA-65 signature", payload + S["mldsa65_sig"])):
        n = frames(total, mtu)
        t = n * air
        row[label] = {"bytes": total, "frames": n, "airtime_s": round(t, 1), "elapsed_at_1pct_duty_min": round(t * 100 / 60, 1)}
    lora[name] = row
out["lora"] = lora
out["lorawan_note"] = {"bytes_on_air_for_51_bytes_of_lorawan_payload": 64,
                       "airtime_s_sf12_125khz": round(lora_airtime_s(64, 12), 3),
                       "airtime_s_sf12_125khz_51_bytes_on_air": round(lora_airtime_s(51, 12), 3)}

json.dump(out, open("pq_budget.json", "w"), indent=1)


def show(title, d, unit="B"):
    print(f"\n{title}")
    for k, v in d.items():
        print(f"  {v:>10,.0f} {unit}  {k}")


show("Handshake size, both directions", hs)
show("Per-message authentication overhead", per_msg)
for hops in (1, 4, 8):
    show(f"Bytes on the wire, 200-byte payload, {hops} hop(s)", out[f"bytes_on_wire_{hops}_hops_200B_payload"])
show("Capability chain, depth 3", out["capability_chain_depth3_bytes"])
show("A signed weight patch", out["weight_patch_bytes"])
print("\nLoRa (1% duty cycle)")
for name, row in lora.items():
    print(f"  {name}: {row['airtime_per_full_frame_s']} s per full frame")
    for k, v in row.items():
        if isinstance(v, dict):
            print(f"    {v['frames']:>4} frames  {v['airtime_s']:>7} s on air  {v['elapsed_at_1pct_duty_min']:>7} min elapsed   {k} ({v['bytes']} B)")
