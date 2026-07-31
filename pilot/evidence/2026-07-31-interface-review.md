# Findings

## Phase-1 findings (frozen)

**F1 · misconstructed premise · lines 116–132 · severity conclusion-breaking**

- **description:** A monocular PnP solve on a known, visually symmetric 1.5 m square can recover its plane and metric pose only after establishing 2D–3D corner correspondences; it does not establish which of the two plane-normal directions is the course's *approach* side. The contract supplies neither an asymmetric gate marker nor permitted directed gate geometry. Thus `normal_body` can be geometrically consistent while pointing to the exit side, contradicting the stated use of `pos_body + d * normal_body` as an approach point.
- **what changes:** The policy cannot safely use this field to make the claimed “enter at the normal” decision. The interface must either carry an independently observed/authorised approach-side cue and state its failure mode, or expose an unsigned plane normal and remove approach-point guidance from the contract.
- **status:** unchanged. The transcript repeats the PnP-to-approach-normal inference but provides no permitted directed gate cue or contract rule that selects the course approach side; `active_gate_index` identifies sequence, not a plane's directed normal.

**F2 · scope blindness · lines 97–105 and 223–234 · severity conclusion-weakening**

- **description:** `AUTO_ATTENTION` is specified to drive the full 3-D `target_dir_body` to the frame centre using yaw alone. A yaw rotation can reduce only the target's horizontal bearing; it cannot alter elevation. With the declared camera pitched 20° upward, a gate at the aircraft's altitude is intrinsically below the image centre even at zero bearing. No reachable yaw command can satisfy the stated servo objective.
- **what changes:** The advertised reduction to a yaw “free gimbal” needs a different, testable objective (for example, zero horizontal bearing only), and the policy/perception contract must retain responsibility for vertical framing. Otherwise servo tuning will chase an irreducible vertical error and the claimed camera-attention behaviour is not implementable.
- **status:** unchanged. The transcript independently describes the servo error as the target's *horizontal* bearing and identifies vertical framing as the hard constraint, which conflicts with the artifact's full-3-D-centering promise rather than resolving it.

**F3 · contract violation · lines 247–270 · severity conclusion-weakening**

- **description:** `Observation.to_vector()` unconditionally serializes every gate position/normal and every ribbon sample, although its contract says invalid entries are zeroed. For example, a `GateObs(valid=False, pos_body=[1, 2, 3])` produces those nonzero values in the learning vector; likewise `normal_valid=False` does not suppress `normal_body`. The flags are emitted, but the encoder itself does not enforce the represented-data invariant.
- **what changes:** A stale or partially failed producer can leak arbitrary pseudo-observations into a learned policy, making “not seen” dependent on producer discipline rather than this frozen interface. Mask values by their corresponding validity flags in this method (or make invalid payloads unrepresentable) so the surrogate and live paths share the promised encoding.
- **status:** unchanged. The transcript expressly says `to_vector()` zeroes invalid entries, but the implementation still serializes the payloads unconditionally; context therefore confirms the intended invariant but not its enforcement.

## New findings

**F4 · misconstructed premise · lines 62–63 and 116–120 · severity conclusion-weakening**

- **description:** `range_m = 480 / gate_px` is a pinhole relation for a specified fronto-parallel dimension, but `gate_px` is not defined and a gate's projected extent also changes with its unknown obliquity. The declared fallback uses centroid plus apparent size precisely when the quadrilateral/PnP fit has failed, so it has no pose estimate with which to remove foreshortening while still claiming a metre-valued `pos_body`.
- **what changes:** The fallback cannot be treated as a calibrated 3-D position measurement. It should expose bearing plus an explicitly uncertain/interval range (or just an image-size cue), reserving metric `pos_body` for a pose-bearing fit; otherwise range-based speed, handoff, and approach decisions receive systematically biased distance on oblique gates.

## Required interaction checks

- **Interaction capture:** Not found for F1–F4. The assistant introduced the PnP-normal and yaw-attention architecture before the user's later questions about entering gates normally and camera pointing; the user did not supply the unsupported full-3-D yaw or apparent-size premises.
- **Adopted error:** Not found for F1–F4. The relevant user messages are questions or requests for a policy interface, not demonstrably false assertions. The transcript's later user corrections concern stale VQ1 caution and sign verification, neither of which underlies these findings.

# Verdicts

**Prior art** — Autonomous, vision-based drone racing, learned visual control, and attention-guided racing are established rather than novel: [Kaufmann et al. (2018)](https://proceedings.mlr.press/v87/kaufmann18a.html) map images to waypoints and speed for agile racing, [Kim et al. (2020)](https://proceedings.mlr.press/v123/kim20b.html) completed simulated gate tracks with vision-based control, and [Pfeiffer et al. (2022)](https://arxiv.org/abs/2201.02569) specifically studied visual-attention prediction for racing agents. These do not eliminate the need for a VQ2-specific restricted-telemetry contract, but they make it an integration artifact rather than a new autonomy method.

**Decision relevance** — High. This boundary fixes the information that every controller and surrogate can consume; the directed-normal, yaw-attention, invalid-data, and range-fallback semantics directly determine whether a policy is trained on achievable and physically meaningful observations.

**Tractability** — Repairable with the available data and simulator. The smallest viable reformulation is to define a permitted directed-normal source (or make it unsigned), specify AUTO_ATTENTION as azimuth-only, enforce serialization masks, and downgrade the no-PnP fallback from metric pose to bearing/uncertain range. Those are interface and validation changes, not a new control project.

**Overall verdict: sound-with-repairs.**

