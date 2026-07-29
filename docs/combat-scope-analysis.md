# FightLab Combat Scope Analysis: Boxing vs Kickboxing vs Full Combat

**Version:** 1.1
**Date:** 2026-07-28
**Status:** Decision document (accepted)
**Author:** FightLab maintainer review
**Supersedes:** Any prior docstring/comment implying boxing-only

> **Decision (2026-07-28):** Kickboxing (punches + kicks, no grappling)
> is the launch ruleset. The platform is combat-agnostic. The 43-joint
> G1 model is used with finger joints locked (23-DoF control) for
> kickboxing; the full 43-joint model is retained for future MMA
> expansion. See the decision log at the bottom of this document.

---

## TL;DR (the recommendation)

**FightLab should be a combat-agnostic platform that ships with a
"full-combat (punches + kicks)" ruleset from day one, NOT boxing-only.**

Three points drive this:

1. **The platform is already combat-agnostic by design.** `RULESET.md`
   §7.1 already permits punches, kicks, elbows, knees, grappling, and
   takedowns. The eval harness, league, render pipeline, and miner SDK
   are all technique-agnostic; only the damage-contact geometry is
   boxing-specific. Restricting to punches is a *ruleset choice*, not an
   *architecture* choice — and the user's stated vision was always
   "autonomous humanoid combat," not boxing.

2. **Kicks are already proven on the G1.** Unitree staged a real-world
   G1 kickboxing match (May 2025, broadcast on Chinese state TV). HuB
   (arXiv:2505.07294) demonstrated a G1 holding a Bruce Lee high-kick
   pose on one leg under physical disturbance. KungfuBot ships
   `Side_kick.pkl` and `Roundhouse_kick.pkl` as example motions. The
   `exptech/g1-moves` dataset has 27 karate clips including kicks. The
   hardware+sim feasibility question is settled; what remains is
   training-difficulty and motion-data questions.

3. **The marginal cost of allowing kicks is small, and the platform
   upside is large.** Allowing kicks adds (a) ankle geoms as second
   weapon class (already done in the v1 `combat.py`), (b) ~2x more
   motion clips in the mocap library, (c) a 1.5x damage multiplier for
   kicks (already done), and (d) one-leg-balance risk that the v2
   tracker will learn anyway. In exchange, the league gets dramatically
   more dynamic fights, more strategic diversity, and a story that
   matches the real-world G1 kickboxing events the audience has
   already seen.

**The risk is NOT "can the G1 kick?" — it's "can our v2 tracker learn
to kick stably from limited mocap?"** That risk is real but it is a
*training-pipeline* risk, not a *scope* risk. We mitigate it by making
the *platform* combat-agnostic and the *first king* boxing-only if we
have to, then expanding. The platform should never be the bottleneck.

---

## 1. Technical Feasibility

### 1.1 How much harder is it to train kicking vs punching only?

**Tracking difficulty: kicks are harder, but not categorically so.**
Kicks require single-leg support during the strike, which means the
tracker must simultaneously (a) lift and extend one leg with high
velocity, (b) balance on the other leg, (c) recover to a stable stance.
Punches only require the legs to stay planted. This is a harder
tracking problem because the reference motion has higher dynamic range
and the support polygon is smaller.

KungfuBot's own contribution (adaptive tracking sigma) exists *because*
high-dynamic motions like kicks are hard to track with fixed
tolerances. Their paper explicitly motivates the adaptive curriculum
by the failure of prior methods (H2O, OmniH2O, ExBody2) on
"highly-dynamic" motions. So: kicking is the hard case that the
state-of-the-art was *designed for*. We're not inventing the difficulty;
we're using the method that solved it.

**Combat RL difficulty: kicks add a balance-failure mode.** A missed
kick or a kick checked by the opponent can topple the striker. The
terminal penalty (fall = lose) makes kicks higher-variance actions.
This is *good* for strategic diversity (high-risk/high-reward) but
*harder* for RL convergence because the policy must learn when NOT to
kick. RoboStriker's sandbag-warmup-then-self-play curriculum handles
this naturally — the bot learns kicks against a passive target first,
then learns the risk in self-play.

**Net:** kicking roughly doubles the motion-library size and adds one
balance-failure mode. With the v2 pipeline (DeepMimic tracker → latent
distillation → LS-NFSP), this is a 1.5-2x training time increase, not a
10x increase. It's a "more data + more seeds" cost, not an "open
research problem" cost.

### 1.2 Does RoboStriker's approach work for kicks, or is it boxing-specific?

**RoboStriker is boxing-specific in its motion library and reward
shape, but architecture-agnostic in its pipeline.**

What's boxing-specific in RoboStriker:
- **Motion library:** 46 professional boxing clips (jabs, hooks,
  uppercuts, slips, blocks, weaves). No kicks.
- **Goal observation (§C.1):** offensive target = relative position of
  *fists* to opponent torso; defensive target = opponent *fists* to
  ego torso. This is fist-centric (12D = 2 fists × 3D × 2 roles).
- **Hit reward (§C.3):** contact on ego *wrist* AND opponent *torso*
  with relative *punching* velocity > threshold. Wrist/torso only.
- **AMP discriminator:** trained on boxing mocap. Encodes boxing
  *style*.

What's technique-agnostic in RoboStriker:
- **The 3-stage pipeline** (track → distill → self-play). This works
  for any motion library.
- **The latent-space CVAE + hypersphere projection.** Motion-agnostic.
- **LS-NFSP.** Game-theoretic, technique-agnostic.
- **The hierarchical decoupling thesis.** "Decoupling balance
  maintenance from tactical exploration is a fundamental prerequisite."
  This is *more* true for kicks than punches — kicks need *more*
  balance decoupling, not less.

**To extend RoboStriker to kicks, you change three things:**
1. Add kick mocap to the motion library (Stage 0).
2. Generalize the goal observation from fists-only to
   `{fists, feet}` → opponent `{torso, head, legs}`. This roughly
   doubles the goal-obs dimension (12D → ~24D). Cheap.
3. Generalize the hit reward from wrist/torso to
   `{wrist, ankle}` striker geoms × `{torso, head, legs}` target geoms,
   with a velocity gate. This is exactly what v1's `combat.py` already
   does (weapons = wrists + ankles, damage zones = head/body/legs).

So: **RoboStriker is a boxing *instance* of a combat-agnostic
architecture.** The architecture transfers; the data and the contact
geometry are the delta. This is the strongest argument for going
beyond boxing: we'd be *under-using* the architecture by restricting it
to punches.

### 1.3 KungfuBot handles kicks (Side_kick example motion) — how well?

**Well enough to ship as a reference, with caveats.**

Verified facts:
- `repos/PBHC/example/motion_data/` contains `Side_kick.pkl` and
  `Roundhouse_kick.pkl` alongside `Hooks_punch.pkl` and
  `Horse-stance_punch.pkl`. These are the paper's showcase motions.
- KungfuBot's paper explicitly lists Kungfu (martial arts) and dancing
  as the target skill classes, with kicks being central to Kungfu.
- The paper's contribution (adaptive tracking sigma) was motivated by
  exactly these high-dynamic motions and claims significantly lower
  tracking error than H2O/OmniH2O/ExBody2 baselines on them.
- Real-world deployment on the G1 is shown in the paper's Figure 11
  ("more dynamic skills in the real world").

Caveats:
- KungfuBot is **CC-BY-NC-SA 4.0** — non-commercial. We can use the
  *code and approach* for research/the league, but cannot ship a
  commercial product on top of it without a license. The motion
  *processing pipeline* (retarget, filter, correct) is reusable; the
  *motions themselves* we should source independently (CMU, AMASS,
  g1-moves, custom) to stay license-clean.
- KungfuBot is a *tracker*, not a *combatant*. It tracks reference
  motions; it does not do self-play or contact damage. The combat layer
  (RoboStriker Stages 2-3) is still needed on top.

**Verdict:** KungfuBot proves the G1 can physically track kicks in sim
and on hardware. It is the *lower bound* of feasibility — if the tracker
works, combat RL on top of the tracker works (RoboStriker's thesis).

### 1.4 What does allowing kicks do to the action space complexity?

**Nothing, in the latent-space architecture. This is the key point.**

The v2 pipeline operates in a **32-D latent space** (RoboStriker §3.3),
not in raw joint space. The action the combat policy outputs is a
latent code; the decoder maps it to 23/29-DoF joint targets. Adding
kicks does NOT change the latent dimension — it changes the *contents*
of the latent space (the decoder now also has to reconstruct kick
motions). The MARL problem (LS-NFSP) stays 32-D.

What does change:
- **Motion library size:** roughly doubles (boxing clips + kick clips).
- **Decoder training:** more diverse target motions → harder
  reconstruction → may need a slightly larger decoder or more
  distillation epochs. 32-D is likely still enough; RoboStriker chose
  32-D to "capture boxing motion diversity" — adding kicks roughly
  doubles the diversity but 32-D has headroom (their t-SNE showed
  distinct clusters with room to spare).
- **Goal observation:** ~doubles (fists + feet, as above).
- **Raw joint-space action (if anyone trains in joint space, which v1
  did and v2 does NOT):** unchanged at 29-DoF. The joints already exist;
  kicks use the same hip/knee/ankle joints as walking. There is no new
  actuator.

**The v1 raw 17-D action (`[vel_3 | arm_14]`) is boxing-specific and
CANNOT kick** — it has no leg control beyond the velocity command.
This is a v1 limitation, not a platform limitation. v2 (latent space)
does not have this limitation. **This is the single biggest reason to
move to v2: v1 is structurally boxing-only; v2 is structurally
combat-agnostic.**

### 1.5 Balance risk: kicks require single-leg support — how stable is G1 for this?

**Stable enough, with the right policy. Not stable with a naive one.**

Evidence:
- **HuB (arXiv:2505.07294)** trained a G1 to hold Bruce Lee's high-kick
  pose (full leg extension, 1.5m, single-leg support) and the Swallow
  Balance (torso horizontal, T-pose on one leg) under physical
  disturbance (soccer ball strike). 97% success rate vs 0% for H2O and
  OmniH2O baselines. This is the existence proof.
- **KungfuBot** demonstrates dynamic kicks (Side_kick, Roundhouse) on
  the real G1 in Figure 11.
- **Unitree's own kickboxing event** (May 2025) showed four G1s kicking
  each other, sometimes falling but recovering. The falls were
  attributed to the G1 being "fairly slow" (punches lacked power, kicks
  missed) — i.e., a *policy quality* issue, not a *hardware
  capability* issue.

The risk is NOT "can the G1 stand on one leg" (HuB says yes). The risk
is "can a *combat* policy maintain single-leg balance while also
fighting." This is exactly what the hierarchical decoupling
(RoboStriker/KungfuBot) solves: the *low-level* tracker maintains
balance; the *high-level* latent policy picks the strike. If the
tracker can track a kick reference motion without falling, the combat
policy can *select* that kick without independently learning balance.

**The v2 tracker's job is to make kicks "free" at the combat level.**
If the tracker can track `Side_kick.pkl` for N steps without falling,
then the latent code that decodes to that kick is a safe action. The
combat policy doesn't need to learn single-leg balance — it needs to
learn *when* to invoke a kick latent code. This is the whole point of
the architecture.

### 1.6 Can MuJoCo handle foot-to-torso contact damage the same way as wrist-to-torso?

**Yes. MuJoCo's contact model is body-pair agnostic.**

MuJoCo computes contacts between any two geoms that have compatible
`contype`/`conaffinity` masks. The damage computation (relative
velocity × normal force) is identical regardless of which body pair
produces the contact. The only requirements:

1. **The foot geoms must have collision enabled.** In v1's
   `street_arena.py`, the mesh-mesh stripping (`contype=0` for
   non-essential bodies) historically disabled wrist collision and had
   to be re-enabled via `KEEP_COLLISION`. The same must be done for
   ankle geoms. This is a one-line config change, already done in v1's
   `combat.py` (ankle geoms added to `fist_geoms` as kick weapons).
2. **The foot geoms must be large enough for reliable contact.** The
   same `geom_margin=0.02` fix applied to wrists (because G1 arms are
   short) applies to ankles. The ankle geoms are larger than wrist
   geoms, so this is *easier* for kicks than punches.
3. **The contact force threshold must be tuned per weapon.** Kicks
   carry more mass and velocity, so a kick's force threshold can be
   *higher* than a punch's (filter out trivial foot-brush contacts
   while still registering real strikes). v1 uses a 1.5x damage
   multiplier for kicks, which approximates this.

**Caveat — foot-ground contacts are ever-present.** The feet are
always in contact with the floor during standing/walking. The damage
system must exclude foot-ground contacts from damage computation (only
foot-to-opponent contacts count). This is a `contype`/`conaffinity`
masking detail: foot geoms collide with the floor (locomotion) and
with the opponent (damage), but the damage filter only triggers on
opponent-body contacts. v1's `_detect_foul` already excludes
non-opponent contacts. **This is solved, not open.**

---

## 2. Motion Data Availability

### 2.1 Boxing mocap (the boxing-only path)

**Abundant.**

| Source | Content | Format | License |
|---|---|---|---|
| **CMU category 13** | 46 boxing clips (jabs, hooks, uppercuts, slips, blocks, footwork) | BVH | Research-only |
| **AMASS** (unifies CMU + 14 other datasets) | Boxing subset via CMU; unified SMPL | SMPL | Research |
| **RoboStriker's custom mocap** | 46 clips, 50Hz Xsens inertial, left-right mirrored to ~92 | Custom | Not public |
| **Mixamo** | Some boxing/martial arts animations | FBX | Adobe license |
| **`exptech/g1-moves`** | 27 karate clips (punches + kicks, already retargeted to G1 29-DoF) | NPZ/PKL | HuggingFace |

Boxing-only is the *easy* path: CMU 13 alone is enough, and the v2
pipeline (§v2-pipeline.md Phase 1) already specifies it as the primary
source.

### 2.2 Kickboxing/MMA mocap (the full-combat path)

**Available, but requires more assembly.**

| Source | Kick content | Format | Notes |
|---|---|---|---|
| **CMU category 85 (martial arts)** | Confirmed: "bicycle kick flip" (CMU-85-01, -02), breaking dances with kicks, martial arts walks. Mixed quality; not pure kickboxing. | BVH | Research-only. Needs filtering for kick-specific clips. |
| **`exptech/g1-moves` karate/** | 27 fighting motions incl. `B_AttackKarate`, `B_ChopsKarate`, `B_SpinKarate` — already retargeted to G1. Contains kicks. | NPZ | HuggingFace. **The single best kick source for G1 specifically.** |
| **KungfuBot example motions** | `Side_kick.pkl`, `Roundhouse_kick.pkl`, `Horse-stance_punch.pkl` — retargeted, physics-filtered. | PKL | CC-BY-NC-SA (code/motions). Usable as reference; source independent clips for the league. |
| **Szczęsna et al. 2021 (PMC7813879)** | 1,411 recordings, 3,229 single kicks and punches, C3D format. Karate: Mae-Geri (front), Mawashi-Geri (roundhouse), Ura-Mawashi (hook), Yoko-Geri (side). | C3D | Research. Needs SMPL conversion. High quality, kick-specific. |
| **Mixamo** | Some martial-arts/kickboxing animations | FBX | Adobe license |
| **AMASS martial arts subsets** | Limited; mostly in CMU subsets | SMPL | Research |
| **Custom mocap** | The gold standard (RoboStriker recorded their own) | Custom | Budget required |

**Assessment:** There is enough kick mocap to build a kick motion
library, but it is *more scattered* than boxing mocap. The
`exptech/g1-moves` karate clips are the fastest start (already on the
G1's 29-DoF), and CMU 85 + the Szczęsna dataset provide depth. A
full-combat motion library is ~2x the assembly effort of a boxing-only
library. **This is the main incremental cost of allowing kicks.**

### 2.3 Are there enough kicking motion sources?

**Yes, for a credible v2 launch.** The `exptech/g1-moves` karate
clips (27 motions, already retargeted) plus KungfuBot's two example
kicks give enough to train a tracker that can execute front, side, and
roundhouse kicks. CMU 85 and the Szczęsna dataset provide expansion
material for a richer library later.

**No, if you want the *diversity* of RoboStriker's boxing library
(46 clips) mirrored for kicks.** That would require custom mocap or a
significant filtering effort across CMU 85 + AMASS + Mixamo. For a v2
launch, the existing sources are sufficient; for a *mature* league,
budget for custom kick mocap.

### 2.4 Would we need significantly more motion data to cover kicks + punches vs punches alone?

**Roughly 2x more clips, but not 2x more *work per clip* because the
pipeline is the same.**

The v2 Stage 0 pipeline (CMU/AMASS → SMPL → filter → retarget → G1) is
the same for punches and kicks. Adding kicks means running more clips
through the same pipeline. The retargeting (Mink IK) is the same. The
physics filter (CoM-CoP stability) is the same — it actually *matters
more* for kicks (single-leg support), so the filter will reject more
kick clips than punch clips, which is correct behavior.

**Estimate:**
- Boxing-only: ~30-50 clips (CMU 13 + g1-moves punches).
- Full combat: ~60-100 clips (above + g1-moves karate + CMU 85 kicks +
  KungfuBot reference kicks + filtered AMASS/CMU martial arts).

This is a 1-2 week data-pipeline increase, not a multi-month effort.

---

## 3. Competition and Excitement

### 3.1 Boxing only

- **Simpler to train:** smaller motion library, no single-leg balance
  failure mode, proven by RoboStriker (η_hit 0.685, base stability
  0.942).
- **Easier to referee:** wrist-to-torso contact is unambiguous;
  foot-to-torso requires distinguishing kicks from stomps/trips.
- **Strategic:** ranges are fixed, defense is guard/slip/footwork,
  offense is jab/cross/hook/uppercut. Chess-like.
- **Audience:** boxing is a known sport; the "robot boxing" framing is
  immediately legible. RoboStriker's framing works.

### 3.2 Full combat (punches + kicks)

- **More dynamic:** head kicks, leg kicks, body kicks, spinning
  back kicks — much higher visual variety per bout.
- **More strategic diversity:** range management (kicks = long range,
  punches = short range), stance switching (southpaw vs orthodox vs
  bladed), high-risk/high-reward techniques.
- **More exciting to watch:** this is the UFC/K-1/GLORY appeal vs.
  pure boxing. A head-kick KO is the highlight-reel moment boxing
  can't produce.
- **Matches the real-world precedent:** Unitree's G1 kickboxing event
  (May 2025) set the audience expectation. A "robot combat" league
  that *doesn't* allow kicks would look regresssive next to what
  Unitree already demoed.
- **Harder to referee:** distinguishing a legal kick from a trip, a
  stomp from a kick, a knee from a collision. The damage zones
  (head/body/legs) and weapon classes (wrist/ankle) handle this, but
  edge cases exist (e.g., a shin-to-torso contact — is that a kick?).

### 3.3 Full combat (punches + kicks + grappling)

- **Most dynamic, hardest to train and referee.** Grappling requires
  sustained multi-contact control (clinching, joint forces) that the
  current contact model does not reward — it rewards *velocity-gated
  strikes*. Grappling would need a new reward class (positional
  control, submission-like termination).
- **MuJoCo can do it** (contact-rich manipulation is its strength),
  but the *policy* is much harder: grappling is a different motor
  skill than striking, with different balance requirements (two
  bodies in contact, combined CoM).
- **Recommendation: defer grappling.** Start with strikes (punches +
  kicks). Grappling is a Phase 2 expansion once the striking league
  is mature. The platform (RULESET §7.1) already permits it; the
  *first ruleset* doesn't have to.

### 3.4 What does the audience (Bittensor miners — tech community) want?

**The tech/ML audience wants the most impressive thing that works.**
A boxing league that works is more impressive than a kickboxing league
that doesn't. But a kickboxing league that works is more impressive
than a boxing league that works — and the G1 kickboxing precedent
(Unitree's event) means the audience has *already seen* G1s kick.

The Bittensor miner community is also a *builder* community: they will
train whatever the ruleset rewards. If the ruleset rewards kicks
(higher damage multiplier, longer reach), miners will train kicks. If
it rewards only punches, they'll train boxing. The ruleset *shapes*
what gets built. A combat-agnostic platform with a permissive
ruleset gives miners the most creative surface area, which is what
attracts builders to a subnet in the first place.

**Risk:** a permissive ruleset with weak refereeing invites degenerate
strategies (e.g., a "sumo" policy that just shoves, since shoving is
legal under §7.1). The damage gate (must deal ≥50 damage to win) and
the velocity-gated hit reward (no reward for shoving) are the
guardrails. They're already designed for this.

### 3.5 Is there a middle ground?

**Yes: punches + kicks, no grappling.** This is K-1/GLORY kickboxing,
not MMA. It's the natural middle ground:
- All the visual excitement of kicks.
- None of the refereeing/training complexity of grappling.
- A well-defined real-world sport (K-1 rules) to model the ruleset on.
- The v1 `combat.py` already implements exactly this (wrist + ankle
  weapons, no sustained-clinch reward).

**This is the recommended launch ruleset.**

---

## 4. Impact on Existing Components

### 4.1 `fight_env.py` (v1, currently wrist-to-torso only)

**Current state:** the v1 `fight_env.py` docstring says "Damage:
wrist-to-torso contact (punches only, no kicks)" and the 17-D action
(`[vel_3 | arm_14]`) has no leg control — it *cannot* kick. This is
the v1 walker-based stack.

**What changes for kicks:**
- The 17-D action is structurally boxing-only (no leg joints). v2
  replaces this with the 32-D latent space, which is technique-agnostic.
- The damage model must add ankle geoms as weapons (already done in
  v1's `combat.py`/`CombatJudge`, which is the *non-walker* stack).
- The foot-ground contact exclusion must be verified (done in v1's
  `_detect_foul`).

**Net:** v1's `fight_env.py` is boxing-only by action-space limitation.
v2 (latent space) removes this limitation. The damage-model changes
are already done in v1's combat rules. **The env change is "ship v2,"
not "patch v1."**

### 4.2 `eval_harness.py` (the referee)

**Current state:** the harness is technique-agnostic. It runs a
`BoutRunner` and consumes `SeedResult` (damage_dealt, winner, etc.).
It does not know what a "punch" or "kick" is — it only sees damage
totals and the damage gate.

**What changes for kicks:** Nothing in the harness itself. The
`env_config` hash captures the damage model (weapon geoms, target
geoms, multipliers), so a kick-allowing config is just a different
`env_config` with a different hash. The harness already supports
this via `BoutConfig.env_config`.

**Net: zero harness changes.** The harness was correctly designed to
be combat-agnostic. This is the strongest validation of the
"combat-agnostic platform" thesis: the referee doesn't care what
technique produced the damage.

### 4.3 `RULESET.md` (legal techniques)

**Current state:** §7.1 *already* permits punches, kicks, elbows,
knees, grappling, and takedowns. The ruleset was written
combat-agnostic. The boxing-only constraint was in the *environment*
(v1 `fight_env.py` docstring), not the ruleset.

**What changes for kicks:** Nothing in §7.1. The damage zones (§4.1,
§6) may need to explicitly enumerate kick damage multipliers if we
want them in the authoritative doc (currently in `env_config`).
§7.2 (illegal techniques) may need a "no stomping a downed opponent"
clause if we want to be precise, but §7.2 already covers "striking a
downed opponent."

**Net: minor doc clarifications.** The ruleset is already
permissive. The boxing-only framing was an env limitation, not a
ruleset limitation. **Fix the env, don't restrict the ruleset.**

### 4.4 Render pipeline

**No change.** The renderer already visualizes the full G1 body (all
29 DoF, all geoms). A kick is just the leg moving; the renderer shows
it. The v1 `combat.py` ShadowBoxer already has a `kick` action
(forward lunge) and the render pipeline shows it. Full-body rendering
was always the design.

### 4.5 League / SDK

**No change.** The league (ELO, round-robin, king) and the miner SDK
(obs/action interface, submission) are technique-agnostic. The SDK
documents the 41-D obs / 17-D action (v1) or the latent-space interface
(v2); neither knows what a kick is.

**One caveat:** the v1 SDK's 17-D action (`[vel_3 | arm_14]`) cannot
kick, so a v1 miner *cannot* submit a kicking policy. v2's latent
interface removes this. The SDK change is "ship v2 interface," not
"patch v1."

---

## 5. RoboStriker vs KungfuBot Architecture Implications

### 5.1 RoboStriker is boxing-specific — would we deviate from their proven recipe?

**We would extend it, not deviate from it.** The 3-stage architecture
(track → distill → self-play) is technique-agnostic. The
boxing-specific parts (motion library, fist-centric goal obs,
wrist/torso hit reward, boxing AMP) are *instances*, not *the
architecture*. Replacing the boxing instances with combat instances
(mixed motion library, fist+foot goal obs, wrist+ankle weapons,
combat AMP) is a straightforward extension that preserves the
proven pipeline.

The risk is that RoboStriker's *ablation numbers* (η_hit 0.685, base
stability 0.942) are for boxing. We don't have ablation numbers for
kickboxing. The architecture *should* transfer (the decoupling thesis
is more true for kicks, not less), but we won't know the absolute
numbers until we run it. This is a *research risk*, not an
*architecture risk*.

**Mitigation:** start the v2 tracker on boxing mocap (proven path),
then add kick mocap incrementally and re-distill the latent space.
This de-risks the tracker (boxing works) while extending to kicks
(progressive). The latent space can be re-distilled without retraining
the tracker.

### 5.2 KungfuBot is general — could we use their approach for full combat?

**Yes, for the tracker (Stage 1). No, for combat (Stages 2-3).**

KungfuBot is a *tracker* — it tracks reference motions. It is not a
combatant. It has no contact damage, no opponent, no self-play. To get
combat, you need RoboStriker's Stages 2-3 (latent distillation +
LS-NFSP) on top of a KungfuBot-quality tracker.

**The best architecture is a hybrid:**
- **Stage 1 (tracker):** KungfuBot's adaptive-sigma DeepMimic, trained
  on a mixed boxing+kickboxing motion library. This is strictly more
  general than RoboStriker's boxing-only tracker and is the
  state-of-the-art for high-dynamic motions (which kicks are).
- **Stages 2-3 (combat):** RoboStriker's latent distillation + LS-NFSP,
  generalized to fist+foot weapons and head/body/leg targets.

This is exactly what the v2-pipeline.md already specifies
(KungfuBot's adaptive sigma in Stage 1, RoboStriker's CVAE+NFSP in
Stages 2-3). The only change for full combat is the motion library
(Stage 0) and the contact geometry (in the combat env). The
architecture is unchanged.

### 5.3 Is there a paper that does humanoid kickboxing/MMA RL?

**No single paper does humanoid kickboxing/MMA RL end-to-end.**
The closest:
- **RoboStriker** — boxing only, the combat-RL pioneer.
- **KungfuBot** — tracks Kungfu (incl. kicks) but no combat/self-play.
- **HuB** — extreme balance (incl. high kicks) but no combat, no
  opponent.
- **OmniH2O / H2O / ExBody2** — general whole-body tracking, no combat.

**The gap FightLab fills:** combining KungfuBot's kick-capable
tracker with RoboStriker's combat RL, in a *competitive league*
setting. No published paper does exactly this. This is a feature, not
a bug — it means FightLab is novel, not a reproduction.

**Real-world precedent (not a paper):** Unitree's G1 kickboxing event
(May 2025) used AI-trained policies (per CCTV interview with Unitree
director Wang Qixin: "We used AI technology to train them"). The
policies are not public, but the *existence* of AI-trained G1
kickboxing is established. FightLab would be the *open, competitive,
sim-only* version of what Unitree demoed closed-source.

---

## 6. Honest Recommendation

### 6.1 What should FightLab be: boxing, kickboxing, or full combat?

**Kickboxing (punches + kicks, no grappling) at launch; full combat
(grappling added) as a later expansion.**

Reasoning:
- **Boxing-only is too narrow** for the user's stated vision
  ("autonomous humanoid combat, NOT boxing") and for the audience
  that has already seen G1 kickboxing. It under-uses the
  combat-agnostic platform and the combat-agnostic v2 architecture.
- **Full combat (with grappling) is too much at launch.** Grappling
  is a different motor skill with different refereeing needs. It
  doubles the training difficulty for unclear marginal excitement
  (the highlight-reel moments are strikes, not clinches).
- **Kickboxing (K-1/GLORY-style) is the sweet spot:** all the strike
  diversity, none of the grappling complexity, a well-defined
  real-world ruleset to model, and the v1 codebase already implements
  it (`combat.py` with wrist+ankle weapons).

### 6.2 Risk/reward tradeoff

| Dimension | Boxing-only | Kickboxing (recommended) | Full combat (later) |
|---|---|---|---|
| **Motion data effort** | Low (CMU 13) | Medium (+g1-moves karate, CMU 85) | High (+grappling mocap, scarce) |
| **Tracker training** | Proven (RoboStriker) | Likely works (KungfuBot shows kicks track) | Research risk (no precedent) |
| **Combat RL** | Proven (RoboStriker) | Likely works (same architecture, more reward terms) | Open (grappling reward design) |
| **Refereeing** | Easy | Medium (kick vs trip, damage zones) | Hard (clinch scoring, submissions) |
| **Audience appeal** | Moderate (known) | High (matches Unitree precedent) | High but risky (MMA is polarizing) |
| **Strategic diversity** | Low (fixed range) | High (range management, stance switching) | Highest (ground game) |
| **Highlight-reel potential** | Low | High (head kicks) | Highest (but rare) |
| **Platform agnosticism** | Under-uses it | Uses it | Fully uses it |
| **License risk** | Low | Low (source kicks independently of KungfuBot) | Medium (grappling mocap scarce) |

**The kickboxing column dominates boxing-only on every dimension
except "proven tracker."** That one risk is mitigated by starting the
v2 tracker on boxing mocap and adding kicks incrementally.

### 6.3 Starting with boxing and expanding later vs starting with full combat

**Start with boxing tracker, ship kickboxing ruleset.** This is the
de-risked path:

1. **Build the v2 tracker on boxing mocap first** (CMU 13). This is
   the proven RoboStriker path. Verify the tracker works (η_hit,
   base stability) on boxing.
2. **Add kick mocap to the motion library** (g1-moves karate,
   KungfuBot reference kicks). Re-distill the latent space to include
   kicks. The tracker does NOT need retraining if it was trained on a
   diverse enough library; only the CVAE (Stage 2) is re-distilled.
3. **Ship the kickboxing ruleset from day one.** The ruleset permits
   kicks; the *first king* may be a boxing-only policy if no one
   trains kicks yet. That's fine — the platform allows it, and miners
   will train kicks to gain the reach/damage advantage.
4. **Expand to grappling in a later phase** when the striking league
   is mature and the refereeing infrastructure is hardened.

This sequence means: the *platform* is combat-agnostic from day one;
the *first king* may be a boxer; the *league* rewards kicks from day
one; the *grappling* expansion is deferred. No re-architecture is
needed at any step — only motion-library and ruleset-config changes.

### 6.4 Can we design the platform to be combat-agnostic and start with boxing rules, then expand?

**Yes — and it's already mostly done.**

The platform is already combat-agnostic:
- `RULESET.md §7.1` permits all techniques.
- `eval_harness.py` is technique-agnostic (consumes damage totals).
- The league/SDK are technique-agnostic (ELO, submission).
- The render pipeline is full-body.

The only boxing-specific artifacts are:
- v1 `fight_env.py` docstring + 17-D action (replaced by v2 latent
  space).
- The v2-pipeline.md Stage 0 motion library spec (boxing mocap) —
  extend to include kick mocap.
- The goal observation (fist-centric) — generalize to fist+foot.
- The AMP discriminator (boxing mocap) — retrain on mixed mocap.

All four are *configuration/data* changes, not *architecture* changes.
The platform was designed to be combat-agnostic; the boxing-only
framing was a v1 convenience, not a v1 design choice. **Confirm the
platform is combat-agnostic by fixing the boxing-specific artifacts,
not by restricting the ruleset.**

### 6.5 Concrete action items (if this recommendation is accepted)

1. **Update `fight_env.py` docstring** (v2) to say "full combat:
   punches, kicks, spins" (already done in v1 `combat.py`; carry into
   v2).
2. **Extend v2-pipeline.md Stage 0** to include kick mocap sources
   (g1-moves karate, CMU 85, KungfuBot reference kicks) alongside
   boxing mocap (CMU 13).
3. **Generalize the v2 goal observation** (Stage 3a.2) from
   fists-only to fists+feet vs opponent torso/head/legs.
4. **Generalize the v2 hit reward** (Stage 3a.3) to wrist+ankle
   weapons with the v1 `combat.py` damage multipliers (kick 1.5x,
   head 2.0x, leg 0.5x).
5. **Retrain the AMP discriminator** on mixed boxing+kickboxing mocap.
6. **Add a "no stomping a downed opponent" clause** to RULESET §7.2
   for precision (kicks to a standing opponent are legal; kicks to a
   downed opponent during the count are not — already covered by
   §7.2 but worth being explicit now that kicks exist).
7. **Do NOT add grappling** to the launch ruleset. Document it as a
   Phase 2 expansion in ROADMAP.md.

---

## 7. What this document does NOT decide

- **Whether to train the first king with kicks or boxing-only.** That's
  a miner/training decision, not a platform decision. The platform
  permits both; the first king is whoever wins.
- **The exact kick damage multipliers.** v1 uses 1.5x (body kick) /
  2.0x (head kick) / 0.5x (leg kick). These are `env_config` values,
  tunable without ruleset changes.
- **Whether to allow spinning kicks / spinning back fists.** These are
  legal under §7.1 (any technique achievable in sim). The motion
  library (g1-moves has `B_SpinKarate`) and the tracker determine
  whether they're *learnable*, not the ruleset.
- **Sim-to-real.** FightLab is sim-only; sim-to-real is a Phase 4
  roadmap item. Kicks are harder sim-to-real (single-leg balance on
  hardware), but that's a future problem, not a launch problem.

---

## 8. References

1. **RoboStriker** — Yin et al., ICML 2026, arXiv:2601.22517.
   Boxing-specific 3-stage combat RL. Project:
   yinkangning0124.github.io/RoboStriker/
2. **KungfuBot/PBHC** — Xie et al., NeurIPS 2025, arXiv:2506.12851.
   Adaptive-sigma DeepMimic tracker for highly-dynamic motions
   (incl. Side_kick, Roundhouse_kick). Code: github.com/TeleHuman/PBHC
3. **HuB** — Zhang et al., arXiv:2505.07294. Extreme humanoid balance
   on G1, incl. Bruce Lee high-kick pose (single-leg, 1.5m, 97%
   success under disturbance). Project: hub-robot.github.io
4. **Unitree G1 kickboxing event** — Forbes, 2025-05-31. Real-world
   G1 kickboxing match, AI-trained policies, broadcast on CCTV.
   https://www.forbes.com/sites/johnkoetsier/2025/05/31/watch-the-worlds-first-humanoid-robot-kickboxing-match/
5. **`exptech/g1-moves`** — HuggingFace dataset. 60 retargeted G1
   29-DoF motions, 27 karate clips (incl. kicks). NPZ/PKL.
   https://huggingface.co/datasets/exptech/g1-moves
6. **CMU MoCap category 13 (boxing)** and **category 85 (martial
   arts)** — mocap.cs.cmu.edu. Boxing clips (cat 13) and martial
   arts/kick clips (cat 85, incl. bicycle kick flips).
7. **Szczęsna et al. 2021** — Optical mocap dataset of karate
   techniques, 1,411 recordings, 3,229 kicks/punches (Mae-Geri,
   Mawashi-Geri, Ura-Mawashi, Yoko-Geri). PMC7813879. C3D format.
8. **AMASS** — Mahmood et al., ICCV 2019. Unified SMPL motion dataset
   (15 mocap datasets). amass.is.tue.mpg.de
9. **FightLab v2 Pipeline** — docs/v2-pipeline.md. The 3-stage
   architecture this analysis extends.
10. **FightLab v1 combat.py** — implements wrist+ankle weapons,
    head/body/leg damage zones, kick 1.5x multiplier. The
    combat-agnostic damage model this analysis recommends carrying
    into v2.

---

## 9. Decision log

- **2026-07-27:** Analysis completed. Recommendation: combat-agnostic
  platform, kickboxing (punches + kicks) launch ruleset, defer
  grappling. Platform is already combat-agnostic; the boxing-only
  framing was a v1 env limitation, not a design choice. v2 latent
  space removes the limitation. Pending maintainer sign-off.
- **2026-07-28:** 43-joint decision. The G1 USD model has 43 joints
  (includes 14 finger joints across both hands). For the kickboxing
  ruleset, finger joints are locked, yielding a 23-DoF control space
  (the same 23-DoF used by RoboStriker/KungfuBot with wrists locked).
  The 43-joint model is retained for future MMA/grappling expansion
  where finger articulation (gripping, clinching) becomes relevant.
  Rationale: locking fingers now simplifies tracker training and
  matches the proven RoboStriker setup; keeping the full model avoids
  a future re-export when grappling is added in Phase 2+. The finger
  joints are locked via actuator configuration in the Isaac Lab
  tracker env (no policy change needed).
