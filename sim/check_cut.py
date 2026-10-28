#!/usr/bin/env python3
"""Step 1 of the SHEAR sim plan: one stem, one leaf, a blade on each fingertip. Does the blade cut the
petiole only when it should, leave the stub it was placed for, and let the leaf drop?

  python check_cut.py --windowed     watch it: the jaw closes on the petiole, the blades cut it next to
                                     the stem and the leaf drops, then it starts over. The terminal
                                     prints every cut. Double-click the leaf, Ctrl + right-drag to pull it.
  python check_cut.py                headless checks, PASS/FAIL per check, exit 1 on a FAIL

Run in the lerobot env (it has mujoco), under the 9 GB cap:
  systemd-run --user --scope -q -p MemoryMax=9G -p MemorySwapMax=0 python check_cut.py
Writes nothing. Needs ./fetch_so101.sh once.
"""
import argparse
import sys
import time

import mujoco
import numpy as np

import plant as pl
import scene as sc

SETTLE_S = 1.0      # the leaf droops and stops swinging before the jaw moves
CLOSE_S = 0.4       # ⚠️ the jaw target ramps shut over this: a step command slams the blade into the stalk
HOLD_S = 1.1        # after the ramp: the cut, the leaf's fall, the jaw closing through
TAIL_S = 0.2        # the steady blade force is the median over the last this-many seconds
PULL_N = 20.0       # the "hard pull" check: a yank on the leaf about the size of the cut force (unverified)
STEM_TOUCH_N = 0.5  # gripper force on the stem that counts as touching it (unverified)
STUB_TOL = 0.001    # reported stub vs where the blade was placed
PAD_SHIFT = sc.BLADE_LEN / 2 + pl.PETIOLE_R + 0.001        # petiole 1 mm behind the blades' back end
MISS_SHIFT = -(sc.BLADE_LEN / 2 + pl.PETIOLE_R + 0.003)    # petiole 3 mm past the fingertips


def blade_gap(m, d):
    """Edge-to-edge distance between the blades along the closing axis (negative = pressed together)."""
    axis = d.xmat[m.body("arm/gripper").id].reshape(3, 3)[:, 0]
    f, mv = m.geom("arm/blade_fixed").id, m.geom("arm/blade_moving").id
    edge_f = d.geom_xpos[f] + axis * sc.BLADE_DEPTH / 2
    edge_m = d.geom_xpos[mv] - d.geom_xmat[mv].reshape(3, 3)[:, 0] * sc.BLADE_DEPTH / 2
    return float((edge_m - edge_f) @ axis)


def jaw_target(info, t):
    u = np.clip((t - SETTLE_S) / CLOSE_S, 0.0, 1.0)
    return info["jaw_open"] + (sc.GRIPPER_RANGE_RAD[0] - info["jaw_open"]) * u


def run_case(offset=0.002, shift=0.0, torque=sc.GRIPPER_TORQUE, cut_force=pl.CUT_FORCE_N, close=True,
             pull=0.0, leaf_len=pl.LEAF_LEN):
    m, d, info = sc.build(offset=offset, shift=shift, gripper_torque=torque, leaf_len=leaf_len)
    cut = pl.Cutter(m, d, info, cut_force)
    jaw, leaf, tip = m.actuator("arm/gripper").id, m.body("leaf").id, m.geom("leaflet_tip").id
    leaf_dofs = np.flatnonzero(m.body_rootid[m.dof_bodyid] == leaf)
    dt, end = m.opt.timestep, SETTLE_S + CLOSE_S + HOLD_S
    n_settle, n = round(SETTLE_S / dt), round(end / dt)
    out, tail, over, longest_over = {}, [], 0, 0
    for k in range(n):
        if close:
            d.ctrl[jaw] = jaw_target(info, d.time)
        if pull and d.time >= SETTLE_S:
            d.xfrc_applied[leaf, :3] = pull * info["a"]
        mujoco.mj_step(m, d)
        cut.step()
        over = over + 1 if cut.squeeze >= cut_force else 0
        longest_over = max(longest_over, over)
        if k == n_settle - 1:
            touching = [i for i in range(d.ncon) if {int(d.contact[i].geom1), int(d.contact[i].geom2)} & cut.robot_geoms
                        and {int(d.contact[i].geom1), int(d.contact[i].geom2)} & (set(cut.leaf_geoms) | {cut.stem})]
            out.update(droop=d.xpos[leaf][2] - d.geom_xpos[tip][2], settle_speed=float(np.abs(d.qvel[leaf_dofs]).max()),
                       z_settle=d.xpos[leaf][2], touch_at_rest=len(touching))
        if d.time > end - TAIL_S:
            tail.append(cut.squeeze)
    out.update(cut=cut.cut_t is not None, t_cut=cut.cut_t, stub=cut.stub, squeeze_ss=float(np.median(tail)),
               gap=blade_gap(m, d), leaf_drop=out["z_settle"] - d.xpos[leaf][2], attached=cut.attached,
               root_disp=float(np.linalg.norm(d.xpos[leaf] - info["S"])), blade_touch=cut.blade_touch,
               longest_over=longest_over * dt, **{f"{k}_peak": v for k, v in cut.peak.items()})
    return out


def run_checks(cut_force, leaf_len):
    fails = 0

    def report(ok, msg):
        nonlocal fails
        fails += not ok
        print(("PASS " if ok else "FAIL ") + msg)

    D = 2 * pl.PETIOLE_R
    jg = sc.jaw_geometry()
    q, _, gmat = sc.settled_arm()
    print(f"petiole {D * 1000:.0f} mm, stem {2 * pl.STEM_R * 1000:.0f} mm, leaf {leaf_len * 100:.0f} cm / "
          f"{pl.LEAF_MASS * 1000:.0f} g (unverified guesses). Blade edges stand {jg['protrude'] * 1000:.1f} mm "
          f"proud of the fingertips and meet at {sc.JAW_MEET_DEG:.0f} deg. Wrist rolled "
          f"{np.degrees(q['wrist_roll']):.2f} deg, hinge level to {np.degrees(np.arcsin(gmat[2, 1])):.3f} deg.\n")
    run = lambda **kw: run_case(leaf_len=leaf_len, **kw)

    h = run(close=False)
    report(h["attached"] and h["droop"] > 0.02 and h["settle_speed"] < 0.05 and not h["touch_at_rest"],
           f"the leaf hangs from the stem: its tip droops {h['droop'] * 1000:.0f} mm, leaf speed "
           f"{h['settle_speed']:.3f} after {SETTLE_S:.0f} s, {h['touch_at_rest']} robot contacts")

    h50 = run(cut_force=np.inf)
    h25 = run(cut_force=np.inf, torque=sc.GRIPPER_TORQUE_STALLED)
    report(min(h50["gap"], h25["gap"]) >= 0.8 * D and h50["attached"] and h25["attached"],
           f"a blade that can't cut holds the petiole: the edges stop {h50['gap'] * 1000:.1f} / "
           f"{h25['gap'] * 1000:.1f} mm apart on the {D * 1000:.0f} mm petiole (50 / 25 % torque), nothing tunnels")
    s50, s25 = h50["squeeze_ss"], h25["squeeze_ss"]
    report(s50 > s25 > 0, f"blade force on the petiole, sim estimate from the 12 V servo's stall torque (an upper "
                          f"bound): {s50:.1f} N at 50 % torque, {s25:.1f} N at 25 % (stalled servo)")
    T = 0.5 * (s50 + s25)
    print(f"     the mechanism checks below cut at {T:.1f} N, between the two torque levels")

    for off in (0.002, 0.008):
        c = run(offset=off, cut_force=T)
        report(c["cut"], f"blade {off * 1000:.0f} mm off the stem: " +
               (f"cut at t = {c['t_cut']:.2f} s" if c["cut"] else f"no cut (blades pressed {c['squeeze_peak']:.1f} N)"))
        if not c["cut"]:
            continue
        report(abs(c["stub"] - off) <= STUB_TOL, f"stub {c['stub'] * 1000:.2f} mm for a blade placed "
                                                 f"{off * 1000:.0f} mm off the stem (tolerance {STUB_TOL * 1000:.0f} mm)")
        if off == 0.002:
            report(c["leaf_drop"] > 0.05 and not c["attached"], f"the cut leaf falls clear: dropped {c['leaf_drop'] * 100:.0f} cm")
            report(c["gap"] < 0.0005, f"the jaw closes through the cut: edges {c['gap'] * 1000:.2f} mm apart at the end")
            report(c["stem_force_peak"] < STEM_TOUCH_N,
                   f"gripper on the stem: peak {c['stem_force_peak']:.2f} N (limit {STEM_TOUCH_N} N). Load from "
                   f"the leaf on the stem peaked at {c['weld_force_peak']:.1f} N, mostly the blade landing on a rigid petiole")

    low = run(cut_force=T, torque=sc.GRIPPER_TORQUE_STALLED)
    report(not low["cut"], f"25 % torque: the blades settle at {low['squeeze_ss']:.1f} N, under the {T:.1f} N cut "
                           f"force. Their {low['squeeze_peak']:.0f} N landing spike stays over it {low['longest_over'] * 1000:.0f} ms, "
                           f"short of the {pl.CUT_DWELL_S * 1000:.0f} ms stroke: no cut")
    pad = run(cut_force=T, shift=PAD_SHIFT)
    report(not pad["cut"] and not pad["blade_touch"] and pad["pad_squeeze_peak"] > T,
           f"petiole behind the blades: the pads squeeze it with {pad['pad_squeeze_peak']:.1f} N (over the "
           f"{T:.1f} N cut force), the blades never touch it, no cut")
    miss = run(cut_force=T, shift=MISS_SHIFT)
    report(not miss["cut"] and not miss["blade_touch"] and miss["gap"] < 0.0005,
           f"petiole past the fingertips: the jaw closes on air (edges {miss['gap'] * 1000:.2f} mm apart), no cut")
    p = run(close=False, pull=PULL_N)
    report(p["attached"] and p["root_disp"] < 0.002,
           f"pulled with {PULL_N:.0f} N: the leaf stays on (it never snaps), its base moved "
           f"{p['root_disp'] * 1000:.2f} mm, weld force {p['weld_force_peak']:.1f} N")

    print(f"INFO cut-force placeholder {cut_force:.0f} N (unverified): the stock gripper would "
          f"{'' if s50 >= cut_force else 'NOT '}cut at 50 % torque and {'' if s25 >= cut_force else 'NOT '}at 25 %. "
          f"Test a real petiole against {s50:.0f} / {s25:.0f} N.")
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAIL'}")
    return fails


def windowed(offset, cut_force, torque, leaf_len):
    import mujoco.viewer
    m, d, info = sc.build(offset=offset, gripper_torque=torque, leaf_len=leaf_len)
    cut = pl.Cutter(m, d, info, cut_force)
    jaw = m.actuator("arm/gripper").id
    cycle = SETTLE_S + CLOSE_S + HOLD_S + 1.5
    print(f"Blade plane {offset * 1000:.0f} mm off the stem, cut force {cut_force:.1f} N per edge (unverified), "
          f"gripper torque {torque:.2f} N.m.\nThe jaw closes {SETTLE_S:.0f} s in; every {cycle:.0f} s it starts over.\n")
    last = 0.0
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = info["B"]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.30, 150, -20
        while v.is_running():
            t_wall = time.time()
            with v.lock():
                d.ctrl[jaw] = jaw_target(info, d.time)
                mujoco.mj_step(m, d)
                cut.step()
                if cut.just_cut:
                    print(f"\n>>> cut at {cut.squeeze_at_cut:.1f} N: stub {cut.stub * 1000:.1f} mm "
                          f"(blade placed {offset * 1000:.0f} mm off the stem)")
                if d.time >= cycle:
                    if cut.cut_t is None:
                        print(f"\n--- no cut: the blades settled at {cut.squeeze:.1f} N against a {cut_force:.1f} N "
                              f"cut force (the {cut.peak['squeeze']:.0f} N landing spike is shorter than a stroke)")
                    print(f"    peak gripper force on the stem {cut.peak['stem_force']:.2f} N, "
                          f"peak leaf pull on the stem {cut.peak['weld_force']:.2f} N")
                    mujoco.mj_resetData(m, d)
                    d.qpos[:], d.ctrl[:] = info["q0"], info["ctrl0"]
                    cut.reset()
                    mujoco.mj_forward(m, d)
            if t_wall - last > 0.2:
                last = t_wall
                print(f"\rblade force {cut.squeeze:6.2f} N   pads {cut.pad_squeeze:6.2f} N   "
                      f"leaf on stem {cut.weld_force:6.2f} N   ", end="", flush=True)
            v.sync()
            wait = sc.TIMESTEP - (time.time() - t_wall)
            if wait > 0:
                time.sleep(wait)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windowed", action="store_true", help="watch the cut in a window (it loops)")
    ap.add_argument("--offset-mm", type=float, default=2.0, help="windowed: blade plane to stem surface, mm")
    ap.add_argument("--cut-force", type=float, default=pl.CUT_FORCE_N,
                    help="N each blade edge must press with; unverified placeholder (default %(default)s)")
    ap.add_argument("--torque", choices=("50", "25"), default="50", help="windowed: gripper torque, %% of max")
    ap.add_argument("--leaf-len", type=float, default=pl.LEAF_LEN, help="stem surface to leaf tip, m (unverified)")
    a = ap.parse_args()
    if a.windowed:
        torque = sc.GRIPPER_TORQUE if a.torque == "50" else sc.GRIPPER_TORQUE_STALLED
        windowed(a.offset_mm / 1000, a.cut_force, torque, a.leaf_len)
    else:
        sys.exit(1 if run_checks(a.cut_force, a.leaf_len) else 0)


if __name__ == "__main__":
    main()
