"""Read-only probe for two candidate Unitree Dex1 DDS layouts.

Creates subscribers only. Topic presence is not hardware identity validation.
Run inside an existing unitree_sdk2_python environment on the robot.
"""
import argparse
import json
import math
import threading
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interface", default="eth0")
    p.add_argument("--layout", choices=["external", "internal"], required=True)
    p.add_argument("--seconds", type=float, default=5.0)
    args = p.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 60:
        p.error("--seconds must be in (0, 60]")
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    ChannelFactoryInitialize(0, args.interface)
    lock = threading.Lock()
    result = {"left": {"messages": 0}, "right": {"messages": 0}}
    subscribers = []

    def record(side, value):
        value = float(value)
        if not math.isfinite(value):
            return
        with lock:
            item = result[side]
            item["messages"] += 1
            item["last_q"] = value
            item["min_q"] = min(item.get("min_q", value), value)
            item["max_q"] = max(item.get("max_q", value), value)

    if args.layout == "external":
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
        def handler(side):
            def callback(message):
                if len(message.states) > 0:
                    record(side, message.states[0].q)
            return callback
        for side in ("left", "right"):
            sub = ChannelSubscriber(f"rt/dex1/{side}/state", MotorStates_)
            sub.Init(handler(side), 10)
            subscribers.append(sub)
    else:
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        def callback(message):
            if len(message.motor_state) >= 34:
                record("left", message.motor_state[31].q)
                record("right", message.motor_state[33].q)
        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(callback, 10)
        subscribers.append(sub)
    try:
        time.sleep(args.seconds)
    finally:
        for sub in subscribers:
            sub.Close()
    with lock:
        print(json.dumps({"source": "read_only_candidate_dds_probe",
                          "layout": args.layout, "duration_s": args.seconds,
                          "hardware_binding_verified": False, "result": result,
                          "note": "No publisher created. Zero q is valid; absence of change does not establish identity."},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
