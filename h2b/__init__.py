"""Hand2Body: generate whole-body SMPL motion from a single left-hand 12D signal.

Pipeline:  table-tennis hand generator  ->  Hand2Body  ->  GMR retarget  ->  HoloMotion (Unitree G1).
See docs/CONTRACT.md for the inter-stage data contract.
"""

__version__ = "0.0.1"
