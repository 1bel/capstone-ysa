"""
verify_labeling.py
-------------------
Manual verification of the labeling rule against known cases.
Run this to confirm label_sample() matches hand-calculated expectations
before trusting the trained model.
"""
from data_utils import label_sample

# (body_temp, resp_rate, ambient_temp, expected_label, reason)
test_cases = [
    (39.8, 35, 32, 1, "All 3 stressed -> ON"),
    (39.8, 35, 25, 1, "Body+Resp stressed, ambient normal -> ON (2/3)"),
    (38.5, 35, 32, 1, "Resp+Ambient stressed, body normal -> ON (2/3)"),
    (39.8, 20, 25, 0, "Only body stressed -> OFF (1/3)"),
    (38.5, 20, 25, 0, "None stressed -> OFF (0/3)"),
    (39.5, 35, 32, 1, "Body temp exactly at cutoff (not >39.5, so NOT stressed) but resp+ambient = 2/3 -> ON"),
    (38.0, 29, 25, 0, "Resp at lower bound of stressed band (29) but alone -> OFF (1/3)"),
    # --- Edge cases added after code review ---
    (38.5, 65, 32, 1, "Resp=65 exceeds table max of 64 but is still stressed; resp+ambient = 2/3 -> ON"),
    (38.5, 30, 25, 0, "Resp=30 is >= 29 (stressed) but alone; body+ambient normal -> OFF (1/3)"),
    (38.5, 28, 25, 0, "Resp=28 below heat threshold (< 29); none stressed -> OFF (0/3)"),
    (39.6, 28, 31, 1, "Body+ambient stressed, resp=28 normal -> ON (2/3)"),
]

print(f"{'body_temp':>10} {'resp_rate':>10} {'ambient':>8} {'expected':>9} {'actual':>7} {'match':>6}  reason")
all_pass = True
for bt, rr, at, expected, reason in test_cases:
    actual = label_sample(bt, rr, at)
    match = "YES" if actual == expected else "NO"
    if actual != expected:
        all_pass = False
    print(f"{bt:>10} {rr:>10} {at:>8} {expected:>9} {actual:>7} {match:>6}  {reason}")

print("\nALL TESTS PASSED" if all_pass else "\nSOME TESTS FAILED - review thresholds")
