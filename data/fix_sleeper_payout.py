import sys

# ── Fix 1: simulator.py -- sleeper accounts must get a UNIQUE payout
#    (not drawn from ring_pays) so they have no payout co-sharing with ring members
src = open("data/simulator.py", encoding="utf-8").read()

# In generate_promo_return_rings (the bottom wrapper function), sleeper payout logic:
# Currently: member_pay = (f"PAY_VARIED_{...}" if is_varied else r.choice(ring_pays))
# Sleepers need their own unique payout REGARDLESS of is_varied
old_pay = (
    '            member_pay = (f"PAY_VARIED_{ring_id_counter[0]:03d}_{m_idx:02d}"\n'
    '                          if is_varied else r.choice(ring_pays))'
)
new_pay = (
    '            if is_sleeper:\n'
    '                # [A2] Sleeper structural suppression: unique payout breaks payout co-sharing\n'
    '                member_pay = f"PAY_SLEEPER_{ring_id_counter[0]:03d}_{m_idx:02d}"\n'
    '            elif is_varied:\n'
    '                member_pay = f"PAY_VARIED_{ring_id_counter[0]:03d}_{m_idx:02d}"\n'
    '            else:\n'
    '                member_pay = r.choice(ring_pays)'
)

if old_pay not in src:
    print("ERROR: old sleeper pay block not found")
    # Show nearby context
    idx = src.find("PAY_VARIED_")
    print("Context:", repr(src[max(0,idx-200):idx+200]))
    sys.exit(1)

src = src.replace(old_pay, new_pay, 1)
with open("data/simulator.py", "w", encoding="utf-8", newline="") as f:
    f.write(src)
import ast; ast.parse(src)
print("Fix 1 applied: sleeper payout suppression. Syntax OK.")