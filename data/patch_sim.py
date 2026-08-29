import sys

src = open("data/simulator.py", encoding="utf-8").read()

OLD_INIT = (
    "    labels_dict = {acc: {} for acc in bi_accounts + bc_accounts +\n"
    "                   ac_promo_accs + ac_return_accs + ac_ref_accs}"
)
NEW_INIT = (
    "    # Pre-populate label_true for ALL accounts from their pool membership.\n"
    "    # Accounts dropped by partition_into_groups tail are never processed by\n"
    "    # any generator, so {} init caused KeyError in apply_label_noise.\n"
    "    labels_dict = {}\n"
    "    for acc in bi_accounts:\n"
    "        labels_dict[acc] = {\"label_true\": \"benign_independent\",\n"
    "                             \"partial_signal\": False, \"counterfactual_subset\": None}\n"
    "    for acc in bc_accounts:\n"
    "        labels_dict[acc] = {\"label_true\": \"benign_coordinated\",\n"
    "                             \"partial_signal\": False, \"counterfactual_subset\": None}\n"
    "    for acc in ac_promo_accs + ac_return_accs + ac_ref_accs:\n"
    "        labels_dict[acc] = {\"label_true\": \"abusive_coordinated\",\n"
    "                             \"partial_signal\": False, \"counterfactual_subset\": None}\n"
)

if OLD_INIT not in src:
    print("ERROR: OLD_INIT not found")
    sys.exit(1)
src = src.replace(OLD_INIT, NEW_INIT, 1)
print("Fix 1 applied")

OLD_NOISE = (
    "    ac_accounts = [acc for acc, d in labels_dict.items()\n"
    "                   if d[\"label_true\"] == \"abusive_coordinated\"]"
)
NEW_NOISE = (
    "    ac_accounts = [acc for acc, d in labels_dict.items()\n"
    "                   if d.get(\"label_true\") == \"abusive_coordinated\"]"
)
if OLD_NOISE not in src:
    print("ERROR: OLD_NOISE not found")
    sys.exit(1)
src = src.replace(OLD_NOISE, NEW_NOISE, 1)
print("Fix 2 applied")

with open("data/simulator.py", "w", encoding="utf-8", newline="") as f:
    f.write(src)

import ast
ast.parse(src)
print("Syntax OK")