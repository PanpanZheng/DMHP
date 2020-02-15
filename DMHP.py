import sys
sys.path.append("./")
from DMHP_SMC import SMC

for usr in ["AAB0398","AAC0610","AAC0668","AAC3270", "AAD2188"]:
    print("***", usr)
    SMC(usr, 0.1, "./raw_input")

exit(0)