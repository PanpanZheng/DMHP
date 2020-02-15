import sys
sys.path.append("./")
from DMHP_SMC import SMC

for usr in ["AAB0398","AAC0610","AAC0668","AAC3270", "AAD2188", "ACM2278", "CMP2946", "PLJ1771", "CDE1846", "MBG3183"]:
    print("***", usr)
    SMC(usr, 0.1, "./raw_input")

exit(0)