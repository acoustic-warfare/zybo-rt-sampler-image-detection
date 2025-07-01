from lib.beamformer import *

import sys

if sys.argv[1] == "miso":
    miso()
elif sys.argv[1] == "mimo":
    mimo()
elif sys.argv[1] == "record":
    record_sensorfusion()
else:
    print("invalid argument")

# mimo()