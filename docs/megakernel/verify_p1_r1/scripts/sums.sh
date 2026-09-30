#!/bin/bash
# md5 of every source the device path compiles or imports (pi0_5 tt/, common/, kernels) + the kernel digest; argv1 = label
R=/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5/models/experimental/pi0_5
echo "# $1 $(date +%F_%T) git=$(git -C /home/deepgadget/experiments/tt-models/ports/tt-pi-0.5 rev-parse --short HEAD)"
find $R/tt $R/common -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.py' \) -not -path '*__pycache__*' | sort | xargs md5sum
find $R/tt $R/common -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.py' \) -not -path '*__pycache__*' | sort | xargs cat | md5sum | sed 's/^/ALL /'
