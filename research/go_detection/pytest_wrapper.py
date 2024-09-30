import os
import sys

import debugpy
import pytest

if __name__ == "__main__":
    if os.environ.get("ENABLE_DEBUGPY"):
        print("")
        print("\033[31mWaiting for debugger to connect\033[0m")
        debugpy.listen(5678)
        debugpy.wait_for_client()

    argv = sys.argv[1:]
    pytest.main(argv)
