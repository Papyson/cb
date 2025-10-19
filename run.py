# This script makes the project runnable directly from its root directory.
# It solves the "flat layout" import problem by adding the project's
# parent directory to the Python path.

import runpy
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(project_root)

sys.path.insert(0, parent_dir)

runpy.run_module('citybuilder_env.main', run_name='__main__')