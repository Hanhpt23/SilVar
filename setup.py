# import sys
# sys.path.append("/home1/tpham2023/Hanh/SilVar/SilVar/silvar")

# from setuptools import setup, find_packages

# setup(
#     name='silvar',
#     version='0.1',
#     packages=find_packages(),
#     install_requires=[
#         # Add any dependencies here
#     ],
# )


import os
import sys
import runpy

def run_script(script_path, script_args):
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    sys.argv = [script_path] + script_args
    runpy.run_path(script_path, run_name="__main__")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python setup.py <script_path> [script_args...]")
        sys.exit(1)
    
    script_path = sys.argv[1]
    script_args = sys.argv[2:]
    print('script_path: ', script_path)
    run_script(script_path, script_args)