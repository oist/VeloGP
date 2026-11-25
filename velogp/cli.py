# Entry points of the `velogp` and `velomix` commands installed by pip.
#
# The command-line tools are implemented as scripts in velogp/__main__.py and
# velogp/velomix/__main__.py, which run on import. These functions execute
# them exactly as `python -m velogp` and `python -m velogp.velomix` do.
import runpy


def velogp():
    runpy.run_module("velogp", run_name="__main__", alter_sys=True)


def velomix():
    runpy.run_module("velogp.velomix", run_name="__main__", alter_sys=True)
