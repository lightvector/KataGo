"""Remove the work directory aiecc leaves beside a compiled artifact.

aiecc writes a <name>.prj tree next to whatever --xclbin points at: MLIR at
several lowering stages, one ELF and one linker script per core, .ll files, a
.pdi. For a 32-core design that is a few thousand files and tens of megabytes,
and none of it is an input to anything afterwards - only the .xclbin and the
.insts.bin are. Left behind under cpp/external/ryzenai_artifacts it also gets
copied next to katago.exe by the build, which is how a deployed tree once ended
up carrying several hundred stray files.

Keep it when a compile fails: the logs in there are the only record of why.
"""

import shutil
from pathlib import Path


def clean(xclbin_path, keep=False):
    """Delete the .prj beside xclbin_path. Returns what it removed, or None."""
    prj = Path(xclbin_path).with_suffix(".prj")
    if keep or not prj.is_dir():
        return None
    shutil.rmtree(prj, ignore_errors=True)
    return prj if not prj.exists() else None
