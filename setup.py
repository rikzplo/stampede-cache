"""Build configuration for stampede C extensions.

The C extension is OPTIONAL — if compilation fails (e.g. no C compiler),
stampede falls back to pure Python automatically. Users never need to
do anything special; `pip install stampede` just works.
"""

import os
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """Build C extensions as optional — don't fail the install if they can't compile."""

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:
            print(
                f"\n  WARNING: Could not compile C extension '{ext.name}': {e}\n"
                f"  stampede will use pure-Python fallback (slower hashing).\n"
                f"  This is fine — all functionality works, just ~10-50x slower for hashing.\n"
            )

    def run(self):
        try:
            super().run()
        except Exception as e:
            print(
                f"\n  WARNING: Could not run C extension build: {e}\n"
                f"  stampede will use pure-Python fallback.\n"
            )


setup(
    ext_modules=[
        Extension(
            "stampede._native_hash",
            sources=["src/stampede/_native_hash.c"],
            # Reasonable optimization for all platforms
            extra_compile_args=["-O2"] if os.name != "nt" else ["/O2"],
        ),
    ],
    cmdclass={"build_ext": OptionalBuildExt},
)
