# Licensed under a 3-clause BSD style license - see LICENSE.rst
from setuptools import setup

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}


entry_points = {
    "console_scripts": ["aca_hi_bgd_update=aca_hi_bgd.update_bgd_events:main"]
}


setup(
    name="aca_hi_bgd",
    author="Jean Connelly, Tom Aldcroft",
    description="ACA hi background monitor",
    author_email="jconnelly@cfa.harvard.edu",
    use_scm_version=True,
    setup_requires=["setuptools_scm", "setuptools_scm_git_archive"],
    zip_safe=False,
    packages=["aca_hi_bgd"],
    package_data={
        "aca_hi_bgd": [
            "top_level_template.html",
            "per_obs_template.html",
            "task_schedule.cfg",
        ]
    },
    tests_require=["pytest"],
    cmdclass=cmdclass,
    entry_points=entry_points,
    include_package_data=True,
)
