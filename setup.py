import os
import sys
# Licensed under a 3-clause BSD style license - see LICENSE.rst
from aca_hi_bgd import __version__

from setuptools import setup

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}


entry_points = {'console_scripts': [
    'aca_hi_bgd_update=aca_hi_bgd.update_bgd_events:main']}


if "--user" not in sys.argv:
    share_path = os.path.join(sys.prefix, "share", "aca_hi_bgd")
    data_files = [(share_path, ['task_schedule.cfg'])]
else:
    data_files = None


setup(name='aca_hi_bgd',
      author='Jean Connelly, Tom Aldcroft',
      description='ACA hi background monitor',
      author_email='jconnelly@cfa.harvard.edu',
      version=__version__,
      zip_safe=False,
      packages=['aca_hi_bgd'],
      package_data={'aca_hi_bgd': ['top_level_template.html', 'per_obs_template.html']},
      tests_require=['pytest'],
      cmdclass=cmdclass,
      data_files=data_files,
      entry_points=entry_points,
      include_package_data=True,
      )
