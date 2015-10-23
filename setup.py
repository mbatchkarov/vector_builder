# -*- coding: utf-8 -*-

from setuptools import setup, Command


# https://pytest.org/latest/goodpractises.html#integrating-with-setuptools-python-setup-py-test
class PyTest(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
        import sys
        errno = subprocess.call([sys.executable, 'runtests.py'])
        raise SystemExit(errno)


setup(name='vector_builder',
      version='1.0',
      packages=['builder'],
      author=['Miroslav Batchkarov'],
      author_email=['M.Batchkarov@sussex.ac.uk'],
      install_requires=['pandas', 'numpy', 'scipy', 'scikit-learn',
                        'scikit-learn', 'joblib', 'configobj',
                        'gensim'],
      tests_require=['pytest>=2.4.2'],
      cmdclass={'test': PyTest}, )
