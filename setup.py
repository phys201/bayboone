from setuptools import setup

setup(name='bayboone',
      version='1.0',
      description='Baysian analysis for MicroBooNE LEE',
      url='http://github.com/phys201/bayboone',
      author='jybook, tcontrer',
      author_email='jbook@g.harvard.edu, taylorcontreras@g.harvard.edu',
      license='GPLv3',
      packages=['bayboone'],
      install_requires=['numpy, matplotlib, pyplot, pandas, seaborn, emcee, pymc3, uproot'])
