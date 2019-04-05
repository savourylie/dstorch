from setuptools import setup, find_packages

setup(name='dstorch',
      version='0.0.5',
      description='A humble package for algorithms and data, implemented in PyTorch.',
   	  package_dir = {
            'data': 'data'
            },
      packages=find_packages(),
      url='http://github.com/savourylie/dstorch',
      author='Calvin Ku',
      author_email='calvin.j.ku@googlemail.com',
      license='MIT',
      zip_safe=False)
