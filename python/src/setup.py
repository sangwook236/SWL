# REF [site] >>
#	https://packaging.python.org/tutorials/packaging-projects/
#	https://packaging.python.org/guides/using-testpypi/
#
# Usage:
#	python setup.py sdist bdist_wheel
#		This command generate two files in the dist directory.
#	twine upload --repository-url https://test.pypi.org/legacy/ dist/*
#		Run Twine to upload all of the archives under dist.
#
#	pip install swl_python-???.whl
#	pip install --index-url https://test.pypi.org/simple/ swl_python
#		Install the package from TestPyPI.
#	pip uninstall swl-python
#
#	python -c "import swl"

import setuptools

with open('README.md', 'r') as fd:
	long_description = fd.read()

setuptools.setup(
	name='swl_python',
	version='1.1.0',
	author='Sang-Wook Lee',
	author_email='sangwook236@gmail.com',
	description="Sang-Wook's Library for Python (SWL-Python)",
	long_description=long_description,
	long_description_content_type='text/markdown',
	url='https://github.com/sangwook236/SWL',
	packages=setuptools.find_packages(),
	classifiers=(
		'Programming Language :: Python :: 3',
		'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
		'Operating System :: OS Independent',
	),
)
