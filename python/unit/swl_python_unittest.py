#!/usr/bin/env python

# REF [site] >> https://docs.python.org/3/library/unittest.html

import unittest
#from add_test import AddTestCase

def suite():
	suite = unittest.TestSuite()
	#suite.addTest(AddTestCase())
	return suite

#%%------------------------------------------------------------------

# Usage:
#	python -m unittest swl_python_unittest
#	python -m unittest swl_python_unittest.py
#
#	python -m unittest discover -s project_directory -p "*_test.py"

if '__main__' == __name__:
	unittest.main()
