#!/usr/bin/env python
""" 
choose_person.py
=================
This script chooses a person to present in a seminar

Written by **** Theo Narh *****
"""

# ***** import python modules *****
import numpy as np
from datetime import date, timedelta, datetime
import warnings
warnings.filterwarnings("ignore")
import time
import argparse
from Tkinter import *
import tkMessageBox
#

def next_presenter(filename, size=1,  duration=7, start_date=None):
	""" 
	** filname:: load names from .txt file
	** size:: number of presenters
	** start_date:: start date for each presenter
	** duration:: regular period of presentation
	** nshuffle:: number of times to shuffle

	returns presenter's name & date of presentation
	"""
	data = open("%s" %filename)
	dat = data.read()
	lst = dat.splitlines()
	c1 = np.array(lst)
	col_names = c1[:-1]

	pnames = np.random.choice(a=col_names, size=size, replace=False, p=None)

	# print pnames, pnames[:size]

	# arr = [i for i in range(len(col_names))]
	# for iter in xrange(nshuffle):
	# 	np.random.shuffle(arr)


	if start_date is not None:
		yr = start_date.year
		mon = start_date.month
		day = start_date.day
		d1 = date(yr, mon, day)               #  start date

		ndate = [d1 + timedelta(days=i) for i in xrange(0, len(col_names)*duration, duration)]

		return pnames[:size][0], ndate[:size][0]

	else: return pnames[:size]		


def valid_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)


def run_test_model():
	#
	#	start-time
	start = time.time()
	startime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
	print "Start at %s" % startime
	#

	parser = argparse.ArgumentParser(description='Print out the next presenter.') 
	# parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum, default=max, help='sum the integers (default: find the max)')
	parser.add_argument('-f', '--file', dest='filenames', required=True, type=str, help='Provide a .txt file containing list of names')
	parser.add_argument('-n ', '--np', dest='nsize', required=False, default=1, type=int, help='Provide the number of presenters')
	parser.add_argument('-d ', '--dur', dest='period', required=False, default=7, type=int, help='Provide the period of presentation')
	# parser.add_argument('-t ', '--shuf', dest='nshuf', required=False, default=1e6, type=int, help='Provide the shuffling number')
	# parser.add_argument('--dt', dest='strtdate', required=False, default=1, type=int, help='Provide the number of presenters (default: 1)')
	parser.add_argument("-s", "--startdate", dest='strtdate', help="The Start Date - format YYYY-MM-DD", required=False, default=None, type=valid_date)

	args = parser.parse_args()
	#

	# ----- SETUP GUIS -----------

	r = Tk()
	w = Label(r, text="My Program")
	w.pack()
	r.option_add('*font', 'Helvetica -20')

	# Welcome the user

	# tkMessageBox.showinfo('welcome', 'add you wel message here!!')


	if args.strtdate is None:
		per_name = next_presenter(filename=args.filenames, size=args.nsize, duration=args.period, start_date=args.strtdate)

		print "\n>> Next Presenter is:----->>", per_name
		tkMessageBox.showinfo('Next Presenter', '%s'%per_name[:])
		r.option_clear()

	else: 
		per_name, pre_date = next_presenter(filename=args.filenames, size=args.nsize, duration=args.period, start_date=args.strtdate) 

		print per_name, '\t', pre_date
	# print per_name, '\t', pre_date

	# # parser.add_argument('--dt', dest='strtdate', action='store_const',
 #                    const=sum, default=max,
 #                    help='sum the integers (default: find the max)')




	stoptime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
	print "\n Stop at %s" % stoptime 
	end = time.time()
	elasped_time = (end - start)/3600.0
	print "Total run time: %7.2f hours" % elasped_time

if __name__=="__main__":
	#
	run_test_model()






