# -*- coding: utf-8 -*-
"""
utils
"""
import csv

def outputCSV(filename,data):
    csvfile=open(filename,'w',newline='')
    writer=csv.writer(csvfile)
    writer.writerows(data)
    csvfile.close()
