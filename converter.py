import csv
import os
BASEDIR='track1'
FILENAME='user_profile'
txt_file = os.path.join(BASEDIR,FILENAME+'.txt')
csv_file = os.path.join(BASEDIR,FILENAME+'.csv')

in_txt = csv.reader(open(txt_file, "r"), delimiter='\t')
out_csv = csv.writer(open(csv_file, 'w',newline=''))

out_csv.writerows(in_txt)
