import csv
import os
import re
BASEDIR = 'track1'
FILENAME = 'user_profile'
txt_file = os.path.join(BASEDIR, FILENAME+'.txt')
csv_file = os.path.join(BASEDIR, FILENAME+''+'.csv')
in_txt = csv.reader(open(txt_file, "r"), delimiter='\t')
out_csv = csv.writer(open(csv_file, 'w', newline=''))
for row in in_txt:
    if not re.match('^[0-9]{4}$',row[1]):
        row[1]=0
    out_csv.writerow(row[:-1])
    