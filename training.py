import csv

lines = []
with open(../datadriving_log.csv) as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
