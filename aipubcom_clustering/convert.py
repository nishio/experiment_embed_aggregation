import csv

rd = csv.reader(open("aipubcom_comments.csv"))
with open("simple.csv", "w") as fo:
    wr = csv.writer(fo)
    for row in rd:
        wr.writerow([row[1], "".join(row[0].splitlines())])
