import csv

a = [[1,2,3,4],[5,6,7,8]]

with open("new_file.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(a)