import csv

input_csv_path = "toyota.csv"

export_mileage_path = "toyota_mileage.csv"
export_price_path = "toyota_price.csv"

# these are the lists we want
list_mileage = []
list_price = []

# file open
with open (input_csv_path, mode ='r', newline = '', encoding = 'utf-8') as csv_input:
    reader = csv.reader(csv_input)
    for row in reader:
        list_price.append( [row[2]] )

with open (export_price_path, mode ='w', newline ='', encoding ='utf-8') as csv_output:
    writer = csv.writer(csv_output)
    for data in list_price:
        writer.writerow(data)

with open (input_csv_path, mode ='r', newline = '', encoding = 'utf-8') as csv_input:
    reader = csv.reader(csv_input)
    for row in reader:
        list_mileage.append( [row[4]] )

with open (export_mileage_path, mode ='w', newline ='', encoding ='utf-8') as csv_output:
    writer = csv.writer(csv_output)
    for data in list_mileage:
        writer.writerow(data)