import csv

'''
This script generates `accepted_object_mappings.txt`, which contains 289 object
labels and their corresponding category_mapping_index as stated in `category_mapping.tsv`.
'''

# Read from `category_mapping.tsv`
# row[0]: index; row[1]: raw_category; row[2]: category (not used); row[3]: count
categories = {}
is_first = True
with open("category_mapping.tsv") as fd:
  rd = csv.reader(fd, delimiter="\t", quotechar='"')
  for row in rd:
    if is_first:
      is_first = False
      continue
    if int(row[3]) >= 5: # only consider objects with at least 5 counts
      categories[row[1]] = row[0]

# Read from `accepted_objects.txt`
f = open("accepted_objects.txt", "r")
results = {}
for line in f:
  line = line.strip()
  if line in categories:
    results[line] = categories[line]
f.close()

# Some objects need to be added manually
results["barricade"] = "191"
results["dinner place setting"] = "356"
results["flower pot"] = "151"
results["shelf with clutter"] = "381"

# Generate mappings between each accepted_object and category_mapping_index
results = sorted(results.items(), key=lambda x: x[0]) # alphabetical order
results = list(map(lambda x: " ".join(list(x)) + "\n", results))

f = open('accepted_object_mappings.txt', "w")
f.writelines(results)
f.close()
