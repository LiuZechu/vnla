import os
import math
import matplotlib.pyplot as plt
import torch

from room_object_cooccurrence import generate_room_object_cooccurrence_counts

DIR_PATH = "./data"

def get_accepted_object_indices():
  res = {} # maps category_mapping_index to 0-based incrementing array index
  f = open("accepted_object_mappings.txt", "r")
  counter = 0
  for line in f:
    tokens = line.split(" ")
    index = int(tokens[-1])
    res[index] = counter
    counter += 1
  f.close()
  return res

def generate_cooccurrence_matrix(accepted_object_indices):
  num_object_categories = len(accepted_object_indices)
  cooccurrence_matrix = [[0 for _ in range(num_object_categories)] \
    for _ in range(num_object_categories)]
  global DIR_PATH

  for filename in os.listdir(DIR_PATH):
    f = open(os.path.join(DIR_PATH, filename), "r")
    
    # initialise info for each house
    category_mappings = []
    region_objects = []

    # go through raw input from .house file
    print(f"Going through raw input from {filename} now...")
    for line in f:
      tokens = line.strip().split()
      if tokens[0] == "R":
        region_objects.append([])
      elif tokens[0] == "C":
        category_mapping_index = int(tokens[2]) # category_mapping_index
        category_mappings.append(category_mapping_index)
      elif tokens[0] == "O":
        category_index = int(tokens[3])
        region_index = int(tokens[2])
        if category_mappings[category_index] in accepted_object_indices: # only consider valid objects
          region_objects[region_index].append(category_mappings[category_index])

    # populate cooccurrence matrix
    for object_list in region_objects:
      object_set = set(object_list)
      for oi in object_set:
        for oj in object_set:
          obj_1 = accepted_object_indices[oi]
          obj_2 = accepted_object_indices[oj]
          cooccurrence_matrix[obj_1][obj_2] += 1
    
    f.close()

  return cooccurrence_matrix

# Added this function to generate a square matrix that includes room-object cooccurrence as well.
def generate_cooccurrence_matrix_with_room(object_cooccurrence_matrix):
  room_object_cooccurrence_matrix = generate_room_object_cooccurrence_counts()
  
  res = []
  # Construct the first |O| rows of the res matrix 
  for index, row in enumerate(object_cooccurrence_matrix):
    new_row = row.copy()
    for room_row in room_object_cooccurrence_matrix:
      new_row.append(room_row[index])
    res.append(new_row)

  # Construct the last |R| rows of the res matrix
  for index, row in enumerate(room_object_cooccurrence_matrix):
    new_row = row.copy()
    suffix = [0] * len(room_object_cooccurrence_matrix)
    suffix[index] = sum(row)
    new_row += suffix
    res.append(new_row)

  return res


def normalise_pmi(cooccurrence_matrix):
  # Normalise by Pairwise Mutual Information (PMI)
  total_num_pairs = 0
  total_num_items = 0
  for i in range(len(cooccurrence_matrix)):
    for j in range(len(cooccurrence_matrix)):
      if i != j:
        total_num_pairs += cooccurrence_matrix[i][j]
      else:
        total_num_items += cooccurrence_matrix[i][j]

  num_object_categories = len(cooccurrence_matrix)
  normalised_matrix = [[0 for _ in range(num_object_categories)] \
    for _ in range(num_object_categories)]
  for i in range(len(normalised_matrix)):
    for j in range(len(normalised_matrix)):
      if cooccurrence_matrix[i][j] != 0:
        value = math.log(cooccurrence_matrix[i][j] / total_num_pairs) \
          - math.log(cooccurrence_matrix[i][i] / total_num_items) \
          - math.log(cooccurrence_matrix[j][j] / total_num_items)
        normalised_matrix[i][j] = 1 / (1 + math.exp(-1 * value))
      if i == j:
        normalised_matrix[i][j] = 1
  
  return normalised_matrix

def normalise_rowwise(cooccurrence_matrix):
  normalised_matrix = [[0 for _ in range(len(cooccurrence_matrix))] \
    for _ in range(len(cooccurrence_matrix))]

  for i in range(len(cooccurrence_matrix)):
    summation = sum(cooccurrence_matrix[i]) - cooccurrence_matrix[i][i]
    for j in range(len(cooccurrence_matrix)):
      if j == i:
        normalised_matrix[i][j] = 1
      else:
        normalised_matrix[i][j] = 0 if summation == 0 else cooccurrence_matrix[i][j] / summation
  
  return normalised_matrix

def visualise(matrix_1, matrix_2, title_1, title_2, is_room_included=False):
  object_words = ['airconditioner', 'altar', 'armchair', 'art', 'bag', 'balconyrailing', 'ball', 'bar', 'barricade', 'basin', 'basket', 'bathtub', 'bed', 'bedsheet', 'bedpost', 'bench', 'bidet', 'bin', 'blanket', 'blinds', 'board', 'book', 'bookcase', 'books', 'bookshelf', 'bottle', 'bottleofsoap', 'bottles', 'bowl', 'bowloffruit', 'box', 'boxes', 'bucket', 'bulletinboard', 'bushes', 'bust', 'candelabra', 'candle', 'candles', 'candlestick', 'car', 'case', 'chair', 'chairbottom', 'chandelier', 'chest', 'chestofdrawers', 'chimney', 'churchseating', 'clock', 'closet', 'closetshelf', 'closetshelving', 'cloth', 'clothes', 'clothesdryer', 'clotheshangerrod', 'clotheshangers', 'clutter', 'coat', 'coathanger', 'coatrack', 'coffeemaker', 'coffeetable', 'commode', 'computer', 'computerdesk', 'container', 'control', 'couch', 'counter', 'cup', 'cupboard', 'curtain', 'curtainrod', 'curtainvalence', 'curtains', 'cushion', 'decor', 'decorativeplate', 'desk', 'deskchair', 'diningchair', 'diningtable', 'dinnerplacesetting', 'dishwasher', 'displaycase', 'doll', 'drawer', 'drawers', 'dress', 'dresser', 'drums', 'duct', 'easel', 'easychair', 'electricwirecasing', 'endtable', 'exercisebike', 'exerciseequipment', 'exercisemachine', 'exitsign', 'fan', 'faucet', 'fence', 'fencing', 'figure', 'firealarm', 'fireextinguisher', 'fireplace', 'flower', 'flowerpot', 'flowers', 'food', 'footrest', 'footstool', 'fridge', 'fruitbowl', 'furniture', 'glass', 'globe', 'grass', 'guitar', 'gymequipment', 'hamper', 'handbag', 'handle', 'hanger', 'hangers', 'hat', 'headboard', 'heater', 'highchair', 'hose', 'jar', 'keyboard', 'kitchenappliance', 'kitchencenterisland', 'kitchencounter', 'kitchenisland', 'kitchenshelf', 'kitchenutensils', 'knickknack', 'knob', 'ladder', 'lamp', 'lampshade', 'landing', 'laundrybasket', 'ledge', 'locker', 'loungechair', 'mask', 'massagebed', 'massagetable', 'microwave', 'mirror', 'monitor', 'nightstand', 'officechair', 'officetable', 'ornament', 'ottoman', 'oven', 'painter', 'painting', 'pan', 'panel', 'paper', 'papertowel', 'papertoweldispenser', 'pedestal', 'pew', 'pews', 'phone', 'photo', 'piano', 'picture', 'pillar', 'pillow', 'pillows', 'placemat', 'plant', 'plantpot', 'plants', 'plate', 'plateoffood', 'plushtoy', 'pool', 'pooltable', 'post', 'pot', 'pottedplant', 'powerbreakerbox', 'printer', 'projector', 'purse', 'rack', 'radiator', 'rail', 'railing', 'rangehood', 'refrigerator', 'ridge', 'robe', 'rope', 'roundtable', 'scale', 'screen', 'sculpture', 'seat', 'shampoo', 'sheet', 'shelf', 'shelfwithclutter', 'shelves', 'shelving', 'shoes', 'showcase', 'shower', 'showerbench', 'showercurtain', 'showercurtainrod', 'showerhandle', 'showerhead', 'showersoapshelf', 'shrubbery', 'sign', 'sink', 'smokealarm', 'smokedetector', 'soap', 'soapdish', 'soapdispenser', 'sofa', 'sofachair', 'sofaset', 'speaker', 'stand', 'statue', 'stool', 'storageshelving', 'stove', 'suitcase', 'switch', 'swivelchair', 'table', 'tablelamp', 'tap', 'teapot', 'telephone', 'thermostat', 'tissuebox', 'tissuepaper', 'toilet', 'toiletbrush', 'toiletpaper', 'toiletpaperdispenser', 'toiletpaperholder', 'toiletry', 'towel', 'towelbar', 'towels', 'toy', 'trash', 'trashbin', 'trashcan', 'tray', 'treadmill', 'tree', 'trees', 'trinket', 'tv', 'tvstand', 'umbrella', 'urn', 'vanity', 'vase', 'vent', 'wardrobe', 'wardroberod', 'washbasin', 'washingmachine', 'watercooler', 'weightmachine', 'weights', 'whiteboard', 'wood', 'woodenchair']
  room_words = ['bathroom', 'bedroom', 'closet', 'dining room', 'entryway', 'familyroom', 'garage', 'hallway', 'library', 'laundryroom', 'kitchen', 'living room', 'meetingroom', 'lounge', 'office', 'porch', 'recreation', 'stairs', 'toilet', 'utilityroom', 'tv', 'gym', 'outdoors', 'balcony', 'other room', 'bar', 'classroom', 'dining booth', 'spa', 'junk', 'no label']

  if is_room_included:
    object_words += room_words

  f, axes = plt.subplots(1, 2)
  f.set_size_inches(10, 10)
  axes[0].matshow(matrix_1)
  axes[0].set_xticks([i for i in range(len(object_words))])
  axes[0].set_xticklabels(object_words, rotation=45, fontsize=5)
  axes[0].set_yticks([i for i in range(len(object_words))])
  axes[0].set_yticklabels(object_words, fontsize=5)
  axes[0].title.set_text(title_1)

  axes[1].matshow(matrix_2)
  axes[1].set_xticks([i for i in range(len(object_words))])
  axes[1].set_xticklabels(object_words, rotation=45, fontsize=5)
  axes[1].set_yticks([i for i in range(len(object_words))])
  axes[1].set_yticklabels(object_words, fontsize=5)
  axes[1].title.set_text(title_2)

  plt.show()

def main():
  accepted_object_indices = get_accepted_object_indices()
  object_cooccurrence_matrix = generate_cooccurrence_matrix(accepted_object_indices)
  room_obj_cooc_matrix = generate_cooccurrence_matrix_with_room(object_cooccurrence_matrix)
  normalised_matrix_1 = normalise_pmi(room_obj_cooc_matrix)
  normalised_matrix_2 = normalise_rowwise(room_obj_cooc_matrix)

  # save matrix
  tensor = torch.FloatTensor(normalised_matrix_1)
  torch.save(tensor, 'matrix.pt', _use_new_zipfile_serialization=False)

  # print("matrix dimension is: ")
  # print(len(normalised_matrix_1))
  # print("====")

  # visualise(normalised_matrix_1, normalised_matrix_2, "PMI Normalised", "Row-wise Normalised", is_room_included=True)

if __name__ == "__main__":
  main()
