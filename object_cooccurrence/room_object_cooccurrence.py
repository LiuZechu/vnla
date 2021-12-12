import os
import matplotlib.pyplot as plt

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

# Only generates room-object cooccurrence matrix, which is not symmetric
# and has the dimension [|R|, |O|].
def generate_cooccurrence_matrix(accepted_object_indices, room_indices):
  num_object_categories = len(accepted_object_indices)
  num_room_categories = len(room_indices) # == 31
  cooccurrence_matrix = [[0 for _ in range(num_object_categories)] \
    for _ in range(num_room_categories)]
  global DIR_PATH

  for filename in os.listdir(DIR_PATH):
    f = open(os.path.join(DIR_PATH, filename), "r")
    
    # initialise info for each house
    category_mappings = []
    region_mappings = [] # maps array index to region label string
    region_objects = []

    # go through raw input from .house file
    for line in f:
      tokens = line.strip().split()
      if tokens[0] == "R":
        region_label = tokens[5]
        region_mappings.append(region_label)
        region_objects.append([])
      elif tokens[0] == "C":
        category_mapping_index = int(tokens[2]) # category_mapping_index
        category_mappings.append(category_mapping_index)
      elif tokens[0] == "O":
        category_index = int(tokens[3])
        region_index = int(tokens[2])
        if category_mappings[category_index] in accepted_object_indices: # only consider valid objects
          region_objects[region_index].append(category_mappings[category_index])

    # populate room-object cooccurrence matrix
    for room_index, object_list in enumerate(region_objects):
      room = room_indices[region_mappings[room_index]]
      object_set = set(object_list)
      for object_index in object_set:
        object = accepted_object_indices[object_index]
        cooccurrence_matrix[room][object] += 1
    
    f.close()

  return cooccurrence_matrix

def visualise(matrix, title):
  object_words = ['airconditioner', 'altar', 'armchair', 'art', 'bag', 'balconyrailing', 'ball', 'bar', 'barricade', 'basin', 'basket', 'bathtub', 'bed', 'bedsheet', 'bedpost', 'bench', 'bidet', 'bin', 'blanket', 'blinds', 'board', 'book', 'bookcase', 'books', 'bookshelf', 'bottle', 'bottleofsoap', 'bottles', 'bowl', 'bowloffruit', 'box', 'boxes', 'bucket', 'bulletinboard', 'bushes', 'bust', 'candelabra', 'candle', 'candles', 'candlestick', 'car', 'case', 'chair', 'chairbottom', 'chandelier', 'chest', 'chestofdrawers', 'chimney', 'churchseating', 'clock', 'closet', 'closetshelf', 'closetshelving', 'cloth', 'clothes', 'clothesdryer', 'clotheshangerrod', 'clotheshangers', 'clutter', 'coat', 'coathanger', 'coatrack', 'coffeemaker', 'coffeetable', 'commode', 'computer', 'computerdesk', 'container', 'control', 'couch', 'counter', 'cup', 'cupboard', 'curtain', 'curtainrod', 'curtainvalence', 'curtains', 'cushion', 'decor', 'decorativeplate', 'desk', 'deskchair', 'diningchair', 'diningtable', 'dinnerplacesetting', 'dishwasher', 'displaycase', 'doll', 'drawer', 'drawers', 'dress', 'dresser', 'drums', 'duct', 'easel', 'easychair', 'electricwirecasing', 'endtable', 'exercisebike', 'exerciseequipment', 'exercisemachine', 'exitsign', 'fan', 'faucet', 'fence', 'fencing', 'figure', 'firealarm', 'fireextinguisher', 'fireplace', 'flower', 'flowerpot', 'flowers', 'food', 'footrest', 'footstool', 'fridge', 'fruitbowl', 'furniture', 'glass', 'globe', 'grass', 'guitar', 'gymequipment', 'hamper', 'handbag', 'handle', 'hanger', 'hangers', 'hat', 'headboard', 'heater', 'highchair', 'hose', 'jar', 'keyboard', 'kitchenappliance', 'kitchencenterisland', 'kitchencounter', 'kitchenisland', 'kitchenshelf', 'kitchenutensils', 'knickknack', 'knob', 'ladder', 'lamp', 'lampshade', 'landing', 'laundrybasket', 'ledge', 'locker', 'loungechair', 'mask', 'massagebed', 'massagetable', 'microwave', 'mirror', 'monitor', 'nightstand', 'officechair', 'officetable', 'ornament', 'ottoman', 'oven', 'painter', 'painting', 'pan', 'panel', 'paper', 'papertowel', 'papertoweldispenser', 'pedestal', 'pew', 'pews', 'phone', 'photo', 'piano', 'picture', 'pillar', 'pillow', 'pillows', 'placemat', 'plant', 'plantpot', 'plants', 'plate', 'plateoffood', 'plushtoy', 'pool', 'pooltable', 'post', 'pot', 'pottedplant', 'powerbreakerbox', 'printer', 'projector', 'purse', 'rack', 'radiator', 'rail', 'railing', 'rangehood', 'refrigerator', 'ridge', 'robe', 'rope', 'roundtable', 'scale', 'screen', 'sculpture', 'seat', 'shampoo', 'sheet', 'shelf', 'shelfwithclutter', 'shelves', 'shelving', 'shoes', 'showcase', 'shower', 'showerbench', 'showercurtain', 'showercurtainrod', 'showerhandle', 'showerhead', 'showersoapshelf', 'shrubbery', 'sign', 'sink', 'smokealarm', 'smokedetector', 'soap', 'soapdish', 'soapdispenser', 'sofa', 'sofachair', 'sofaset', 'speaker', 'stand', 'statue', 'stool', 'storageshelving', 'stove', 'suitcase', 'switch', 'swivelchair', 'table', 'tablelamp', 'tap', 'teapot', 'telephone', 'thermostat', 'tissuebox', 'tissuepaper', 'toilet', 'toiletbrush', 'toiletpaper', 'toiletpaperdispenser', 'toiletpaperholder', 'toiletry', 'towel', 'towelbar', 'towels', 'toy', 'trash', 'trashbin', 'trashcan', 'tray', 'treadmill', 'tree', 'trees', 'trinket', 'tv', 'tvstand', 'umbrella', 'urn', 'vanity', 'vase', 'vent', 'wardrobe', 'wardroberod', 'washbasin', 'washingmachine', 'watercooler', 'weightmachine', 'weights', 'whiteboard', 'wood', 'woodenchair']
  room_words = ['bathroom', 'bedroom', 'closet', 'dining room', 'entryway', 'familyroom', 'garage', 'hallway', 'library', 'laundryroom', 'kitchen', 'living room', 'meetingroom', 'lounge', 'office', 'porch', 'recreation', 'stairs', 'toilet', 'utilityroom', 'tv', 'gym', 'outdoors', 'balcony', 'other room', 'bar', 'classroom', 'dining booth', 'spa', 'junk', 'no label']

  fig = plt.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(matrix)
  fig.colorbar(cax)

  ax.set_xticks([i for i in range(len(object_words))])
  ax.set_xticklabels(object_words, rotation=45, fontsize=5)
  ax.set_yticks([i for i in range(len(room_words))])
  ax.set_yticklabels(room_words, fontsize=5)
  ax.title.set_text(title)

  plt.show()

# External file should call this function to get the cooc counts matrix of dimension [|R|, |O|]
def generate_room_object_cooccurrence_counts():
  room_indices = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7,
    'i': 8,
    'j': 9,
    'k': 10,
    'l': 11,
    'm': 12,
    'n': 13,
    'o': 14,
    'p': 15,
    'r': 16,
    's': 17,
    't': 18,
    'u': 19,
    'v': 20,
    'w': 21,
    'x': 22,
    'y': 23,
    'z': 24,
    'B': 25,
    'C': 26,
    'D': 27,
    'S': 28,
    'Z': 29,
    '-': 30
  }

  accepted_object_indices = get_accepted_object_indices()
  cooccurrence_matrix = generate_cooccurrence_matrix(accepted_object_indices, room_indices)
  return cooccurrence_matrix

def main():
  room_indices = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7,
    'i': 8,
    'j': 9,
    'k': 10,
    'l': 11,
    'm': 12,
    'n': 13,
    'o': 14,
    'p': 15,
    'r': 16,
    's': 17,
    't': 18,
    'u': 19,
    'v': 20,
    'w': 21,
    'x': 22,
    'y': 23,
    'z': 24,
    'B': 25,
    'C': 26,
    'D': 27,
    'S': 28,
    'Z': 29,
    '-': 30
  }

  accepted_object_indices = get_accepted_object_indices()
  cooccurrence_matrix = generate_cooccurrence_matrix(accepted_object_indices, room_indices)
  visualise(cooccurrence_matrix, "Room-Object Co-occurrence Counts")

if __name__ == "__main__":
  main()
