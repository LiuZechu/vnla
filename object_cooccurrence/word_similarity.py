import gensim.downloader as api
import matplotlib.pyplot as plt
import torch

# Load word embeddings
model = api.load("glove-wiki-gigaword-50")

# res = model.most_similar('blue')
# res = model.similarity(w1="sink", w2="toilet")

########################################################

# Returns an array of accepted object names in VNLA
def get_array_of_object_names(is_room_included=False):
  filename = "accepted_objects.txt"
  f = open(filename, "r")
  res = []
  for line in f:
    object_name = line.strip()
    res.append(object_name)
  f.close()

  room_words = ['bathroom', 'bedroom', 'closet', 'dining room', 'entryway', 'family room', 'garage', 'hallway', 'library', 'laundry room', 'kitchen', 'living room', 'meeting room', 'lounge', 'office', 'porch', 'recreation', 'stairs', 'toilet', 'utility room', 'tv', 'gym', 'outdoors', 'balcony', 'other room', 'bar', 'classroom', 'dining booth', 'spa', 'junk', 'no label']
  if is_room_included:
    res += room_words

  return res

# Returns an |O|x|O| square matrix of word similarity
def construct_word_similarity_matrix(object_names):
  # initialise matrix
  num_objects = len(object_names)
  res = [[0 for _ in range(num_objects)] for _ in range(num_objects)]

  for i in range(num_objects):
    for j in range(num_objects):
      tokens_1 = object_names[i].split(" ")
      tokens_2 = object_names[j].split(" ")
      similarity_score = model.n_similarity(tokens_1, tokens_2)
      res[i][j] = similarity_score

  return res

def visualise(matrix, title, is_room_included=False):
  object_words = ['airconditioner', 'altar', 'armchair', 'art', 'bag', 'balconyrailing', 'ball', 'bar', 'barricade', 'basin', 'basket', 'bathtub', 'bed', 'bedsheet', 'bedpost', 'bench', 'bidet', 'bin', 'blanket', 'blinds', 'board', 'book', 'bookcase', 'books', 'bookshelf', 'bottle', 'bottleofsoap', 'bottles', 'bowl', 'bowloffruit', 'box', 'boxes', 'bucket', 'bulletinboard', 'bushes', 'bust', 'candelabra', 'candle', 'candles', 'candlestick', 'car', 'case', 'chair', 'chairbottom', 'chandelier', 'chest', 'chestofdrawers', 'chimney', 'churchseating', 'clock', 'closet', 'closetshelf', 'closetshelving', 'cloth', 'clothes', 'clothesdryer', 'clotheshangerrod', 'clotheshangers', 'clutter', 'coat', 'coathanger', 'coatrack', 'coffeemaker', 'coffeetable', 'commode', 'computer', 'computerdesk', 'container', 'control', 'couch', 'counter', 'cup', 'cupboard', 'curtain', 'curtainrod', 'curtainvalence', 'curtains', 'cushion', 'decor', 'decorativeplate', 'desk', 'deskchair', 'diningchair', 'diningtable', 'dinnerplacesetting', 'dishwasher', 'displaycase', 'doll', 'drawer', 'drawers', 'dress', 'dresser', 'drums', 'duct', 'easel', 'easychair', 'electricwirecasing', 'endtable', 'exercisebike', 'exerciseequipment', 'exercisemachine', 'exitsign', 'fan', 'faucet', 'fence', 'fencing', 'figure', 'firealarm', 'fireextinguisher', 'fireplace', 'flower', 'flowerpot', 'flowers', 'food', 'footrest', 'footstool', 'fridge', 'fruitbowl', 'furniture', 'glass', 'globe', 'grass', 'guitar', 'gymequipment', 'hamper', 'handbag', 'handle', 'hanger', 'hangers', 'hat', 'headboard', 'heater', 'highchair', 'hose', 'jar', 'keyboard', 'kitchenappliance', 'kitchencenterisland', 'kitchencounter', 'kitchenisland', 'kitchenshelf', 'kitchenutensils', 'knickknack', 'knob', 'ladder', 'lamp', 'lampshade', 'landing', 'laundrybasket', 'ledge', 'locker', 'loungechair', 'mask', 'massagebed', 'massagetable', 'microwave', 'mirror', 'monitor', 'nightstand', 'officechair', 'officetable', 'ornament', 'ottoman', 'oven', 'painter', 'painting', 'pan', 'panel', 'paper', 'papertowel', 'papertoweldispenser', 'pedestal', 'pew', 'pews', 'phone', 'photo', 'piano', 'picture', 'pillar', 'pillow', 'pillows', 'placemat', 'plant', 'plantpot', 'plants', 'plate', 'plateoffood', 'plushtoy', 'pool', 'pooltable', 'post', 'pot', 'pottedplant', 'powerbreakerbox', 'printer', 'projector', 'purse', 'rack', 'radiator', 'rail', 'railing', 'rangehood', 'refrigerator', 'ridge', 'robe', 'rope', 'roundtable', 'scale', 'screen', 'sculpture', 'seat', 'shampoo', 'sheet', 'shelf', 'shelfwithclutter', 'shelves', 'shelving', 'shoes', 'showcase', 'shower', 'showerbench', 'showercurtain', 'showercurtainrod', 'showerhandle', 'showerhead', 'showersoapshelf', 'shrubbery', 'sign', 'sink', 'smokealarm', 'smokedetector', 'soap', 'soapdish', 'soapdispenser', 'sofa', 'sofachair', 'sofaset', 'speaker', 'stand', 'statue', 'stool', 'storageshelving', 'stove', 'suitcase', 'switch', 'swivelchair', 'table', 'tablelamp', 'tap', 'teapot', 'telephone', 'thermostat', 'tissuebox', 'tissuepaper', 'toilet', 'toiletbrush', 'toiletpaper', 'toiletpaperdispenser', 'toiletpaperholder', 'toiletry', 'towel', 'towelbar', 'towels', 'toy', 'trash', 'trashbin', 'trashcan', 'tray', 'treadmill', 'tree', 'trees', 'trinket', 'tv', 'tvstand', 'umbrella', 'urn', 'vanity', 'vase', 'vent', 'wardrobe', 'wardroberod', 'washbasin', 'washingmachine', 'watercooler', 'weightmachine', 'weights', 'whiteboard', 'wood', 'woodenchair']
  room_words = ['bathroom', 'bedroom', 'closet', 'dining room', 'entryway', 'family room', 'garage', 'hallway', 'library', 'laundry room', 'kitchen', 'living room', 'meeting room', 'lounge', 'office', 'porch', 'recreation', 'stairs', 'toilet', 'utility room', 'tv', 'gym', 'outdoors', 'balcony', 'other room', 'bar', 'classroom', 'dining booth', 'spa', 'junk', 'no label']

  if is_room_included:
    object_words += room_words

  _, ax = plt.subplots()
  ax.matshow(matrix)
  ax.set_xticks([i for i in range(len(object_words))])
  ax.set_xticklabels(object_words, rotation=45, fontsize=5)
  ax.set_yticks([i for i in range(len(object_words))])
  ax.set_yticklabels(object_words, fontsize=5)
  ax.title.set_text(title)

  plt.show()

def main():
  is_room_included = True # Toggle this based on needs
  object_names = get_array_of_object_names(is_room_included=is_room_included)
  similarity_matrix = construct_word_similarity_matrix(object_names)
  visualise(similarity_matrix, "Word Similarity Matrix", is_room_included=is_room_included)

  # SAVE MATRIX
  # tensor = torch.FloatTensor(similarity_matrix)
  # torch.save(tensor, 'word_matrix.pt', _use_new_zipfile_serialization=False)

if __name__ == "__main__":
  main()
