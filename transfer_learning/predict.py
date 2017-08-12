import sys
import argparse
import os
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import json
import ast

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input


target_size = (229, 229) #fixed size for InceptionV3 architecture


def predict(model, img, target_size):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  return preds[0]

def return_preds(image, preds):
  """Displays image and the top-n predicted probabilities in a bar graph
  Args:
    image: PIL image
    preds: list of predicted labels and their probabilities
  """
  labels_obj = {}
  with open('labels.txt', 'r') as labels_file:
    data = labels_file.read()
    labels_data = ast.literal_eval(data)
    labels = tuple(sorted(labels_data, key=labels_data.get))
  index = range(len(labels))
  # preds = np.around(preds, decimals=2)
  for i, pred in enumerate(preds):
    if(pred>0.03):
      label = labels[i]
      labels_obj[label] = "{:.2f}".format(pred)


  return labels_obj


def plot_preds(image, preds):
  """Displays image and the top-n predicted probabilities in a bar graph
  Args:
    image: PIL image
    preds: list of predicted labels and their probabilities
  """
  plt.imshow(image)
  plt.axis('off')

  plt.figure()
  with open('labels.txt', 'r') as labels_file:
    data = labels_file.read()
    labels_data = ast.literal_eval(data)
    labels = tuple(sorted(labels_data, key=labels_data.get))
  index = range(len(labels))
  new_labels = [] 
  new_preds = []
  for i, pred in enumerate(preds):
    if pred > 0.05:
      new_labels.append(labels[i])
      new_preds.append(pred)
  new_index = range(len(new_labels))
  plt.barh(new_index, new_preds, alpha=0.5)
  plt.yticks(new_index, new_labels)
  plt.xlabel('Probability')
  plt.xlim(0,1.01)
  plt.tight_layout()
  plt.show()


if __name__=="__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--image", help="path to image")
  a.add_argument("--image_folder", help="path to image folder")
  a.add_argument("--image_url", help="url to image")
  a.add_argument("--model")
  args = a.parse_args()

  # if args.image is None and args.image_url is None and args.img_folder is None:
    # a.print_help()
    # sys.exit(1)

  model = load_model(args.model)
  if args.image is not None:
    img = Image.open(args.image)
    preds = predict(model, img, target_size)
    plot_preds(img, preds)

  if args.image_folder is not None:
    results = {}
    img_urls = os.listdir(args.image_folder)
    print('img_urls', img_urls)
    for url in img_urls:
      fname = args.image_folder+"/"+url
      img = Image.open(fname)
      preds = predict(model, img, target_size)
      preds = plot_preds(img, preds)
      results[url] = preds
    with open('results.oil.json', 'w') as results_file:
      json.dump(results, results_file)


  if args.image_url is not None:
    response = requests.get(args.image_url)
    img = Image.open(BytesIO(response.content))
    preds = predict(model, img, target_size)
    plot_preds(img, preds)

