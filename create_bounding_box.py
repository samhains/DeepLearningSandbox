from subprocess import call
from skimage.transform import resize
import scipy
import os

DATA_DIR = "./data"
RESULTS_FNAME = "test.txt"

with open(RESULTS_FNAME, 'w'): pass
classes = os.listdir(DATA_DIR)
MAX_WIDTH = 128

for c in classes:
    image_dir = "{}/{}/".format(DATA_DIR, c)
    images = os.listdir(image_dir)

    for image in images:
        fname = image_dir+image
        try:
            img = scipy.misc.imread(fname)
            width, height, _ = img.shape
            if (width > MAX_WIDTH):
                ratio = float(height)/width 
                new_height = int(MAX_WIDTH*ratio)
                img = resize(img, (MAX_WIDTH, new_height))*255
                width, height, _ = img.shape
                scipy.misc.imsave(fname, img)
            x1 = 0
            y1 = 0
            x2 = width
            y2 = height
            classname = c
                
            with open(RESULTS_FNAME, "a") as myfile:
                myfile.write("{},{},{},{},{},{}\n".format(fname,x1,y1,x2,y2,classname))
        except:
            print('erro')

