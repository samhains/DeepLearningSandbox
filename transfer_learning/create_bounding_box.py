from subprocess import call
from skimage.transform import resize
import scipy
import os

DATA_DIR = "./valid"
RESULTS_FNAME = "test.txt"

with open(RESULTS_FNAME, 'w'): pass
classes = os.listdir(DATA_DIR)
MAX_WIDTH = 400
print(classes)

# for c in classes[-5:]:
    # image_dir = "{}/{}/".format(DATA_DIR, c)
    # images = os.listdir(image_dir)
    # print('class', c)

    # for image in images:
        # fname = image_dir+image
        # print('saving', fname)
        # try:
            # img = scipy.misc.imread(fname)
            # width, height, _ = img.shape
            # if (width > MAX_WIDTH):
                # ratio = float(height)/width 
                # new_height = int(MAX_WIDTH*ratio)
                # img = resize(img, (MAX_WIDTH, new_height))*255
                # width, height, _ = img.shape
                # scipy.misc.imsave(fname, img)

        # except:
            # print('erro')

