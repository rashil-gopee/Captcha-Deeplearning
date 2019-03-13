import os
import os.path
import glob
import re
import sys
import time
import errno
import argparse
from datetime import datetime
import threading
import traceback
import numpy
import cv2
import requests
import bottle
import PIL
from PIL import Image

CAPTCHA_IMAGE_FOLDER = "data_set"
OUTPUT_FOLDER = "cleaned_data_set"
counts = {}

CHAR_WIDTH = 22
CAPTCHA_WIDTH = 200
CAPTCHA_HEIGHT = 80
MAX_LINE_THICKNESS = 8
MIN_DISTANCE = 7
#MAX_LINE_THICKNESS = 15


def check_image(img):
    assert img is not None, 'cannot read image'
    assert img.shape == (CAPTCHA_HEIGHT, CAPTCHA_WIDTH), 'bad image dimensions'


def get_image(fpath):
    img = cv2.imread(fpath, 0)
    check_image(img)
    return cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]


def segment(img):
    chars = []
    lineStarted = False
    lineEnded = False
    foundStart = False
    startX = 0
    # endX = 0
    for col in range(img.shape[1]):
        isWhite = True
        cuts = 0
        thickness = 0
        distance = 0
        for x in range(0, CAPTCHA_HEIGHT):
            if (img[x, col] < 255):
                lineStarted = True
                if (isWhite):
                    cuts = cuts + 1
                    isWhite = False
                thickness = thickness + 1
                distance = distance + 1
            else:
                isWhite = True
            if (x == CAPTCHA_HEIGHT-1):
                # print("cuts: " + str(cuts))
                if (lineStarted and cuts == 0):
                    lineEnded = True
                if (lineEnded):
                    if (cuts >= 1 and not (foundStart)):
                        startX = col
                        foundStart = True
                    elif (lineEnded and foundStart and cuts == 0):
                        crop_img = img[0:79, startX-1:col]
                        chars.append(cv2.resize(crop_img, (25, 80)))
                        # cv2.imshow("cropped", crop_img)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        foundStart = False
                elif (not lineEnded):
                    if (cuts >= 2 and not foundStart):
                        startX = col
                        foundStart = True
                    elif (cuts <= 1 and foundStart and (not thickness > MAX_LINE_THICKNESS) and (not distance >= MIN_DISTANCE)):
                        crop_img = img[0:79, startX-1:col]
                        chars.append(cv2.resize(crop_img, (25, 80)))
                        # cv2.imshow("cropped", crop_img)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        foundStart = False
    return chars


def main():
    # Get a list of all the captcha images we need to process
    captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
    counts = {}

    # loop over the image paths
    for (i, captcha_image_file) in enumerate(captcha_image_files):

        print("[INFO] processing image {}/{}".format(i +
                                                     1, len(captcha_image_files)))
        filename = os.path.basename(captcha_image_file)
        captcha_correct_text = os.path.splitext(filename)[0]

        # Extract the letter from the original image
        captcha_chars = segment(
            get_image(CAPTCHA_IMAGE_FOLDER + "/" + filename))

        if (len(captcha_chars) == len(captcha_correct_text)):
            for letter_image, letter_text in zip(captcha_chars, captcha_correct_text):

                folder_name = letter_text
                # if letter_text.isdigit():
                #     folder_name = letter_text
                if (not letter_text.isdigit() and letter_text.islower()):
                    folder_name = "l" + letter_text
                elif (not letter_text.isdigit() and not letter_text.islower()):
                    folder_name = "u" + letter_text

                # print(folder_name)
                # Get the folder to save the image in
                save_path = os.path.join(OUTPUT_FOLDER, folder_name)

                # if the output directory does not exist, create it
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                # write the letter image to a file
                count = counts.get(letter_text, 1)
                p = os.path.join(
                    save_path, "{}.png".format(str(count).zfill(6)))

                cv2.imwrite(p, letter_image)

                # increment the count for the current key
                counts[letter_text] = count + 1
        else:
            with open("failedlist.txt", "a") as failedlist:
                failedlist.write("{}\n".format(filename))
                failedlist.close()


def rescale_by_width(image, target_width, method=cv2.INTER_LANCZOS4):
    """Rescale `image` to `target_width` (preserving aspect ratio)."""
    h = int(round(target_width * image.shape[0] / image.shape[1]))
    return cv2.resize(image, (target_width, h), interpolation=method)


if __name__ == '__main__':
    main()
