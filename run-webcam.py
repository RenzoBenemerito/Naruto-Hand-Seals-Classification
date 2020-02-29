"""
An Image Classification System for Naruto Hand Seals
by: Renzo Benemerito

"""

# Import Needed Packages
import numpy as np
import cv2
import imageio
from PIL import Image

from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from collections import Counter
import operator

import time
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to model")

args = vars(ap.parse_args())

labels = ['Bird', 'Boar', 'Dog', 'Dragon', 'Hare', 'Horse', 'Monkey', 'Ox', 'Ram', 'Rat', 'Snake', 'Tiger', 'Unknown']
cap = cv2.VideoCapture(0)
pause = 0
chidori = ['Ox','Hare','Monkey']
fire = ['Snake','Ram','Monkey','Boar','Horse','Tiger']
summon = ['Boar','Dog','Bird','Monkey','Ram']
predictions = []
smoothing = []
technique = ""
t_counter = 0
model = load_model(args["model"])

# Utility Functions
def shrink(label,frame,scale):
    frame = frame.copy()
    pic = cv2.imread("data/{}.png".format(label), -1)

    # Image sizes
    f_h,f_w = frame.shape[0:2]
    p_w = int(pic.shape[1] * scale / 100)
    p_h = int(pic.shape[0] * scale / 100)

    top = int((f_h/2) - (p_h/2))
    bottom = int((f_h/2) + (p_h/2))
    left = int((f_w/2) - (p_w/2))
    right = int((f_w/2) + (p_w/2))

    pic = cv2.resize(pic, (p_w,p_h))

    b,g,r,a = cv2.split(pic)
    overlay_color = cv2.merge((b,g,r))

    mask = cv2.medianBlur(a,5)

    roi = frame[top:bottom, left:right]

    # Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(roi.copy(),roi.copy(),mask = cv2.bitwise_not(mask))
    
    # Mask out the logo from the logo image.
    img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)

    frame[top:bottom, left:right ] = cv2.add(img1_bg, img2_fg)


    return frame

def render(label,frame):
    scale = 100
    orig_frame = frame
    while scale != 0:
        frame = shrink(label,orig_frame,scale)
        cv2.imshow("Frame", frame)
        scale -= 10
        cv2.waitKey(33)
        
    return frame
 
def techniques(frame,predictions,scale):
    prev_w = 0
    prev_h = 0
    frame = frame.copy()
    for pred in predictions:
        pic = cv2.imread("data/{}.png".format(pred))
    
        # Image sizes
        f_h,f_w = frame.shape[0:2]
        p_w = int(pic.shape[1] * scale / 100)
        p_h = int(pic.shape[0] * scale / 100)

        pic = cv2.resize(pic, (p_w,p_h))

        top = int((f_h/2) - (p_h/2))
        bottom = int((f_h/2) + (p_h/2))
        left = int((f_w/2) - (p_w/2))
        right = int((f_w/2) + (p_w/2))
        
        if (prev_w + p_w) > f_w:
            prev_h += 70
            prev_w = 0
        # Select the region in the background where we want to add the image and add the images using cv2.addWeighted()
        added_image = cv2.addWeighted(frame[0+prev_h:p_h+prev_h,0+prev_w:p_w+prev_w,:],1,pic[0:p_h,0:p_w,:],1,0)
        # Change the region with the result
        frame[0+prev_h:p_h+prev_h,0+prev_w:p_w+prev_w,:] = added_image
        prev_w += p_w
    
    return frame

def pulse(frame,predictions):
    scale = 20
    orig_frame = frame.copy()
    while scale != 30:
        frame = techniques(orig_frame,predictions,scale)
        cv2.imshow("Frame", frame)
        scale+=5
        cv2.waitKey(33)
    scale = 30
    while scale !=0:
        frame = techniques(orig_frame,predictions,scale)
        cv2.imshow("Frame", frame)
        scale-=5
        cv2.waitKey(33)

def render_technique(frame,imgs, scale):
    original_frame = frame.copy()
    for pic in imgs:
        # Image sizes
        frame = original_frame.copy()
        f_h,f_w = frame.shape[0:2]
        p_w = int(pic.shape[1] * scale / 100)
        p_h = int(pic.shape[0] * scale / 100)

        pic = cv2.resize(pic, (p_w,p_h))

        top = int((f_h/2) - (p_h/2))
        bottom = int((f_h/2) + (p_h/2))
        left = int((f_w/2) - (p_w/2))
        right = int((f_w/2) + (p_w/2))
        
        b,g,r,a = cv2.split(pic)
        overlay_color = cv2.merge((b,g,r))
        # img[img[:, :, 2] < 255, 2] = 255
        a[(overlay_color[:,:,1] < 40) & (overlay_color[:,:,0] < 40) & (overlay_color[:,:,2] < 40)] = 0 
        mask = cv2.medianBlur(a,5)

        roi = frame[top:bottom, left:right]

        # Black-out the area behind the logo in our original ROI
        img1_bg = cv2.bitwise_and(roi.copy(),roi.copy(),mask = cv2.bitwise_not(mask))

        # Mask out the logo from the logo image.
        img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)

        frame[top:bottom, left:right ] = cv2.add(img1_bg, img2_fg)
        cv2.imshow("Frame", frame)
        
        cv2.waitKey(30)
    
def fade_label(label,frame,scale):
    original_frame = frame.copy()
    pic = cv2.imread("data/{}.png".format(label))
    cv2.imwrite("sample.png",pic)
    # Image sizes
    f_h,f_w = frame.shape[0:2]
    p_w = int(pic.shape[1] * scale / 100)
    p_h = int(pic.shape[0] * scale / 100)

    pic = cv2.resize(pic, (p_w,p_h))

    top = int((f_h/2) - (p_h/2))
    bottom = int((f_h/2) + (p_h/2))
    left = int((f_w/2) - (p_w/2))
    right = int((f_w/2) + (p_w/2))

    counter = 0
    while counter != 25:
        frame = original_frame.copy()
        # Select the region in the background where we want to add the image and add the images using cv2.addWeighted()
        added_image = cv2.addWeighted(frame[top:bottom, left:right,:],0,pic[0:p_h,0:p_w,],1,0)
        # Change the region with the result
        frame[top:bottom, left:right, :] = added_image
        counter += 1
        cv2.imshow("Frame", frame)
        
        cv2.waitKey(30)


    fade = 1
    while fade > 0:
        frame = original_frame.copy()
        # Select the region in the background where we want to add the image and add the images using cv2.addWeighted()
        added_image = cv2.addWeighted(frame[top:bottom, left:right,:],1-fade,pic[0:p_h,0:p_w,],fade,0)
        # Change the region with the result
        frame[top:bottom, left:right, :] = added_image

        cv2.imshow("Frame", frame)
        fade -= 0.03
        
        cv2.waitKey(30)
       
while cap.isOpened():
    
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    frame = np.array(frame)
    frame_pred = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pred = preprocess_mobilenet(frame_pred)
    frame_pred = cv2.resize(frame_pred,(224,224))
    frame_pred = np.expand_dims(frame_pred,axis=0)

    # Technique Rendering
    if predictions == chidori or technique == "chidori'":
        technique = "chidori"
        t_counter += 1
        if t_counter == 20:
            pulse(frame,predictions)
            gif = imageio.mimread("chidori.gif")
            gif = gif * 2
            fade_label("chidori",frame,20)
            imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGRA) for img in gif]
            render_technique(frame,imgs,110)
            t_counter = 0
            technique = ""
            predictions.clear()
    elif predictions == fire or technique == "fire'":
        technique = "fire"
        t_counter += 1
        if t_counter == 20:
            pulse(frame,predictions)
            gif = imageio.mimread("fire.gif")
            gif = gif * 5
            imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGRA) for img in gif]
            fade_label("fire",frame,20)
            render_technique(frame,imgs,170)
            t_counter = 0
            technique = ""
            predictions.clear()
    elif predictions == summon or technique == "summon":
        technique = "summon"
        t_counter += 1
        if t_counter == 20:
            pulse(frame,predictions)
            gif = imageio.mimread("frog.gif")
            gif = gif * 2
            fade_label("summon",frame,20)
            imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGRA) for img in gif]
            frame = render("seal",frame)
            render_technique(frame,imgs,40)
            t_counter = 0
            technique = ""
            predictions.clear()

    # Render Hand Seal Symbol at Top Left
    if len(predictions) != 0:
        frame = techniques(frame,predictions,20)

    # Predict Every Third Frame
    if pause == 3:
        pred = model.predict(frame_pred)[0]
        max_pred = max(pred)
        if max_pred > 0.8:
            smoothing.append(labels[pred.argmax()])
            print(labels[pred.argmax()])
        pause=0
    pause+=1

    # Smoothing Predictions Function
    if len(smoothing) == 5:
        occurences = Counter(smoothing)
        high = sorted(occurences.items(), key=operator.itemgetter(1), reverse=True)
        sign = high[0][0]
        if sign != "Unknown":
            if len(predictions) == 0:
                predictions.append(sign)
                frame = render(sign,frame)
            elif sign != predictions[len(predictions)-1]:
                predictions.append(sign)
                frame = render(sign,frame)
        smoothing = []

    # Keyboard Mappings
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("1"):
        frame = render("Bird",frame)
        predictions.append("Bird")
        continue
    elif key == ord("2"):
        frame = render("Boar",frame)
        predictions.append("Boar")
        continue
    elif key == ord("3"):
        frame = render("Dog",frame)
        predictions.append("Dog")
        continue
    elif key == ord("4"):
        frame = render("Dragon",frame)
        predictions.append("Dragon")
        continue
    elif key == ord("5"):
        frame = render("Hare",frame)
        predictions.append("Hare")
        continue
    elif key == ord("6"):
        frame = render("Horse",frame)
        predictions.append("Horse")
        continue
    elif key == ord("7"):
        frame = render("Monkey",frame)
        predictions.append("Monkey")
        continue
    elif key == ord("8"):
        frame = render("Ox",frame)
        predictions.append("Ox")
        continue
    elif key == ord("9"):
        frame = render("Ram",frame)
        predictions.append("Ram")
        continue
    elif key == ord("0"):
        frame = render("Rat",frame)
        predictions.append("Rat")
        continue
    elif key == ord("-"):
        frame = render("Snake",frame)
        predictions.append("Snake")
        continue
    elif key == ord("="):
        frame = render("Tiger",frame)
        predictions.append("Tiger")
        continue
    elif key == ord("w"):
        predictions.clear()
    cv2.imshow("Frame", frame)
