# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import dlib
import os
import warper
import blender

def weighted_average_points(start_points, end_points, percent=0.5):
  """ Weighted average of two sets of supplied points

  :param start_points: *m* x 2 array of start face points.
  :param end_points: *m* x 2 array of end face points.
  :param percent: [0, 1] percentage weight on start_points
  :returns: *m* x 2 array of weighted average points
  """
  if percent <= 0:
    return end_points
  elif percent >= 1:
    return start_points
  else:
    return np.asarray(start_points*percent + end_points*(1-percent), np.int32)

path_image = sys.argv[1]
path_morph = sys.argv[2]


for images in os.listdir(path_image):
    
    names = images.split()
    image = names[1]
    image2 = names[2]
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    img_src = cv2.imread(path_image + "/" +image)
    img_src_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    height_src, width_src, channels_src = img_src.shape

    faces = detector(img_src_gray)
    for face in faces:
        landmarks_src = predictor(img_src_gray, face)
        landmarks_points_src = []
        for n in range(0, 68):
            x = landmarks_src.part(n).x
            y = landmarks_src.part(n).y
            landmarks_points_src.append((x, y))

    src_points = np.array(landmarks_points_src, np.int32)
    
    for image2 in os.listdir(path_image):
        if image != image2:                
            dest_img = cv2.imread(path_image + "/" +image2)
            img2_gray = cv2.cvtColor(dest_img, cv2.COLOR_BGR2GRAY)
    
            height, width, channels = dest_img.shape        
            faces = detector(img2_gray)
            for face in faces:
                landmarks = predictor(img2_gray, face)
                landmarks_points = []
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    landmarks_points.append((x, y))
    
            dest_points = np.array(landmarks_points, np.int32)
            convexhull2 = cv2.convexHull(dest_points)
            
            rect_src = cv2.boundingRect(np.array([src_points], np.int32))

            size = (height, width)
            points = weighted_average_points(src_points, dest_points, 0.5)
            
            # rescale point to dest_points size
            (x1, y1, w1, h1) = cv2.boundingRect(points)
            (x2, y2, w2, h2) = cv2.boundingRect(dest_points)
    
            points[:,0] = (points[:,0]-(x1 - w1/2))*(w2/w1)*.95
            points[:,1] = (points[:,1]-(x2 - w2/2))*(w2/w1)*.95
            
            (x1, y1, w1, h1) = cv2.boundingRect(points)
            (x2, y2, w2, h2) = cv2.boundingRect(dest_points)
    
            x_trans = int((x2 + x2 + w2) / 2)-int((x1 + x1 + w1) / 2)
            y_trans = int((y2 + y2 + h2) / 2)-int((y1 + y1 + h1) / 2)
            
            points[:, 0] = points[:, 0] + x_trans
            points[:, 1] = points[:, 1] + y_trans
            
            src_face = warper.warp_image(img_src, src_points, points, size)
            end_face = warper.warp_image(dest_img, dest_points, points, size)
            average_face = blender.weighted_average(src_face, end_face, 0.5)
    
            convexhull2 = cv2.convexHull(points)
    
            img2_face_mask = np.zeros_like(img2_gray)
            img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
            img2_face_mask = cv2.bitwise_not(img2_head_mask)
            img2_head_noface = cv2.bitwise_and(dest_img, dest_img, mask=img2_face_mask)
            result = cv2.add(img2_head_noface, average_face)
    
            (x, y, w, h) = cv2.boundingRect(convexhull2)
            center_face = (int((x + x + w) / 2), int((y + y + h) / 2))
            seamlessclone = cv2.seamlessClone(result, dest_img, img2_head_mask, center_face, cv2.NORMAL_CLONE)
            
            cv2.imwrite(os.path.join(path_morph, image2+"_"+image+".jpg"), face)