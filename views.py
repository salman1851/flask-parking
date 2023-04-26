from flask import Blueprint, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from shapely.geometry import Polygon
import urllib

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import argparse
import os
import glob

model = os.getcwd()+'/flask_parking/detectron2_parking_model/model_final.pth'
config = os.getcwd()+'/flask_parking/detectron2_parking_model/config.yml'
detectron_threshold = 0.8
iou_threshold = 0.2
image_path = os.getcwd()+'/flask_parking/static/parking_sample_snap.png'
save_path = os.getcwd()+'/flask_parking/static/parking_sample_snap_inf.png'


def get_model(model_path, config_path, detectron_threshold):
    # Create config
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = detectron_threshold
    cfg.MODEL.WEIGHTS = model_path

    return DefaultPredictor(cfg), cfg


def get_resize_factor(source_width, source_height, target_width, target_height):
    x_factor = target_width / source_width
    y_factor = target_height / source_height
    return min(x_factor, y_factor)


main = Blueprint('main', __name__)


@main.after_request
def add_header(response):
    response.cache_control.max_age = 0
    return response


@main.route('/')
def index():
    return render_template('index.html')


@main.route('/address_post', methods=['POST'])
def address_post():
    new_address = request.form.get('new_address')
    print('new address ', new_address)

    filename = os.getcwd()+'/flask_parking/static/camera_address.txt'
    f = open(filename, "w")
    f.write(new_address)
    f.close()

    ########## Capture latest parking scene ##########
    source_address_path = os.getcwd()+'/flask_parking/static/camera_address.txt'
    with open(source_address_path, 'r') as file:
        source_address = file.readline()
    if source_address == '':
        print('empty source address, loading default image')
        img = cv2.imread(os.getcwd()+'/flask_parking/static/default_image.png')
        cv2.imwrite(os.getcwd()+'/flask_parking/static/parking_sample_snap.png',img)
    else:
        print('source found ', source_address)
        cap = cv2.VideoCapture(source_address)
        ret, frame = cap.read()
        if cap.isOpened():
            _,frame = cap.read()
            if _ and frame is not None:
                cv2.imwrite(os.getcwd()+'/flask_parking/static/parking_sample_snap.png',frame)
                cap.release()
    ########## Capture latest parking scene ##########
    
    return redirect(url_for('main.index'))


@main.route('/mark_spots')
def mark_spots():
    
    filename = os.getcwd()+'/flask_parking/static/coordinates.txt'
    img = cv2.imread(
        os.getcwd()+'/flask_parking/static/parking_sample_snap.png')
    r_c_points = []
    if os.stat(filename).st_size > 0:   # if coordinates file is not empty
        line_num = 0
        coor_counter = 0
        id_array = []
        coor_array = []
        resized_coor = []
        with open(filename) as f:
            for line in f:
                if line_num % 5 == 0:
                    coordinate_id = int(line[0])
                    id_array.append(coordinate_id)
                    temp_coor = []
                else:
                    coordinates = line
                    x = int(coordinates.split(",")[0])
                    y = int(coordinates.split(",")[1])
                    temp_coor.append((x, y))
                    coor_counter += 1
                    if coor_counter == 4:
                        coor_counter = 0
                        coor_array.append(temp_coor)
                line_num += 1
        f.close()
        # print('coordinate ids: ', id_array)
        # print('coordinates array: ', coor_array)

        height, width, channels = img.shape
        resize_factor = get_resize_factor(800, 450, width, height)

        for i, o in enumerate(id_array):
            resized_coordinates = [(round(x*resize_factor,2), round(y*resize_factor,2))
                                    for x, y in coor_array[i]]
            resized_coor.append(resized_coordinates)
            points = np.array([resized_coordinates], np.int32)
            r_c_points.append(points)
            # cv2.polylines(img, [points], True, (0, 0, 255), 2)
        # print('resized coordinates ', resized_coor)
    cv2.polylines(img, r_c_points, True, (0, 0, 255), 2)        # draw spot polygons on the image
    cv2.imwrite(os.getcwd()+'/flask_parking/static/parking_sample_snap_poly.png',img)   # save image
    return render_template('mark_spots.html')

@main.route("/save_coordinates/", methods=['POST'])
def save_coordinates():
    coordinates = request.form.get('coordinates')
    num_coordinates = 0
    for i in coordinates:
        if i == '\n':
            num_coordinates += 1

    if num_coordinates == 4:
        f_path = os.getcwd()+'/flask_parking/static/coordinates.txt'
    
        f = open(f_path, "r")
        content = f.read()
        f.close()
    
        if content == "":
            f = open(f_path, "a")
            f.write("1\n")
            f.write(coordinates)
            f.close()
        else:
            if len(coordinates) != 0:
                lines = content.split("\n")
                fifth_last_line = lines[-6]
                number = int(fifth_last_line)
                number += 1
                # print('last serial number ', number)
                f = open(f_path, "a")
                f.write(str(number)+'\n')
                f.write(coordinates)
                f.close()
    return redirect(url_for('main.mark_spots'))

@main.route("/delete_last_coordinate")
def delete_last_coordinate():
    f_path = os.getcwd()+'/flask_parking/static/coordinates.txt'
    f = open(f_path,'r')
    lines = f.readlines()
    lines = lines[:-5]
    f.close()
    f = open(f_path,'w')
    f.writelines(lines)
    f.close()
    return redirect(url_for('main.mark_spots'))

@main.route("/delete_coordinates")
def delete_coordinates():
    f_path = os.getcwd()+'/flask_parking/static/coordinates.txt'
    with open(f_path, 'w') as f:
        f.write('')
    f.close()
    return redirect(url_for('main.mark_spots'))


@main.route('/view_available_spots')
def view_available_spots():
    ########## Capture latest parking scene ##########
    source_address_path = os.getcwd()+'/flask_parking/static/camera_address.txt'
    with open(source_address_path, 'r') as file:
        source_address = file.readline()
    if source_address == '':
        print('empty source address, loading default image')
        img = cv2.imread(os.getcwd()+'/flask_parking/static/default_image.png')
        cv2.imwrite(os.getcwd()+'/flask_parking/static/parking_sample_snap.png',img)
    else:
        print('source found ', source_address)
        cap = cv2.VideoCapture(source_address)
        ret, frame = cap.read()
        if cap.isOpened():
            _,frame = cap.read()
            if _ and frame is not None:
                cv2.imwrite(os.getcwd()+'/flask_parking/static/parking_sample_snap.png',frame)
                cap.release()
    ########## Capture latest parking scene ##########
    
    ########## Read spot coordinates ##########
    filename = os.getcwd()+'/flask_parking/static/coordinates.txt'
    img = cv2.imread(
        os.getcwd()+'/flask_parking/static/parking_sample_snap.png')
    coor_array = []
    resized_coordinates = []
    r_c_points = []
    if os.stat(filename).st_size > 0:   # if coordinates file is not empty
        line_num = 0
        coor_counter = 0
        id_array = []
        coor_array = []
        resized_coor = []
        with open(os.getcwd()+'/flask_parking/static/coordinates.txt') as f:
            for line in f:
                if line_num % 5 == 0:
                    coordinate_id = int(line[0])
                    id_array.append(coordinate_id)
                    temp_coor = []
                else:
                    coordinates = line
                    x = int(coordinates.split(",")[0])
                    y = int(coordinates.split(",")[1])
                    temp_coor.append((x, y))
                    coor_counter += 1
                    if coor_counter == 4:
                        coor_counter = 0
                        coor_array.append(temp_coor)
                line_num += 1
        f.close()
        height, width, channels = img.shape
        resize_factor = get_resize_factor(800, 450, width, height)

        for i, o in enumerate(id_array):
            resized_coordinates = [(round(x*resize_factor,2), round(y*resize_factor,2))
                                    for x, y in coor_array[i]]
            resized_coor.append(resized_coordinates)
            points = np.array([resized_coordinates], np.int32)
            r_c_points.append(points)
    ########## Read spot coordinates ##########

    ########## Detectron inference ##########
    predictor, cfg = get_model(model, config, detectron_threshold)
    image = img
    blended_img = img
    outputs = predictor(image)
    ########## Detectron inference ##########
    
    ########## Inference mask ##########
    contours = []
    masks = outputs["instances"].to("cpu").pred_masks.numpy()
    for mask in masks:
        contour, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours.append(contour[0])
    
    inf_mask = np.zeros(img.shape, np.uint8)
    inf_mask = cv2.drawContours(inf_mask, contours, -1, (255,255,255),-1)   # inference mask
    # detectron_inf_mask_save_path = os.getcwd()+'/flask_parking/static/detectron_inference_mask.png'
    # cv2.imwrite(detectron_inf_mask_save_path, inf_mask)
    ########## Inference mask ##########
    
    ########## Spot mask ##########
    spot_mask = np.zeros(img.shape, np.uint8)
    contours_spots = []
    for region in resized_coor:
        points = np.array([region], dtype=np.int32)
        contours_spots.append(points)
    spot_mask = cv2.drawContours(spot_mask, contours_spots, -1, (255,255,255),-1)   # spot mask
    # spot_mask_save_path = os.getcwd()+'/flask_parking/static/spot_mask.png'
    # cv2.imwrite(spot_mask_save_path, spot_mask)
    ########## Spot mask ##########
    
    ########## IOU thresholding ##########
    inf_spot_and = np.bitwise_and(inf_mask, spot_mask)      # intersection mask       
    inf_spot_and_gray = cv2.cvtColor(inf_spot_and, cv2.COLOR_BGR2GRAY)
    inf_spot_and_contours, _ = cv2.findContours(inf_spot_and_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)      # intersection contours
    areas = [cv2.contourArea(c) for c in inf_spot_and_contours]     # area of intersections
    
    image = np.zeros(img.shape, np.uint8)
    
    occupancy = []
    for i,spot in enumerate(r_c_points):
        
        # print('spot coordinates ', spot[0])
        x_sum = 0
        y_sum = 0
        coor_count = 0
        for xy in spot[0]:
            x_sum = x_sum + xy[0]
            y_sum = y_sum + xy[1]
            coor_count += 1
        x_c = int(x_sum/coor_count)
        y_c = int(y_sum/coor_count)
        # print('spot centres ', (x_c, y_c))
        
        image = np.zeros(img.shape, np.uint8)
        temp_spot_mask = cv2.drawContours(image, spot, -1, (255,255,255),-1) 
        temp_intersection = np.bitwise_and(inf_spot_and, temp_spot_mask)          # spot filter
        white_count_intersection = np.sum(temp_intersection == 255)
        white_count_union = np.sum(temp_spot_mask == 255)
        # print('white count intersection ', white_count_intersection)
        # print('white count union ', white_count_union)
        iou = round(white_count_intersection/white_count_union, 4)
        
        # print('iou ', iou)
        if iou >= iou_threshold:
            status = 'occupied'
            cv2.fillPoly(temp_spot_mask, spot, (0, 0, 255, 50))
            blended_img = cv2.addWeighted(temp_spot_mask, 0.5, blended_img, 1, 0)
        else:
            status = 'free'
            cv2.fillPoly(temp_spot_mask, spot, (0, 255, 0, 50))
            blended_img = cv2.addWeighted(temp_spot_mask, 0.5, blended_img, 1, 0)
        blended_img = cv2.putText(blended_img, str(i+1), (x_c,y_c), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
        occupancy.append(status)
    
    cv2.imwrite(save_path, blended_img)
    
    f = open(os.getcwd()+'/flask_parking/static/occupancy.txt', 'w')
    for i,o in enumerate(occupancy):
        f.write(str(i+1) + ' ' + o + '\n')
    f.close()
    
    cv2.polylines(inf_spot_and, r_c_points, True, (0, 0, 255), 2)
    and_mask_save_path = os.getcwd()+'/flask_parking/static/and_mask.png'
    # cv2.imwrite(and_mask_save_path, inf_spot_and) 
    
    # detectron_inf_save_path = os.getcwd()+'/flask_parking/static/detectron_inference.png'
    # v = Visualizer(img[:, :, ::-1],
    #                 MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imwrite(detectron_inf_save_path, v.get_image()[:, :, ::-1])

    # cv2.polylines(img, r_c_points, True, (0, 0, 255), 2)
    # cv2.imwrite(os.getcwd()+'/flask_parking/static/parking_sample_snap_poly.png',img)   # draw spot polygons on the image
    
    ########## IOU thresholding ##########

    return render_template('view_available_spots.html')

@main.route('/inference')
def inference():
    return render_template('inference.html')