# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 11:46:34 2021

@author: liuyz
"""

# from tensorflow.python.keras.applications.resnet50 import ResNet50
# from tensorflow.python.keras.applications.resnet50 import decode_predictions, preprocess_input

from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.inception_v3 import decode_predictions, preprocess_input

from tensorflow.python.keras.preprocessing import image
import numpy as np
import math
import copy
import cv2


from differential_evolution import differential_evolution



#扰动图像
def perturb_image(xs, img):
    if xs.ndim < 2:
        xs = np.array([xs])
    tile = [len(xs)] + [1]*(xs.ndim+1)
    imgs = np.tile(img, tile)
    xs = xs.astype(int)
    for x,img in zip(xs, imgs):
        pixels = np.split(x, len(x) // 5)
        for pixel in pixels:
            x_pos, y_pos, *rgb = pixel
            img[x_pos, y_pos] = rgb
    return imgs

def attack_success(x, image, target_class, model, targeted_attack=False, verbose=False):
    
    attack_image = perturb_image(x, image)
    preds = model.predict(attack_image)
    predicted = decode_predictions(preds, top=1)[0]
    predicted_class = predicted[0][0]
    if (verbose):
        print('Confidence:', predicted[0][2])
    if ((targeted_attack and predicted_class == target_class) or
        (not targeted_attack and predicted_class != target_class)):
        return True

def predict_classes(image, model, minimize=True):
    if image.ndim < 4:
        image = np.array([image])
    
    image = preprocess_input(image)
    preds = model.predict(image)
    predicted = decode_predictions(preds, top=1)
    return predicted

def distance(xs, img):
    if xs.ndim < 2:
        xs = np.array([xs])

    xs = xs.astype(np.int32)
    
    #找到图像某个坐标的周边坐标的下数
    around = [[] for i in range(xs.shape[0])]
    for i in range(0, xs.shape[0]):
        around[i] = [[xs[i][0]-1,xs[i][1]-1], [xs[i][0]-1,xs[i][1]], [xs[i][0]-1,xs[i][1]+1], 
               [xs[i][0],xs[i][1]-1], [xs[i][0],xs[i][1]], [xs[i][0],xs[i][1]+1],
               [xs[i][0]+1,xs[i][1]-1], [xs[i][0]+1,xs[i][1]], [xs[i][0]+1,xs[i][1]+1]]
    ard = copy.deepcopy(around)
    for i in range(0, xs.shape[0]):
        for j in range(0, 9):
            if around[i][j][0]<0 or around[i][j][1]<0 or around[i][j][0]>img.shape[0]-1 or around[i][j][1]>img.shape[1]-1:
                ard[i].remove(around[i][j])

    #像素敏感度：浮点型
    xsshape = xs.shape[0]
    pixels = [[] for _ in range(xsshape)]
#    pixels = [[]*xs.shape[0]]
    windstd = np.zeros((xs.shape[0], img.ndim), dtype=np.float32)
    windMax = np.zeros((xs.shape[0], img.ndim), dtype=np.float32)
#    windMean = np.zeros((xs.shape[0], img.ndim), dtype=np.int)
    for i in range(0, xs.shape[0]):
        for j in range(0, len(ard[i])):
            pixels[i].append(img[ard[i][j][0], ard[i][j][1]])
        na = np.array(pixels[i])
        windstd[i] = np.std(na, axis=0)
        windMax[i] = np.amax(na, axis=0)
#        windMean[i] = np.mean(na, axis=0)
        #处理标准差为0的情况
        for k in range(0, windstd.shape[1]):
            if windstd[i][k] == 0:
                windstd[i][k] = 0.0000001    #0.0000001表示标准差最小，敏感度最大
    sens=1/windstd
    
#    计算恰可感知阈值-以窗口中的最大像素值为基准
#    la = jdnMatrix(xs, windMax)
    
    #像素改变强度,浮点型
    intensity = np.zeros_like(sens)       
    for i in range(0, xs.shape[0]):
        for j in range(0, img.ndim):
            intensity[i][j] = perceivedError(windMax[i][j], xs[i][2+j] - img[xs[i][0], xs[i][1], j])*10       
    distance = intensity*sens
#   color modulation  b g r
    distanceColor = distance[:,0]*0.299 + distance[:,1]*0.587+distance[:,2]*0.114
    return distanceColor

def jndCalculate(base):
    if base <= 127:
        jnd = 17*(1 - math.sqrt(base/127))+3
    else:
        jnd = 3/128 *(base-127)+3
    return jnd

def perceivedError(base, error):
#    计算灰度值扰动error对视觉的影响
#    base:背景颜色
#    error:当前背景下像素的扰动量
    superJnd = 0
    for i in range (0,256):
        jdn = jndCalculate(i)
        superJnd = superJnd + 1/jdn    
    superSum = 256/superJnd
    
    pb = 0
    if error > 0:
        start = int(base)
        end = int(start + error)
        for i in range(start, end, 1):
            pb = pb + superSum/jndCalculate(i)
    else:
        start = int(base)
        end = int(start + error) 
        for i in range(start, end, -1):
            pb = pb + superSum/jndCalculate(i)
    return pb



#以PSD范数定义视觉距离，计算扰动优先级
def perturbPriority(xs, image, model):
    #img是图像索引,image是原图
    imgs_perturbed = perturb_image(xs, image)
    actual_predicted = predict_classes(image, model, minimize=True)
    actual_confidence = [i[0][2] for i in actual_predicted]
    perturbed_predicted = predict_classes(imgs_perturbed, model, minimize=True)
    perturbed_confidence = [i[0][2] for i in perturbed_predicted]    
    cdiffs = np.array(actual_confidence) - np.array(perturbed_confidence)
    dist = distance(xs, image)
    priority = np.zeros_like(dist)
    for i in range(0, dist.shape[0]):
        if dist[i] == 0:
            priority[i] = 1000  #值越小优先级越高，当dist==0时说明未对图像进行扰动，优先级最低。
        else:
            priority[i] = -cdiffs[i]/dist[i]  #差分进化计算最小值，因此取负
    return priority

def attack(image, model, target=None, pixel_count=1, 
           maxiter=60, popsize=300, verbose=False):
    bounds = [(0,224), (0,224), (0,256), (0,256), (0,256)] * pixel_count
    
    popmul = max(1, popsize // len(bounds))
    
    priority_fn = lambda xs: perturbPriority(xs, image, model)
    
    [attack_result, popsetAll, optimum_per_generation] = differential_evolution(
        priority_fn, bounds, maxiter=maxiter, popsize=popmul,
        recombination=1, atol=-1, callback=None, disp=True, polish=False)
    
    #对所有代按（x,y）全局去除重
    popset=np.array([popsetAll[0]])
    for i in range(0, popsetAll.shape[0]):
        flag = 0
        for j in range(0,popset.shape[0]):
            if(popsetAll[i][0]==popset[j][0] and popsetAll[i][1]==popset[j][1]):
                if (popsetAll[i][5]>=popset[j][5]):
                    popset[j,0:6]=popsetAll[i,0:6]
            else:
                flag = flag+1
        if flag == popset.shape[0]:
            popset=np.append(popset, [popsetAll[i, 0:6]], axis=0)   
    
    #贪婪算法生成最优扰动的像素组合
    popset = popset[np.lexsort(-popset.T)]  #扰动优先率排序
    dist = 0   #初始化视觉距离

    for i in range(0, popset.shape[0]):
        if i==0:
            attackx = np.array(popset[0, 0:5])
        else :
            attackx = np.append(attackx, popset[i, 0:5])
        attack_image = perturb_image(attackx, image)[0]
        predicted = predict_classes(attack_image, model, minimize=True)[0]
        preclass = predicted[0][0]
        accpredicted = predict_classes(image, model, minimize=True)[0]
        acclass = accpredicted[0][0]
        #预测错误表示攻击成功
        if (preclass != acclass):
#            计算对抗样本与原图的视觉距离
            dist = distance(np.array(attackx).reshape(len(attackx) // 5,5), image)
            break
    attackxs = np.array(attackx).reshape(len(attackx) // 5,5)   # 对抗扰动
    #原始类别
    actual_predicted = predict_classes(image, model, minimize=True)[0]  
    actual_class = actual_predicted[0][1]
    #对抗类别
    attack_predicted = predict_classes(attack_image, model, minimize=True)[0]
    attack_class = attack_predicted[0][1]
    #判断攻击是否成功
    success = attack_class != actual_class
    
    return [attack_image, actual_predicted, attack_predicted, success, dist, attackxs]

if __name__ == "__main__":
    #攻击的神经网络
    # model = ResNet50(weights='imagenet')  
    model = InceptionV3(weights='imagenet')  
    #原始图像及预处理
    img_path = './images/0.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    #实施攻击
    [attack_image, actual_predicted, attack_predicted, success, distance, attackxs] = attack(img, model)

#   画原始样本并保存 
    pltimage = img.astype(np.uint8)
    cv2.imwrite('./images/0ben.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

#   画对抗样本并保存
    pltattack_image = attack_image.astype(np.uint8)
    cv2.imwrite('./images/0adv.jpg', pltattack_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])