import argparse
import os
import shutil
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging, rotate_non_max_suppression)
from utils.torch_utils import select_device, load_classifier, time_synchronized
import Insertion_db

def detect(opt,save_img=False):

    out, source, weights, view_img, save_txt, imgsz , facteur= \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size,opt.facteur
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    # 如果存在ouput文件夹，删掉重新创建空文件夹
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    # 存检测的信息
    os.makedirs(out+'/labels')
    os.makedirs(out+'/masques')

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    # 验证输入得图像大小是否为32倍数
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img)
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        # pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        pred = rotate_non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=True)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            mask = np.zeros_like(im0)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 6].unique():
                    n = (det[:, 6] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                # reversed转置了det矩阵，行列互换(1,7)变(7,1)
                for *xywh, conf, cls in reversed(det):

                    if save_img or view_img:  # Add bbox to image
                        label = '%.2f' % (conf)
                        txt_name=os.path.basename(txt_path).split('.')[0]
                        plot_one_box(xywh, im0, txt_name,save_txt,mask,facteur,path,save_path,label=label, color=colors[int(cls)], line_thickness=2)
                imgname = os.path.basename(save_path).split('.')[0]
                maskpath = out+'/masques/'+imgname+'_mask.png'
                print('保存有效mask {}'.format(maskpath))
                mask_out = cv2.bitwise_not(mask, mask)
                cv2.imwrite(maskpath, mask_out)
            else:
                imgname = os.path.basename(save_path).split('.')[0]
                maskpath = out + '/masques/' + imgname + '_mask.png'
                print('保存空白mask {}'.format(maskpath))
                mask_out = cv2.bitwise_not(mask, mask)
                cv2.imwrite(maskpath, mask_out)

            shutil.copy(maskpath, "./inpainting/mask")
            shutil.copy(p,  "./inpainting/images")
            savepathimg=os.path.join("./inpainting/images",os.path.basename(p))
            Insertion_db.insertion_database(savepathimg, im0, os.path.basename(p).split('.')[0])
            # Print time (inference + NMS)
            str1='%sDone. (%.3fs)' % (s, t2 - t1)
            print(str1)


            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':

                    cv2.imwrite(save_path, im0)

                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))

    print('Done. (%.3fs)' % (time.time() - t0))



if __name__ == '__main__':
    sourcepath='./big_images/raw_images'
    destpath='./big_images/split_images'
    maskpath_imgmerge = './inference/output/masques'
    srcpath_imgmerge = './inference/output'
    dstpath_imgmerge = './inpainting/images'
    maskdest_imgmerge='./inpainting/mask'
    inpainting_imgpath = './inpainting/images'
    inpainting_maskpath = './inpainting/mask'
    inpainting_rstpath='./inpainting/result'

    # t_start=time.time()
    # print("start to split images")
    # if os.path.exists(destpath):
    #     shutil.rmtree(destpath)
    # os.makedirs(destpath)
    # split = image_split.splitbase(sourcepath, destpath)
    # split.splitdata(1)
    # t_end_split = time.time()
    # print("time spent of split images is {}s".format(t_end_split - t_start))

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=r"C:\Users\86139\Downloads\e500\best.pt", help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=r"D:\BaiduNetdiskDownload\UCAS_AOD\data\PLANE", help='source')
    parser.add_argument('--data', type=str, default= 'mydatas/mycoco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--output', type=str, default='./inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', default=True,action='store_true', help='save results to *.txt')
    parser.add_argument('--classes',nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--facteur',default=1.5, help='facteur du masque')

    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect(opt)

    # t_end=time.time()
    # print("time spent from split images to finish inference is {}s".format(t_end - t_start))
    # image_merge.mergeinpainting(srcpath_imgmerge, dstpath_imgmerge)
    # image_merge.mergeinpainting(maskpath_imgmerge, maskdest_imgmerge)

    # programme_debute = time.time()
    # ind = 0
    # num_img = len(os.listdir(inpainting_imgpath))
    # if os.path.exists(inpainting_rstpath):
    #     shutil.rmtree(inpainting_rstpath)
    # os.makedirs(inpainting_rstpath)
    # while ind < num_img:
    #     programme_debute = time.time()
    #     img_courant = os.path.join(inpainting_imgpath, os.listdir(inpainting_imgpath)[ind])
    #     mask_courant = os.path.join(inpainting_maskpath, os.listdir(inpainting_maskpath)[ind])
    #     image, masque, confiance, xsize, ysize = inpainting.openImages(img_courant, mask_courant, 170)
    #     image = inpainting.createMask(image, masque, 20, xsize, ysize, confiance)
    #     filename = os.listdir(inpainting_imgpath)[ind].split('.')[0]
    #     cv2.imwrite(inpainting_rstpath+"/{}.png".format(filename.split('.')[0]), image)
    #     print("Exécution des itérations de l'image {} en {} secondes".format(img_courant,time.time() - programme_debute))
    #     ind += 1

