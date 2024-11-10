
import os

import PIL.Image as Image

# 前提是拥有所有子图
import Insertion_db


def mergeinpainting(srcpath,dstpath,savedb=False):
    gap = 100
    subsize = 640
    slide=subsize-gap
    infolist=[]
    taille_dict={'ELLX':(0,0),'KPSP':(0,0),'LFRS':(0,0),'PHNL':(0,0)}
    num_dict = {'ELLX': 0, 'KPSP': 0, 'LFRS': 0, 'PHNL': 0}
    num=0
    prename='ELLX'
    for fullname in os.listdir(srcpath):
        if os.path.splitext(fullname)[1] == '.png':
            splitlines = fullname.split('.')[0].split('_')
            origin_name=fullname.split('.')[0].split('_')[0]
            left = splitlines[4]
            up = splitlines[7]
            if (prename != origin_name):
                num=0
            if(int(left)>taille_dict[origin_name][0] or int(up)>taille_dict[origin_name][1]):
                taille_dict[origin_name]=(int(left),int(up))
            if len(splitlines) == 8 or len(splitlines)==9:
                num=num+1
                num_dict[origin_name]=num
                prename=origin_name
                infolist.append([fullname, left, up])
    for key,value in taille_dict.items():
        width=value[0]
        height=value[1]
        if value[0]%slide!=0:
            width=value[0]+subsize
        if value[1]%slide!=0:
            height=value[1]+subsize
        taille_dict[key] = (width,height)
    newimage=True
    ind=0
    for info in infolist:
        fullname=info[0]
        left=int(info[1])
        up=int(info[2])
        imgname=fullname.split('.')[0].split('_')[0]
        if newimage:
            to_image = Image.new('RGB', (taille_dict[imgname][0], taille_dict[imgname][1]),color="white")
            # plt.figure('2')
            # plt.imshow(to_image)
            newimage=False
            ind=0
        imgpath=os.path.join(srcpath,fullname)
        from_image=Image.open(imgpath)
        # plt.figure('1')
        # plt.imshow(from_image)
        to_image.paste(from_image, (left, up))
        ind=ind+1

        if(ind==num_dict[imgname]):
            savepath=os.path.join(dstpath,'{}.png'.format(imgname))
            to_image.save(savepath)
            # plt.figure("paste")
            # plt.imshow(to_image)
            # plt.show()
            newimage=True
            if savedb:
                Insertion_db.update_image_apres(imgname,savepath,to_image,'merge')








if __name__ == '__main__':

    srcpath = r'C:\Users\86139\Downloads\rotation-yolov5-master\interface\inpainting\images'
    dstpath = r'C:\Users\86139\Downloads\rotation-yolov5-master\interface\inpainting\images'
    mergeinpainting(srcpath, dstpath)