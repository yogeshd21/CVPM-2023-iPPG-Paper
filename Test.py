## Author: Yogesh Deshpande Aug 2021 - May 2023

import pandas as pd
from pathlib import Path
import cv2
import os
from scipy import signal
from preprocess import preprocesses
import h5py
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

def datamaker(dir_path,frame_path,align_frm):
    imdir = Path(dir_path)
    Videolist = list(imdir.glob(r'**/*.avi'))
    ##BVPlist = list(imdir.glob(r'**/*.csv'))
    BVPlist = list(imdir.glob(r'**/*.hdf5'))
    
    Videolist = [str(i) for i in Videolist]
    BVPlist = [str(i) for i in BVPlist]

    Videolist.sort(key=natural_keys)
    BVPlist.sort(key=natural_keys)

    # Videolist = sorted(Videolist, key = os.path.getmtime)
    # BVPlist = sorted(BVPlist, key = os.path.getmtime)

    print(Videolist)
    print(BVPlist)

    j = 1
    frames_out = pd.DataFrame(columns=['Filepath', 'BVP Values'])
    
    for v in range(len(Videolist)):
        cap = cv2.VideoCapture(str(Videolist[v]))
        ##bvpval = pd.read_csv(BVPlist[v], names=['BVP Values'])
        ##resmplbvp = list(bvpval['BVP Values'][:])
        
        with h5py.File(BVPlist[v], "r") as f:
            resmplbvp = list(f['pulse'])
        
        f = list(signal.resample(resmplbvp, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
        #f = [round(i, 3) for i in f]
        i = 0
        path = frame_path
        if not os.path.exists(path + '/' + str(v + 1)):
            os.makedirs(path + '/' + str(v + 1))
            path = path + '/' + str(v + 1)
        while (cap.isOpened() and i < int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):################################ Number of frames total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


            # Capture frame-by-frame
            succ, frame = cap.read()
            if succ == True:
                cv2.imwrite(os.path.join(path, str(i+1) + '.png'), frame)
                frames_out.loc[j, 'Filepath'] = os.path.join(path, str(i+1) + '.png')
                frames_out.loc[j, 'BVP Values'] = f[i]
                j+=1

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            # Break the loop
            else:
                break
            i += 1

        # When everything done, release
        # the video capture object
        cap.release()

        # Closes all the frames
        cv2.destroyAllWindows()
        frames_out.to_csv('./myout_cohfaceframes.csv')

###### Frame -> Face
    input_datadir = frame_path
    output_datadir = align_frm

    obj = preprocesses(input_datadir, output_datadir)
    nrof_images_total, nrof_successfully_aligned = obj.collect_data()

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
    
    ####
    face = pd.DataFrame(columns=['Filepath', 'BVP Values'])
    framelst = pd.read_csv('./myout_cohfaceframes.csv')
    
    j = 1
    for i in range(len(framelst)):
        if os.path.exists('./cohfacealigned_img'+framelst['Filepath'][i][15:]):
            face.loc[j, 'Filepath'] = './cohfacealigned_img'+framelst['Filepath'][i][15:]
            face.loc[j, 'BVP Values'] = framelst['BVP Values'][i]
            j+=1
    
    face.to_csv('./myout_cohfaceface.csv')


dir_path = './cohface'
frame_path = './cohfaceFrames'
align_frm  = './cohfacealigned_img'
datamaker(dir_path, frame_path, align_frm)