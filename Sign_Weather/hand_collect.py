import sys
sys.path.insert(0, 'pytorch-openpose/')
import cv2
from src import model
from src import util
from src.body import Body
from src.hand import Hand
import matplotlib.pyplot as plt
import copy
import numpy as np
import os.path
import pandas as pd

class OpenPose():
    def __init__(self,modelpath):  
        self.body_estimation=Body(modelpath+'/body_pose_model.pth')
        self.hand_estimation = Hand(modelpath+'/hand_pose_model.pth')

    def handpt(self, oriImg):    
        candidate, subset = self.body_estimation(oriImg)
        canvas = copy.deepcopy(oriImg)
        canvas = util.draw_bodypose(canvas, candidate, subset)

        hands_list = util.handDetect(candidate, subset, oriImg)
        all_hand_peaks = np.array([])
        cnt=0
        
        for x, y, w, is_left in hands_list:
            peaks = self.hand_estimation(oriImg[y:y+w, x:x+w, :])
            peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
            peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)

            # x좌표 먼저 저장
            all_hand_peaks= np.append(all_hand_peaks,peaks[:, 0])
            all_hand_peaks= np.append(all_hand_peaks,peaks[:, 1])

        '''canvas = util.draw_handpose(canvas, all_hand_peaks)
        plt.imshow(canvas[:, :, [2, 1, 0]])
        plt.axis('off')
        plt.show()'''
        return all_hand_peaks
    
if __name__ == "__main__":
    
    df.to_csv('hand_keypoint.csv',header=False, index=False) # csv파일로 저장       
    vidcap.release()
