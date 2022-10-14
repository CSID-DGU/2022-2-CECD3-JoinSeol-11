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
    flist=list()
    modelpt='/pytorch-openpose/model' #모델 파일 위치
    
    # 사용할 변수 생성
    pose=OpenPose(modelpt)
    arr = []
    df = pd.DataFrame(columns=range(84))

    #파일 Path 설정
    base_path = '/AIHUB/WEATHER'
    file_list = os.listdir(base_path)
    file_list.sort()
    print(file_list)
    
    for classPath in file_list:
      vid_list = os.listdir(base_path+'/'+classPath)
      vid_list.sort()
      print('ClassPath: ' + classPath)
      
      for vid in vid_list:
        vidcap = cv2.VideoCapture(base_path + '/' + classPath + '/' + vid)
        label = int(classPath) #Label명 가져오기
        count = 0
        temp_peaks = np.array([]) #이전 frame의 keypoint
        print('Video: ' + vid)

        while(vidcap.isOpened()):
            ret, image = vidcap.read()
            if not ret:
              break
            # 이미지 사이즈 변경
            #image = cv2.resize(image, (3024, 4032))

            #영상 길이에 따라 이미지 추출 간격 조절
            total_frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
            frame = int(total_frame_count/20)

            # n 프레임당 하나씩 이미지 추출
            if(int(vidcap.get(1)) % frame == 0 and count < 20):

                #hand detect
                position = pose.handpt(image)

                #shape 수정 및 frame, label 추가
                out_arr = np.ravel(position, order='C') #1차원으로 만들기(0:84)
                out_arr = np.append(out_arr, count) #84에 frame번호
                out_arr = np.append(out_arr, label) #85에 label번호
                
                #손인식 안될때 오류 수정
                if out_arr.size == 44:
                    out_arr = temp_peaks.reshape(1, 86)
                if out_arr.size == 86:
                    temp_peaks = out_arr
                    out_arr = out_arr.reshape(1, 86)
                    
                #데이터 한 줄로 만들고 dataframe 생성
                out_arr = out_arr.reshape(1,86)
                out_df = pd.DataFrame(out_arr)
                df = df.append(out_df)

                # 추출된 이미지 저장
                #cv2.imwrite("/NIA/frame%s.jpg" % count, image)
                #print('Saved frame%d.jpg' % count)
                count += 1

    df.to_csv('hand_keypoint.csv',header=False, index=False) # csv파일로 저장       
    vidcap.release()
