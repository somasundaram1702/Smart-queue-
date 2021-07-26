
import numpy as np
from openvino.inference_engine import IENetwork,IECore
from openvino.inference_engine import IEPlugin
import os
import cv2
import argparse
import time
import matplotlib.pyplot as plt

class Queue:
    '''
    Class for dealing with queues
    '''
    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame
    
    def check_coords(self, coords):
        d={k+1:0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0]>q[0] and coord[2]<q[2]:
                    d[i+1]+=1
        return d

class PersonDetect:
    def __init__(self):
        self.plugin = None
        self.network = None
        self.mod_bin = None
        self.mod_xml = None
        self.exec_network = None
        self.input_blob = None
        self.output_blob = None
        self.net_input_shape = None
        self.frame = None


#Loading the model
    def load_model(self,model,device):
        self.mod_bin = model+'.bin'
        self.mod_xml = model+'.xml'
        self.plugin = IECore()
        self.network = IENetwork(model=self.mod_xml, weights=self.mod_bin)
        self.exec_network = self.plugin.load_network(self.network,device_name=device)
        #raise NotImplementedError

    def inp_out_blob(self):
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        self.net_input_shape=self.network.inputs[self.input_blob].shape
        return(self.net_input_shape)

    def preprocess_input(self, image):
        self.x=self.net_input_shape[3]
        self.y=self.net_input_shape[2]
        self.Y,self.X,_ = np.shape(image)
        self.frame_resized = cv2.resize(image, (self.net_input_shape[3], self.net_input_shape[2]))
        self.frame = self.frame_resized.transpose((2,0,1))
        self.frame = self.frame.reshape(1, *self.frame.shape)
        return(self.frame,self.frame_resized)

        
    def predict(self, image):
        self.exec_network.start_async(request_id=0, inputs={self.input_blob: image})
        while True:
            status = self.exec_network.requests[0].wait(-1)               
            if status == 0:
                break
            else:
                time.sleep(1)
        return(self)

 
    def wait(self):
        status = self.exec_network.requests[0].wait(-1)
        return status

    def get_output(self):
        return self.exec_network.requests[0].outputs[self.output_blob]

    def preprocess_output(self,result,thresh):
        for i in range(len(result[0][0])):
            if result[0][0][i][2]>thresh:
                #print('entered')
                x1,y1,x2,y2 = result[0][0][i][3],result[0][0][i][4],result[0][0][i][5],result[0][0][i][6]
                
                self.x1 = x1*self.net_input_shape[3]*(self.X/self.x)
                self.y1 = y1*self.net_input_shape[2]*(self.Y/self.y)
                self.x2 = x2*self.net_input_shape[3]*(self.X/self.x)
                self.y2 = y2*self.net_input_shape[2]*(self.Y/self.y)
                
                yield(int(self.x1),int(self.y1),int(self.x2),int(self.y2))
            else:
                yield(0,0,0,0)


def main(args):
#extensions=args.extensions
    model= args.model
    device=args.device
    #visualise=args.visualise
    queue_param = args.queue_param
    max_people = int(args.max_people)
    #visualize = True
    video_file=args.video
    thresh = float(args.threshold)

    start=time.time()
    pd=PersonDetect()
    pd.load_model(model,device)
    mod_inp_shape = pd.inp_out_blob()

    print(f"Time taken to load the model is: {time.time()-start}")

    queue=Queue()
    #Queue Parameters
    if queue_param == 'retail':
       # For retail
        queue.add_queue([620, 1, 915, 562])
        queue.add_queue([1000, 1, 1264, 461])

    if queue_param == 'manufacturing':
       # For manufacturing
        queue.add_queue([15, 180, 730, 780])
        queue.add_queue([921, 144, 1424, 704])

    if queue_param == 'transportation':
       # For transportation
        queue.add_queue([150, 0, 1150, 794])
        queue.add_queue([1151, 0, 1915, 841])


    cap=cv2.VideoCapture(video_file)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    vid_write = cv2.VideoWriter(str(queue_param)+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10,(int(width),int(height)))
    #print(video_fps)
    #print(width,height)
    count = 0


    while cap.isOpened():
        ret, frame_org=cap.read()

        count+=1
        if not ret:
            break
        w,h,_ = np.shape(frame_org)
        #print(w,h)
        #for crop_frame in queue.get_queues(frame):

        frame, frame_resized = pd.preprocess_input(frame_org)
        pd.predict(frame)
        cords=[]
        # Get the output of inference
        if pd.wait() == 0:

            result = pd.get_output()
            #print(np.shape(result))

        for x1,y1,x2,y2 in pd.preprocess_output(result,thresh):
            cv2.rectangle(frame_org,(x1,y1),(x2,y2),(0,0,255),2)

            cords.append([x1,y1,x2,y2])
            #ppl=pd.count_person(result)
            #print('Number of people in Queue_'+str(count2)+' is {}'.format(ppl))
        #print(np.shape(frame_org))
        d = queue.check_coords(cords)
        stn_1 = 'No of person in Queue one :'
        stn_2 = 'No of person in Queue two :'
        cv2.putText(frame_org,stn_1+str(d[1]),(5,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv2.LINE_AA)
        cv2.putText(frame_org,stn_2+str(d[2]),(5,135),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv2.LINE_AA)

        mx_q1 = 'Too many persons in Queue one, move to other queue'
        mx_q2 = 'Too many persons in Queue two, move to other queue'    
        if d[1] > max_people:
            cv2.putText(frame_org,mx_q1,(5,700),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
        if d[2] > max_people:
            cv2.putText(frame_org,mx_q2,(5,725),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)


        #videowriter = cv2.VideoWriter('Haar.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (500,550))

        if args.visualize:
            cv2.imshow('output',frame_org)
            vid_write.write(frame_org)
            cv2.waitKey(1)


        else:
            print(stn_1+str(d[1]))
            print(stn_2+str(d[2]))

    print(f"Time taken for model inference is: {time.time()-start}")
    print('model runs at {} FPS'.format(round(count/(time.time()-start),2)))


    cap.release()
    cv2.destroyAllWindows()
    vid_write.release()





if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--extensions', default=None)
    
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--max_people', default=None)
    parser.add_argument('--threshold', default=None)
    
    args=parser.parse_args()

    main(args)
