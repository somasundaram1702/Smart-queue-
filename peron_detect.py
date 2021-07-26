
import numpy as np
from openvino.inference_engine import IENetwork,IECore
from openvino.inference_engine import IEPlugin
import os
import cv2
import argparse
import time

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
        self.mod_bin = model
        self.mod_xml = '.'+self.mod_bin.split('.')[1]+'.xml'
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
        #print(self.net_input_shape)
        self.frame = cv2.resize(image, (self.net_input_shape[3], self.net_input_shape[2]))
        self.frame = self.frame.transpose((2,0,1))
        self.frame = self.frame.reshape(1, *self.frame.shape)
        return(self.frame)

        
    def predict(self, image):
        #print('entered predict')
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

    def count_person(self,result):
        d=0
        for i in range(200):
            if result[0][0][i][2]>0.5:
                d+=1
                #print('A person detected')
        return(d)


'''
    def preprocess_outputs(self, outputs):
   
    raise NotImplementedError
'''


def main(args):
    extensions=args.extensions
    model=args.model
    device=args.device
    #visualise=args.visualise

    start=time.time()
    pd=PersonDetect()
    pd.load_model(model,device)
    mod_inp_shape = pd.inp_out_blob()

    print(f"Time taken to load the model is: {time.time()-start}")
    lenr = [[620, 1, 915, 562], [1000, 1, 1264, 461]]
    lenm = [[15, 180, 730, 780],[921, 144, 1424, 704]]
    lent = [[50, 90, 838, 794],[1000, 74, 1915, 841]]
    
     #Queue Parameters
        # For retail
        #queue.add_queue([620, 1, 915, 562])
        #queue.add_queue([1000, 1, 1264, 461])
        # For manufacturing
        #[15, 180, 550, 780]
        #[600, 144, 1000, 704]
        #queue.add_queue([15, 180, 730, 780])
        #queue.add_queue([921, 144, 1424, 704])
        # For Transport 
        #queue.add_queue([50, 90, 838, 794])
        #queue.add_queue([852, 74, 1430, 841])


    try:
        queue=Queue()
        queue.add_queue([620, 1, 915, 562])
        queue.add_queue([1000, 1, 1264, 461])
        video_file=args.video
        cap=cv2.VideoCapture(video_file)
        count = 0
        
        while cap.isOpened():
            ret, frame=cap.read()
            
            if not ret:
                break
            
            count+=1
            count2 = 0
            for crop_frame in queue.get_queues(frame):
                cv2.imwrite('./out_check/manu_'+str(count2)+'.jpg',crop_frame)
                frame = pd.preprocess_input(crop_frame)
                pd.predict(frame)
                # Get the output of inference
                if pd.wait() == 0:
                    count2+=1
                    result = pd.get_output()
                    ppl=pd.count_person(result)
                    print('Number of people in Queue_'+str(count2)+' is {}'.format(ppl))                 
        

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference now", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--extensions', default=None)
    
    parser.add_argument('--visualise', default=None)
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--max_people', default=None)
    parser.add_argument('--threshold', default=None)
    
    args=parser.parse_args()

    main(args)
