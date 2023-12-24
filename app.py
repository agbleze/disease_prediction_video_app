#%%
from flask import Flask, render_template, Response
import cv2
import torch
from utils import preprocess_image, decode_output
import os
from PIL import Image
import numpy as np



curr_dir = os.getcwd()

full_model_save_path = os.path.join(curr_dir, "model_save", "full_model.pth")

model = torch.load(full_model_save_path)
model.eval()

app = Flask(__name__)

camera = cv2.VideoCapture(0)


def gen_frames():  # generate frame by frame from camera
    bgr_color = (0, 255, 0)
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX 
    #org = (50, 50) 
    fontScale = 0.6
    color = (255, 0, 0) 
    #thickness = 1
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame -- in default size
        if not success:
            break
        else:
            actual_width, actual_height, n_channel = frame.shape
            resize_width, resize_height = 640, 640 
    
            width_ratio = actual_width/ resize_width
            height_ratio = actual_height / resize_height
        
            frame_resize = cv2.resize(frame, (resize_width,resize_height))
            img_arr = np.array(frame_resize)
            img_norm_array = np.array(frame_resize)/255
            prep_img = preprocess_image(img=img_norm_array)
            model_res = model([prep_img])  ## prediction is made on resize frame of 640, 640
            
            bbs, confs, labels = decode_output(model_res[0])
            n=len(labels)
            if len(bbs) == 0:
                x1,y1, x2, y2 = 0, 0, 0, 0
                frame = cv2.rectangle(img=frame,
                            pt1=(x1, y1),
                            pt2=(x2, y2),
                            color=(0, 255, 0),
                            thickness=0
                            )
                
                frame = cv2.putText(img=frame, text="No prediction", 
                        org= (50, 50), 
                            fontFace= font,  
                            fontScale=fontScale, 
                            color=color, 
                            thickness=thickness, 
                            lineType=cv2.LINE_AA
                            )
                #img_fromarr = Image.fromarray(frame, 'RGB')
                #return img_fromarr
            else:
                # Draw a rectangle around the faces
                for i in range(n):
                    x1, y1, x2, y2 = bbs[i]
                    
                    ## resize bbs to sync with actual frame size
                    x1 = int(x1 * width_ratio)
                    x2 = int(x2 * width_ratio)
                    y1 =  int(y1 * height_ratio)
                    y2 = int(y2 * height_ratio)
                    
                    label = labels[i]
                    #bgr = (0,0, 255)
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 1)
                    
                    
                    frame = cv2.putText(img=frame, text=label, 
                                org= (x1+15, y1+15), 
                                    fontFace= font,  
                                    fontScale=fontScale, 
                                    color=color, 
                                    thickness=thickness, 
                                    lineType=cv2.LINE_AA
                                    )
                    
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=8080)
# %%
