#Main code to run the model using the trained weights


from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import threading
from matplotlib.pyplot import imshow
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras import backend as K
import os


def load_model():
    try:
        #Loading the model
        json_file = open(r"model1.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(r"weights1.hdf5")
        print("Model successfully loaded from disk.")
        
        #compile the model again
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        return model
    except:
        print("""Model not found""")
        return None
def quad_delete2(index):
    for i in range(1,len(index)):
        if index[i]=='2' and index[i-1]=='x':
            index=np.delete(index,i)
        print(index)


def get_square(image,square_size):

    height,width=image.shape
    if(height>width):
      differ=height
    else:
      differ=width
    differ+=4

    mask = np.zeros((differ,differ), dtype="uint8")   
    x_pos=int((differ-width)/2)
    y_pos=int((differ-height)/2)
    mask[y_pos:y_pos+height,x_pos:x_pos+width]=image[0:height,0:width]
    mask=cv2.resize(mask,(square_size,square_size),interpolation=cv2.INTER_AREA)

    return mask 

img=cv2.imread(r"C:\Users\dell\Desktop\equation_solver\good3.jpeg")
x=img
cv2.imshow('x',x)
cv2.waitKey(0)
#grayscaling
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
#cv2.imshow('grayscaling',gray)

#Thresholding the image
thresh = cv2.adaptiveThreshold(gray,200,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2) #thresholding
#cv2.imshow('thresholding',thresh)

#Opening
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
#cv2.imshow('Opening',opening)

#Dilation
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
kernel[(0,1),:]=0;
kernel[(3,4),:]=0;
dilated=cv2.dilate(opening,kernel,iterations=35) #dilate
cv2.imshow("dilatedfor lines.jpg",dilated) 


Im=[]
#Find the contours
contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print("contour",len(contours))
#for each contour we draw a rectangle
for i,contour in enumerate(contours):
    # get rectangle bounding contour
    [x,y,w,h] = cv2.boundingRect(contour)
    #
    #print('x is',x)

    # discard areas that are too large
    #if h>300 and w>250:
        #continue
    
    # discard areas that are too small
    if h<20 or w<20:
        continue
    
    # draw rectangle around contour on original image
    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
    
        # Getting ROI
    roi = img[int(y-0.2*h):int(y+h+0.2*h), x:x+w]
    print(roi.shape)
    # show ROI
    Im.append(roi)
    cv2.imshow('roi_imgs'+str(i)+'.png', roi)
    
    #cv2.imshow('line'+str(i), roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
     
    
#cv2.waitKey(0)
#cv2.destroyAllWindows()
   #x=img
   

print(len(Im))

coeff = []
t = 0




for img in Im:
    index = []
    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    I=img.copy()

    G_Image=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)

    font = cv2.FONT_HERSHEY_SIMPLEX
#Otsu Thresholding
    blur = cv2.GaussianBlur(G_Image,(1,1),0)
    ret,th = cv2.threshold(blur,95,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#kernel = np.ones((3,3),dtype = np.uint8)
    #th = cv2.erode(th,kernel,iterations =1)

    contours, hierarchy = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(img, contours, -1, (0,255,0), 3)
    print(len(contours))
    i = 0
    #th = get_square(th,45)
    #th2 = cv2.bitwise_not(th)
    th2 = th
# these for loops are for "=" sign 

    for k in range(len(contours)):
        for j in range(i+1,len(contours)-5):
        
            [x1, y1, w1, h1] = cv2.boundingRect(contours[i])
            [x2, y2, w2, h2] = cv2.boundingRect(contours[j])
            if (abs(y1-y2) >2)and(abs(y1-y2) < 10):
                contours = np.delete(contours,j)
            
    print(len(contours))
    X=[]
    Y=[]
    for contour in contours:
        # get rectangle bounding contour
        
        [x, y, w, h] = cv2.boundingRect(contour)
    
        # if size of image is (<45,<45) resizing it to (45,45)
        # will create lots of false pixels (0)
        # so if size < (45,45) increase dimensions of
        # cropping boxes 
    
    
        #if w <45 or h<45 and w>10:
            #cropped = th2[y-30:y+h+10,x-10:x+w+10]
            #print(cropped.shape)
           # X.append(x-10)
        
        #else:
        cropped = th2[int(y):int(y+h),int(x):int(x+w)]
        print(cropped.shape)
        X.append(x)
         
        cropped = get_square(cropped,45)
        cropped = cv2.bitwise_not(cropped)
        Y.append(cropped)
            #cv2.imshow("final",th)

    print("lengthof X and Y ",len(X),len(Y))    
  
    n = len(X)
 
    # Traverse through all array elements
    for i in range(n):
 
        # Last i elements are already in place
        for j in range(0, n-i-1):
 
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if X[j] > X[j+1] :
                Y[j], Y[j+1] = Y[j+1], Y[j]
                X[j], X[j+1] = X[j+1], X[j]
                
    
    for i in X:
        print(i)
    for i in Y:
        model=load_model()


        if model is not None:
            classes=["-", "(" , ")" ,  "+" , "-" , "0" , "1" , "2" , "3" , "4" , "5" , "6" , "7" , "8" , "9" ,"=", "x" , "y" , "z"]
            #img=cv2.imread(r"C:\Users\dell\Desktop\equation_solver\dataset\X\exp67.jpg")
            img = i
       
            print(img.shape)
            #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            x=img
            print('The symbol is')
            img=img.reshape((1,)+img.shape)
            img=img.reshape(img.shape+(1,))
            test_datagen = ImageDataGenerator(rescale=1./255)
            m=test_datagen.flow(img,batch_size=1)
            y_pred=model.predict_generator(m,1)
            print(list(y_pred[0]).index(y_pred[0].max())+1)
            print(classes[list(y_pred[0]).index(y_pred[0].max())+1])

            index.append(classes[list(y_pred[0]).index(y_pred[0].max())+1])
            
            cv2.imshow("final",x)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    
    #cv2.imwrite("cropped_images/User"+str(i)+".jpg",cropped)
    #cv2.imshow('cropped',cropped)
    #if h>300:
    #    continue
       
    # draw rectangle around contour on original image
    #cv2.rectangle(th2, (x,y), (x + w, y + h), (0,0,255), 1)

    

            
    #quad_delete2(index)       
    
    
    number=['0','1','2','3','4','5','6','7','8','9']
    notnumber = ['+','-', '(' , ')' ,'=', 'x' , 'y' , 'z']
    print("index1",index)

    b=[]
    
    for i in index:
        if i in number:
            b.append(int(i))
        else:
            b.append(i) 
                
    print(b)

    cv2.imshow('blabla',th2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    j=0
    a=0

    
   #predicted_numbers=[1,2,'x','+',5,'y','+',1,9,'=',6]
    for i in b:

      a=a+1
      if (type(i)!=int and (type(b[a-2])==int)):
        x = ""
        #coeff.append([])
        for c in range (j,a-1,1):
            x+=str(b[c])
        coeff.append(int(x))
        j=a
      elif (i=='-'):
        j=a
    x=''
    
    for c in range (j,len(b),1):
      x+=str(b[c])
    print(x)
    coeff.append(int(x))
    t+=1
    

print("coeff",coeff)            

#All Coefficients stored in same array
#Kramers Rule
def math(b):
    A=[]
    B=[]
    n=len(b)
    #3 Variables
    if n==12:
        for i in range(12):
            if (i+1)%4==0:
                B.append(b[i])
            else:
                A.append(b[i])
        A=np.reshape(A,(3,3))
        B=np.reshape(B,(3,1))
        X=np.linalg.solve(A,B)
    #2 Variables
    elif n==6:
        for j in range(6):
            if (j+1)%3==0:
                B.append(b[i])
            else:
                A.append(b[i])
        A=np.reshape(A,(2,2))
        B=np.reshape(B,(2,1))
        X=np.linalg.solve(A,B)
    #1 Varaiable
    elif n==2:
        A.append[b[0]]
        B.append[b[1]]
        X=np.linalg.solve(A,B)
        

    print(X)
#Function Call
math(coeff)

def quadratic(b):
    
    A=b[0]
    B=b[1]
    C=b[2]
    D=(B**2-(4*A*C))
    D1=(4*A*C-(B**2))

    if D>0:
        r1=(-B+np.sqrt(D))/2*A
        r2=(-B-np.sqrt(D))/2*A

    elif D==0:
        r1=-B/2*A
        r2=r1
    elif D<0:
        r1=(-B+1j*np.sqrt(D1))/2*A
        r2=(-B-1j*np.sqrt(D1))/2*A
        
    print(r1,r2)
        
#quadratic(coeff)

