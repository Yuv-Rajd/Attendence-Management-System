import pickle
import numpy as np
from flask import Flask,request,render_template,flash,redirect
import requests
import face_recognition
import json
import cv2
import face_recognition as fr
import os

app =Flask(__name__)
app.config['UPLOAD_FOLDER'] ='static/StudentImages'




# Home Page
# __________________________________________________________________________________________________________
@app.route('/face-recognition-page.html',methods=['POST','GET'])
def face_recognition_page():
    l={}
    for f in os.listdir('static/StudentImages'):
        l[(f.split("_")[0])]=(f.split("_")[1].split(".")[0])

    print(l)
    return render_template('face-recognition-page.html',list=l)
# __________________________________________________________________________________________________________



# Add new Student
# __________________________________________________________________________________________________________
def crop_image(file,Fname):
    path = os.path.join(app.config['UPLOAD_FOLDER'],Fname)
    file.save(path)
    print(path)
    image=face_recognition.load_image_file(path)
    face_locations=face_recognition.face_locations(image)
     # Add this to this model for high accuracy (nvidia recommended)model="cnn"
    for face_location in face_locations:
        top, right, bottom, left = face_location

    print(face_locations)
    face_image = image[top:bottom, left:right]
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"static/StudentCroppedImage/{Fname}",face_image)
    return path

@app.route('/newStudent.html',methods=['POST','GET'])
def newStudent():
    if request.method=='POST':
        name=request.form["name"]
        usn=request.form["usn"]
        file=request.files["file"]
        print(file.filename)
        if file.filename=="":
            return "File Not Found"
        path = crop_image(file,f"{name}_{usn}.jpg")
        return render_template('newStudent.html',msg="student added successfully")

    else:
        return render_template('newStudent.html')
# __________________________________________________________________________________________________________


# View Student
# __________________________________________________________________________________________________________
@app.route('/ViewStudent.html',methods=['POST','GET'])
def ViewStudent():
    l=[]
    for f in os.listdir('static/StudentImages'):
        print(f)
        l.append(f)
    return render_template('ViewStudent.html',list=l)
# __________________________________________________________________________________________________________



# Attendance
# __________________________________________________________________________________________________________
def attendanceCheck(image):
    # save the image first
    imagePath="static/AttendanceImages/unchecked.jpg"
    image.save(imagePath)
    file = open('face_encodes_data.dat', 'rb')
    known_face_encodess = pickle.load(file)
    # create list of name and encodings separately
    known_img_names = []
    known_img_encodings = []
    for key, value in known_face_encodess.items():
        known_img_names.append(key)
        known_img_encodings.append(value)

    # load the group photo
    group_image = fr.load_image_file(imagePath)
    group_photo = cv2.cvtColor(group_image, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(group_photo)
    face_encodingg = face_recognition.face_encodings(group_photo, faces)


    i = 0
    detected_faces = {}

    for face_encoding in face_encodingg:
        res = face_recognition.face_distance(known_img_encodings, face_encoding)
        min_index = np.argmin(res)
        min_value = res[min_index]

        if min_value <= 0.5:
            detected_faces[(known_img_names[min_index].split("_")[0])]=(known_img_names[min_index].split("_")[1])
            # draw rectangle to this face
            facelocation = faces[i]
            # draw the reactangle
            start = (facelocation[3], facelocation[0])
            end = (facelocation[1], facelocation[2])

            img = cv2.rectangle(group_photo, start, end, (256, 0, 0), 2)
            cv2.putText(img, known_img_names[min_index].split("_")[0], start, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        i += 1

    




    cv2.putText(img, f"total face detected : {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 2)
    cv2.putText(img, f"total know face     : {len(detected_faces)}", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1),2)

    cv2.imwrite("static/AttendanceImages/checked.jpg",img)

    data={"path":"AttendanceImages/checked.jpg","studentNames":detected_faces,"TotalAttendance":len(detected_faces),"TotalNumOfStudents":i}

    return data

@app.route('/Attendance.html',methods=['POST','GET'])
def Attendance():
    if request.method=='POST':
        image=request.files["file"]
        # print(image.filename)
        if image.filename=="":
            return "File Not Found"

        data=attendanceCheck(image)
        print(data)
        return render_template('Attendance.html',msg1=f"Total Number Of Faces : {data['TotalNumOfStudents']}",
                               msg2=f"Total Number of Students :  {data['TotalAttendance']}",
                               image=data['path'],sName=data['studentNames'],flag=1)

    else:
        return render_template('Attendance.html',flag=0,image="uiui.jpg")

# __________________________________________________________________________________________________________




# Build Model
# __________________________________________________________________________________________________________
def encodeImages():
    facename = []
    face_encodings = []
    for file in os.listdir("static/StudentCroppedImage/"):
        facename.append(file.split(".")[0])
        face_encodings.append(fr.face_encodings(fr.load_image_file(f"static/StudentCroppedImage/{file}"))[0])

    print(facename)
    face_encodes = {}
    for i in range(0, len(facename)):
        face_encodes[facename[i]] = face_encodings[i]

    file = open('face_encodes_data.dat', 'wb')
    pickle.dump(face_encodes, file)
    file.close

@app.route('/BuildModel.html',methods=['POST','GET'])
def BuildModel():
    encodeImages()
    l = {}
    for f in os.listdir('static/StudentImages'):
        l[(f.split("_")[0])] = (f.split("_")[1].split(".")[0])
    return render_template('face-recognition-page.html',msg="Model Built Successfully",flag=1, list=l)
# __________________________________________________________________________________________________________


# login index
# __________________________________________________________________________________________________________
@app.route('/',methods=['POST','GET'])
def index():
    if request.method=='POST':
        username=request.form['username']
        password=request.form['password']

        if username =='admin' and password =='1234':
            l = {}
            for f in os.listdir('static/StudentImages'):
                l[(f.split("_")[0])] = (f.split("_")[1].split(".")[0])

            print(l)
            return render_template('face-recognition-page.html', list=l)
        else:
            return render_template('index-page.html',msg='Either username or password is wrong')
    return render_template('index-page.html')


if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0')
# __________________________________________________________________________________________________________