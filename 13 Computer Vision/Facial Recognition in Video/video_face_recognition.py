import tkinter as tk
from tkinter import filedialog
import face_recognition
import cv2

#Set video capture to a file and open it (opens automatically)
cap = cv2.VideoCapture("vtest2.mp4")

def load_photo():
    '''
    Takes a jpg file as input and outputs image encoding
    '''
    root = tk.Tk()
    root.withdraw()
    photo_loc = filedialog.askopenfilename()
    im = face_recognition.load_image_file(photo_loc)
    return face_recognition.face_encodings(im)[0]

def take_photo():
    return print("Taking photo")

class person(object):
    '''
    Each person has a name and photo.
    
    A person's photo can either be taken by a camera snapshot or loaded as file
    The photocoder method encodes the person's photo using face_recognition
    '''    
    def __init__(self, name, photo_source='snapshot'):
        self.name = name
        self.photo_source = photo_source
        if self.photo_source == 'import':
            self.photo = load_photo()
        elif self.photo_source == 'snapshot':
            self.photo = take_photo()
        else:
            print("Warning - Invalid photo_source: this person will have no\
                  photo and so will not be detected. Recommend re-creating \
                  with a valid photo_source, such as 'import' or 'snapshot'.")

# Train algorithm on the face(s) to be recognized
# If there are multiple faces, it will find all of them, so first one is index 0
people = []
im_enc = []
a = person("Myself")  # Edit this line as desired
people.append(a)

face_locations = []
face_encodings = []
face_names = []
framecount = 1

# While the video is open, do the following
while(cap.isOpened()):
    ret, frame = cap.read()

    # Resize frame for faster face recognition (recommended to start with 0.25)
    small_frame = cv2.resize(frame, (0,0), fx=0.2, fy=0.2)

    # Only process the first frame and every nth frame after that
    if framecount==1 or framecount % 5==0:
        # Find all faces and face encodings in current frame
        face_locations = face_recognition.face_locations(small_frame, number_of_times_to_upsample=2)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        face_colors = []
        for face_encoding in face_encodings:
            # At this point all faces are unknown
            name = "Unknown"
            color = (0, 255, 255)  # Yellow for unkown people
            # Check for face match in known faces
            for idx, im in enumerate(im_enc):
                match = face_recognition.compare_faces([im], face_encoding)
                if match[0]:
                    name = people[idx].name
                    color = (2, 255, 0)  # Green for recognized faces

            face_names.append(name)
            face_colors.append(color)

    framecount += 1

    # Display results (if small_frame is 0.25*original size, then multiply by 4 for example)
    for (top, right, bottom, left), name, color in zip(face_locations, face_names, face_colors):
        top *= 5
        right *= 5
        bottom *= 5
        left *= 5

        # Draw box around face  (0, 0, 255 is red)
        cv2.rectangle(frame, (left-10, top-20), (right+10, bottom+20), color, 2)

        # Draw a label with the name below each face (255, 255, 255 is white)
        font =cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left-10, bottom+40), font, 0.8, (255, 255, 255), 1)
        
    # Display the frame
    cv2.imshow("video", frame)
    
cap.release()
cv2.destroyAllWindows()
