import tkinter as tk
from tkinter import filedialog,ttk
from PIL import Image, ImageTk
import moviepy.editor as mp
from scipy.io import loadmat
import numpy as np
from scipy.signal import butter, lfilter
import pickle
import os
class GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Continious Motor Imagery Classifier")
        self.geometry("700x780")
        self.configure(bg='#5F9EA0')

        # create ttk style
        style = ttk.Style(self)
        style.configure('TButton', background='#FFE4C4', foreground='#000', font=('Arial', 12))

        # create import button
        self.import_button = ttk.Button(self, text="Import", command=self.import_file)
        self.import_button.pack(pady=7)

        # create label and textbox to display file path
        self.file_path_label = ttk.Label(self, text="File path:",font=('Arial', 10))
        self.file_path_label.pack(pady=5)
        self.file_path_text = tk.Text(self, height=1.5)
        self.file_path_text.pack(pady=5)

        # create play button
        self.play_button = ttk.Button(self, text="Classify", command=self.play_video)
        self.play_button.pack(pady=5)

        # create canvas to display video frames
        self.canvas = tk.Canvas(self, width=400, height=600)
        self.canvas.pack()

        # initialize file path variable
        self.file_path = ""


    def import_file(self):
        # open file dialog to select a video file
        self.file_path = filedialog.askopenfilename(filetypes=[("Dataset files", "*.mat")])
        
        self.file_path_text.delete("1.0", "end")
        self.file_path_text.insert("end", self.file_path)

    def play_video(self):
        if self.file_path:
            # load video clip
            def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    
                nyq = 0.5 * fs
                low = lowcut / nyq
                high = highcut / nyq
                b, a = butter(order, [low, high], btype='band')
                #b, a = butter_bandpass(lowcut, highcut, fs, order=order)
                y = lfilter(b, a, data)
                return y

        filename = self.file_path#'BCICIV_calib_ds1d.mat'
        lowcut = 4
        highcut =35
        fs= 100
        order = 4

        matlabfile = loadmat(filename,struct_as_record=True, squeeze_me=False)            # loading matfile
        data = matlabfile['cnt']         

        # print(y.shape)

        i=0

        base = os.path.dirname(__file__)+"/saved_models/"
        filename = base +"mi-nomi_CSP.pkl"
        csp_1 = pickle.load(open(filename, 'rb'))
        filename = base +"which- mi_CSP.pkl"
        csp_2 = pickle.load(open(filename, 'rb'))

        filename =base + "mi-nomi_kbest.pkl"
        kbest_1 = pickle.load(open(filename, 'rb'))
        filename = base +"which- mi_kbest.pkl"
        kbest_2 = pickle.load(open(filename, 'rb'))

        filename = base +"mi1.0.sav"
        cls_1 = pickle.load(open(filename, 'rb'))

        filename = base +"which- mi0.92.sav"
        cls_2 = pickle.load(open(filename, 'rb'))

        while(i+400<=190400):                             # for all remeaining trails extract the respective sample values 
            
            print(i)
            x_MI=data[i:i+400:1]  # Taking the values from the traul position to the next 400 values
            
            x_MI= x_MI.transpose()
            x_MI = x_MI[np.newaxis,:]  
            
            "applying bandpass"
            x_MI= butter_bandpass_filter(x_MI, lowcut, highcut, fs, order)    

            """ Feature engineering"""
            
            #print("csp")
            # print(x_MI.shape)
            ft_train= csp_1.transform(x_MI)
            # print(ft_train.shape)
            ft_train= kbest_1.transform(ft_train)
            # print(ft_train.shape)
            
            ft_train = ft_train.reshape(1, -1)
            res1 = cls_1.predict(ft_train)
            #print(f"First Classifier result {res1} ")

            if res1==1.0:
                i+=200
                continue
            #print("Did not continue")
            ft_train= csp_2.transform(x_MI)
            ft_train= kbest_2.transform(ft_train)
            # print(ft_train.shape)
            
            ft_train = ft_train.reshape(1, -1)
            #print(ft_train.shape)
            res2 = cls_2.predict(ft_train)
            print(f"Second Classifier result {res2[0]} ")
            base = os.path.dirname(__file__)+"/videos/"
            if res2[0] == 1:
                vid_filename = base + "waving_left.mkv"
            else:
                vid_filename = base + "waving_right_hand.mkv"

            video_clip = mp.VideoFileClip(vid_filename)

            # loop over video frames and display in canvas
            for frame in video_clip.iter_frames():
                # convert frame to PIL Image
                image = Image.fromarray(frame)

                # resize image to fit canvas
                image = image.resize((400, 600))

                # convert image to PhotoImage and display in canvas
                photo_image = ImageTk.PhotoImage(image)
                self.canvas.create_image(0, 0, image=photo_image, anchor=tk.NW)
                self.update()

            # release video clip resources
            video_clip.reader.close()
            #video_clip.audio.reader.close_proc()
            i+=800

        else:
            print("Please import a video file first")

# create GUI object and run main loop
gui = GUI()
gui.mainloop()