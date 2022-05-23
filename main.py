from tkinter import *
from tkinter import filedialog
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import ImageTk, Image

root = Tk()
root.title('Detected Coin Application')
root.geometry("1280x720")
root.configure(background="#A982FE")
root.iconbitmap("icon.ico")

# define label
frame = LabelFrame(root, text='Image selected ', padx=5, pady=5, bg="#96B3FF")
frame2 = LabelFrame(root, text='Coins Detected', padx=5, pady=5, bg="#96B3FF")
frame4 = LabelFrame(root, text='Amount ', padx=5, pady=5, bg="#CCD1F8")
frame.grid(row=1, column=0)
frame.place(x=50, y=100)
frame2.grid(row=1, column=2)
frame2.place(x=705, y=100)
frame4.grid(row=2, column=2)
frame4.place(x=470, y=640)

mycoin, ek = 0, 0

def open():
    global my_image, my_label, path, mycoin, ek
    delete()
    root.filename = filedialog.askopenfilename(initialdir="\coins", title="Select Image",
                                               filetypes=(("all files", "*.*"), ("jpg files", "*.jpg")))
    path = root.filename
    mycoin = 1
    ek = 1
    image_normal = Image.open(root.filename)
    resized = image_normal.resize((500, 500), Image.ANTIALIAS)
    my_image = ImageTk.PhotoImage(resized)
    # putting image to something
    my_label = Label(frame, image=my_image, height=500, width=500)
    my_label.grid(row=0, column=0)

def detect():
    global imgtk
    global my_coin
    global e
    MyImg = cv2.imread(path)
    MyImgPlt = plt.imread(path)
    ImgGray = cv2.cvtColor(MyImg, cv2.COLOR_BGR2GRAY)
    ImgBlur = cv2.medianBlur(ImgGray, 7)
    FindCircles = cv2.HoughCircles(ImgBlur, cv2.HOUGH_GRADIENT, 1, 50, param1=100, param2=50, minRadius=10,
                                   maxRadius=380)
    class_names = ['10DH', '10cent', '10ecent', '1dh', '1ecent', '1euro', '20cent', '20ecent', '2dh', '2ecent', '2euro',
                   '50cent', '50ecent', '5DH', '5ecent']
    MyModel = tf.keras.models.load_model("model")
    ImgCopy = MyImg.copy()
    ImgCopy2 = MyImgPlt.copy()

    Total_eu = 0
    Total_dh = 0
    for Circles in FindCircles[0]:
        CircleX, CircleY, CircleRad = Circles
        CircleX = int(CircleX)
        CircleY = int(CircleY)
        CircleRad = int(CircleRad)
        Mask = np.zeros(MyImgPlt.shape[:2], dtype="uint8")
        cv2.circle(Mask, (CircleX, CircleY), CircleRad, (255, 255, 255), -1)
        Coin = cv2.bitwise_and(MyImgPlt, MyImgPlt, mask=Mask)
        X = CircleX - CircleRad
        Y = CircleY - CircleRad
        MyCoin = Coin[Y:(Y + 2 * CircleRad), X:(X + 2 * CircleRad)]
        MyCoin = tf.image.resize(MyCoin, size=[256, 256])
        MyCoin = MyCoin / 255.
        predict = MyModel.predict(tf.expand_dims(MyCoin, axis=0))
        if len(predict[0]) > 1:
            predict_class = class_names[predict.argmax()]
        else:
            predict_class = class_names[int(tf.round(predict)[0][0])]
        if predict_class == '1ecent':
            Total_eu = Total_eu + 0.01
            ImgCopy2 = cv2.circle(ImgCopy2, (CircleX, CircleY), CircleRad, (0, 255, 0), 5)
            ImgCopy2 = cv2.putText(ImgCopy2, '1 cent', (CircleX - 20, CircleY + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                   (0, 0, 0),
                                   3)
        if predict_class == '2ecent':
            Total_eu = Total_eu + 0.02
            ImgCopy2 = cv2.circle(ImgCopy2, (CircleX, CircleY), CircleRad, (0, 255, 0), 5)
            ImgCopy2 = cv2.putText(ImgCopy2, '2 cent', (CircleX - 20, CircleY + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                   (0, 0, 0),
                                   3)
        if predict_class == '5ecent':
            Total_eu = Total_eu + 0.05
            ImgCopy2 = cv2.circle(ImgCopy2, (CircleX, CircleY), CircleRad, (0, 255, 0), 5)
            ImgCopy2 = cv2.putText(ImgCopy2, '5 cent', (CircleX - 20, CircleY + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                   (0, 0, 0),
                                   3)
        if predict_class == '10ecent':
            Total_eu = Total_eu + 0.10
            ImgCopy2 = cv2.circle(ImgCopy2, (CircleX, CircleY), CircleRad, (0, 255, 0), 5)
            ImgCopy2 = cv2.putText(ImgCopy2, '10 cent', (CircleX - 20, CircleY + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                   (0, 0, 0),
                                   3)
        if predict_class == '20ecent':
            Total_eu = Total_eu + 0.20
            ImgCopy2 = cv2.circle(ImgCopy2, (CircleX, CircleY), CircleRad, (0, 255, 0), 5)
            ImgCopy2 = cv2.putText(ImgCopy2, '20 cent', (CircleX - 20, CircleY + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                   (0, 0, 0),
                                   3)
        if predict_class == '50ecent':
            Total_eu = Total_eu + 0.5
            ImgCopy2 = cv2.circle(ImgCopy2, (CircleX, CircleY), CircleRad, (0, 255, 0), 5)
            ImgCopy2 = cv2.putText(ImgCopy2, '50 cent', (CircleX - 20, CircleY + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                   (0, 0, 0),
                                   3)
        if predict_class == '1euro':
            Total_eu = Total_eu + 1.0
            ImgCopy2 = cv2.circle(ImgCopy2, (CircleX, CircleY), CircleRad, (0, 255, 0), 5)
            ImgCopy2 = cv2.putText(ImgCopy2, '1 euro', (CircleX - 20, CircleY + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                   (0, 0, 0),
                                   3)
        if predict_class == '2euro':
            Total_eu = Total_eu + 2.0
            ImgCopy2 = cv2.circle(ImgCopy2, (CircleX, CircleY), CircleRad, (0, 255, 0), 5)
            ImgCopy2 = cv2.putText(ImgCopy2, '2 euro', (CircleX - 20, CircleY + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                   (0, 0, 0),
                                   3)
        if predict_class == '10cent':
            Total_dh = Total_dh + 0.1
            ImgCopy2 = cv2.circle(ImgCopy2, (CircleX, CircleY), CircleRad, (0, 255, 0), 5)
            ImgCopy2 = cv2.putText(ImgCopy2, '0.1 dh', (CircleX - 20, CircleY + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                   (0, 0, 0), 3)
        if predict_class == '20cent':
            Total_dh = Total_dh + 0.2
            ImgCopy2 = cv2.circle(ImgCopy2, (CircleX, CircleY), CircleRad, (0, 255, 0), 5)
            ImgCopy2 = cv2.putText(ImgCopy2, '0.2', (CircleX - 20, CircleY + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                   (0, 0, 0), 3)
        if predict_class == '50cent':
            Total_dh = Total_dh + 0.5
            ImgCopy2 = cv2.circle(ImgCopy2, (CircleX, CircleY), CircleRad, (0, 255, 0), 5)
            ImgCopy2 = cv2.putText(ImgCopy2, '0.5 dh', (CircleX - 20, CircleY + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                   (0, 0, 0), 3)
        if predict_class == '1dh':
            Total_dh = Total_dh + 1.0
            ImgCopy2 = cv2.circle(ImgCopy2, (CircleX, CircleY), CircleRad, (0, 255, 0), 5)
            ImgCopy2 = cv2.putText(ImgCopy2, '1 dh', (CircleX - 20, CircleY + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                   (0, 0, 0),
                                   3)
        if predict_class == '2dh':
            Total_dh = Total_dh + 2.0
            ImgCopy2 = cv2.circle(ImgCopy2, (CircleX, CircleY), CircleRad, (0, 255, 0), 5)
            ImgCopy2 = cv2.putText(ImgCopy2, '2 dh', (CircleX - 20, CircleY + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                   (0, 0, 0),
                                   3)
        if predict_class == '5DH':
            Total_dh = Total_dh + 5.0
            ImgCopy2 = cv2.circle(ImgCopy2, (CircleX, CircleY), CircleRad, (0, 255, 0), 5)
            ImgCopy2 = cv2.putText(ImgCopy2, '5 dh', (CircleX - 20, CircleY + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                   (0, 0, 0),
                                   3)
        if predict_class == '10DH':
            Total_dh = Total_dh + 10.0
            ImgCopy2 = cv2.circle(ImgCopy2, (CircleX, CircleY), CircleRad, (0, 255, 0), 5)
            ImgCopy2 = cv2.putText(ImgCopy2, '10 dh', (CircleX - 20, CircleY + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                   (0, 0, 0),
                                   3)

    # to import the total
    im = Image.fromarray(ImgCopy2)
    resized = im.resize((500, 500), Image.ANTIALIAS)
    imgtk = ImageTk.PhotoImage(resized)

    # Put it in the display window
    my_coin = Label(frame2, image=imgtk, height=500, width=500)
    my_coin.pack()
    # to print the amount
    e = Entry(frame4, width=50, borderwidth=5)
    e.pack()
    # for default text inside the box
    to_print = "Total ammount "
    if Total_eu != 0 and Total_dh != 0:
        to_print = to_print + str(round(Total_eu,2)) + " Euro, and " + str(Total_dh) + " Dh."
    elif Total_eu != 0:
        to_print = to_print + str(round(Total_eu,2)) + " Euro."
    elif Total_dh != 0:
        to_print = to_print + str(Total_dh) + " Dh."

    e.insert(0, to_print)

def delete():
    global my_coin,mycoin,ek,e
    if mycoin == 1:
        mycoin = 0
        my_coin.destroy()
    if ek == 1:
        ek = 0
        e.destroy()

# the buttons
my_button = Button(root, text="Select Image", padx=10, pady=10, command=open,fg="#FFFFFF", bg="#6495ED")
my_button.place(x=580, y=160)
my_button2 = Button(root, text="Detect Coins", padx=10, pady=10, command=detect,fg="#FFFFFF" , bg="#0000FF")
my_button2.place(x=580, y=260)
# my_button4 = Button(frame1, text="Destroy label", padx=10, pady=10, command=delete).pack()
my_button3 = Button(root, text="        Exit        ", pady=10, padx=10, command=root.quit,fg="#FFFFFF" , bg="#FF0000")
my_button3.place(x=580, y=360)
# execution loop
root.mainloop()
