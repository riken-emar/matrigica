# -*- coding: utf-8 -*-
"""
 Copyright from: 2021 Aug
 CSRS, RIKEN Institute
 Auther: K.HARA, A.KUROTANI
"""

import os,sys
import tkinter
from PIL import Image, ImageTk
import PIL.Image
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import subprocess

def dirdialog_clicked_DQ():
    iDir = os.path.abspath(os.path.dirname(__file__))
    iDirPath = filedialog.askdirectory(initialdir = iDir)
    entry1.set(iDirPath)

def dirdialog_clicked_MAPE():
    iDir = os.path.abspath(os.path.dirname(__file__))
    iDirPath = filedialog.askdirectory(initialdir = iDir)
    entry2.set(iDirPath)

def dirdialog_clicked_Ani():
    iDir = os.path.abspath(os.path.dirname(__file__))
    iDirPath = filedialog.askdirectory(initialdir = iDir)
    entry3.set(iDirPath)


def dirdialog_clicked_Out():
    iDir = os.path.abspath(os.path.dirname(__file__))
    iDirPath = filedialog.askdirectory(initialdir = iDir)
    entry4.set(iDirPath)


def showAMoRiImage(dirPathOut, img_BaseName):
    pngfile = dirPathOut+"/"+img_BaseName
    if(os.path.exists(pngfile)):
        img = PIL.Image.open(pngfile)
        img.show()
    else:
        print ("no image data")

def showResult(dirPathOut, res_BaseName):
    resfile = dirPathOut+"/"+res_BaseName
    if(os.path.exists(resfile)):
        f = open(resfile, 'r') 
        allStr = f.read()
        f.close

        awin = Tk()
        awin.geometry("540x430+100+200")
        awin.title("Material Rigidity -output panel")
        textFrame = tk.Frame(awin, width=530, height=420, bg="white")
        textFrame.pack()
        #textFrame.grid(row=1,column=1,sticky=W)
        scrollbar = tk.Scrollbar(textFrame)
        scrollbar.pack(side=tk.RIGHT, fill="y")
        textField = tk.Text(textFrame, width=250, height=100, bd=5, relief="groove")
        textField.pack(side=tk.LEFT, padx=(20, 0), pady=(10, 10))
        textField["yscrollcommand"] = scrollbar.set
        textField.insert(tkinter.END, allStr)
    else:
        print ("no result data")


def conductMain():
    text = ""

    dirPath1 = entry1.get()
    dirPath2 = entry2.get()
    dirPath3 = entry3.get()
    dirPathOut = entry4.get()
    if dirPath1:
        text += "Dir path (DQ):\t" + dirPath1 + "\n"
    if dirPath2:
        text += "Dir path (MAPE):\t" + dirPath2 + "\n"
    if dirPath3:
        text += "Dir path (Anisotropy):\t" + dirPath3 + "\n"
    if dirPathOut:
        text += "Dir path (Output):\t" + dirPathOut

    if text:
        subprocess.run("python getMaterialRigidity.py " + dirPath1 +  " " + dirPath2 + " " + dirPath3 + " 9 20 3 "+ dirPathOut +" calc_ "  + " 0.9 0.9 0.9 0 1 1 ")
        #messagebox.showinfo("info", text)
        outList = os.listdir(dirPathOut)
        res_BaseName = ""
        img_BaseName_f = ""
        img_BaseName_t = ""
        img_BaseName_3d = ""
        for fname in outList:
            if("summary_Ratio" in fname):
                #print (fname)
                res_BaseName = fname
            elif("An_Mo_Ri_freq.png" in fname): #
                img_BaseName_f = fname
            elif("An_Mo_Ri_time.png" in fname):
                img_BaseName_t = fname
            elif("An_Mo_Ri_freq-time_3D.png" in fname):
                img_BaseName_3d = fname

        if(os.path.exists(dirPathOut+"/"+res_BaseName)):
            showResult(dirPathOut, res_BaseName)        # show text
        if(os.path.exists(dirPathOut+"/"+img_BaseName_3d)):
            showAMoRiImage(dirPathOut, img_BaseName_3d) # show image (freq-time)
        if(os.path.exists(dirPathOut+"/"+img_BaseName_t)):
            showAMoRiImage(dirPathOut, img_BaseName_t)  # show image (time)
        if(os.path.exists(dirPathOut+"/"+img_BaseName_f)):
            showAMoRiImage(dirPathOut, img_BaseName_f)  # show image (freq)

    else:
        messagebox.showerror("error", "Please select path")




if __name__ == "__main__":
    # root
    root = Tk()
    root.title("Material Rigidity")
    root.geometry("400x250")


    # Select DQ Dir
    frame1 = ttk.Frame(root, padding=10)
    frame1.grid(row=0, column=1, sticky=E)

    IDirLabel = ttk.Label(frame1, text="DQ Dir >>", padding=(5, 2))
    IDirLabel.pack(side=LEFT)

    entry1 = StringVar()
    IDirEntry = ttk.Entry(frame1, textvariable=entry1, width=30)
    IDirEntry.pack(side=LEFT)

    IDirButton = ttk.Button(frame1, text="Dialog", command=dirdialog_clicked_DQ)
    IDirButton.pack(side=LEFT)



    # Select MAPE Dir
    frame2 = ttk.Frame(root, padding=10)
    frame2.grid(row=2, column=1, sticky=E)

    IFileLabel = ttk.Label(frame2, text="MAPE Dir >>", padding=(5, 2))
    IFileLabel.pack(side=LEFT)

    entry2 = StringVar()
    IFileEntry = ttk.Entry(frame2, textvariable=entry2, width=30)
    IFileEntry.pack(side=LEFT)

    IFileButton = ttk.Button(frame2, text="Dialog", command=dirdialog_clicked_MAPE)
    IFileButton.pack(side=LEFT)



    # Select Ani Dir
    frameAni = ttk.Frame(root, padding=10)
    frameAni.grid(row=4, column=1, sticky=E)

    IFileLabel = ttk.Label(frameAni, text="Anisotropy Dir >>", padding=(5, 2))
    IFileLabel.pack(side=LEFT)

    entry3 = StringVar()
    IFileEntry = ttk.Entry(frameAni, textvariable=entry3, width=30)
    IFileEntry.pack(side=LEFT)

    IFileButton = ttk.Button(frameAni, text="Dialog", command=dirdialog_clicked_Ani)
    IFileButton.pack(side=LEFT)


    # Select Output Dir
    frameAni = ttk.Frame(root, padding=10)
    frameAni.grid(row=6, column=1, sticky=E)

    IFileLabel = ttk.Label(frameAni, text="Output Dir >>", padding=(5, 2))
    IFileLabel.pack(side=LEFT)

    entry4 = StringVar()
    IFileEntry = ttk.Entry(frameAni, textvariable=entry4, width=30)
    IFileEntry.pack(side=LEFT)

    IFileButton = ttk.Button(frameAni, text="Dialog", command=dirdialog_clicked_Out)
    IFileButton.pack(side=LEFT)

    # Buttons
    frame3 = ttk.Frame(root, padding=10)
    frame3.grid(row=7,column=1,sticky=W)

    button1 = ttk.Button(frame3, text="Submit", command=conductMain)
    button1.pack(fill = "x", padx=30, side = "left")

    button2 = ttk.Button(frame3, text=("Close"), command=quit)
    button2.pack(fill = "x", padx=30, side = "left")


    root.mainloop()

