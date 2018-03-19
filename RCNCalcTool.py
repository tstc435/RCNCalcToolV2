import matplotlib.pyplot as plt
from tkinter import *
import tkinter.filedialog
import numpy as np
import cv2
import tifffile as tiff 
from skimage import data
from skimage import filters
from skimage import exposure
from skimage import img_as_ubyte
from RCNCalctor import RCNCalctor


def main():

	def find_in_grid(frame, idx): #find component by index
		cnt = 0
		for children in frame.children.values():
			info = children.grid_info()
			#note that rows and column numbers are stored as string                                                                         
			#if info['row'] == str(row): #and info['column'] == str(column):
			#	return children
			if cnt == idx:
				return children
			cnt = cnt + 1
		return None

	def xz():
		filename_selected = tkinter.filedialog.askopenfilename()
		pathVar.set(filename_selected)
		#filename = filename_selected
		#if filename != '':
		#	#lb.config(text = "您选择的文件是："+filename);
		#	#ta.insert(END, "您选择的文件是："+filename)
		#	ta.delete(1.0,END)
		#	tool = RCNCalctor()

		#	tool.load(filename, int(binText.get("1.0","end-1c")), int(cutoffText.get("1.0","end-1c")))
		#	stability, stdAvg, chipMedRCN, chipAvgRCN = tool.calc()
		#	stableStr = ('stability:%.2f%%' % (stability*100))
		#	ta.insert(END, stableStr+'\n')
		#	noiseStr = ('random noise:%.2f' % (stdAvg))
		#	ta.insert(END, noiseStr+'\n')
		#	ta.insert(END, 'RCN per chip:\n')
		#	rcnStrs = ['%.2f' % (value) for value in chipMedRCN]
		#	for rcnStr in rcnStrs:
		#		ta.insert(END, rcnStr+'\n')

		#else:
		#	ta.insert(END, "您没有选择任何文件")

	def calc():
		
		if pathVar.get() != '':
			#lb.config(text = "您选择的文件是："+filename);
			#ta.insert(END, "您选择的文件是："+filename)
			#ta = find_in_grid(root, 3, 0)
			ta = find_in_grid(root, 7)
			ta.delete(1.0,END)
			tool = RCNCalctor()
			
			tool.load(pathVar.get(), int(binVar.get()), int(cutoffVar.get()))
			stability, stdAvg, chipMedRCN, chipAvgRCN = tool.calc()
			stableStr = ('stability:%.2f%%' % (stability*100))
			ta.insert(END, stableStr+'\n')
			noiseStr = ('random noise:%.2f' % (stdAvg))
			ta.insert(END, noiseStr+'\n')
			ta.insert(END, 'RCN per chip:\n')
			rcnStrs = ['%.2f' % (value) for value in chipMedRCN]
			for rcnStr in rcnStrs:
				ta.insert(END, rcnStr+'\n')

		else:
			ta.insert(END, "您没有选择任何文件")

	root = Tk()
	#root.geometry("480x360")
	root.title("Noise Measure Tool")

	btn = Button(root,text="选择文件",command=xz).grid(row=0, columnspan=1)
	calcBtn = Button(root,text="开始计算",command=calc).grid(row=0,column=1,columnspan=1)
	binLabel = Label(root, text='binning:').grid(row=1, column=0, sticky=W)

	#binLabel.pack()
	pathVar = StringVar(root, value='', name='pathVar')
	binVar = StringVar(root, value='1')
	binText = Entry(root, textvariable=binVar).grid(row=1, column=1)

	cutoffLabel = Label(root, text='cutoff:').grid(row=1,column=2, sticky=W)

	cutoffVar = StringVar(root, value='32')
	cutoffText = Entry(root, textvariable=cutoffVar).grid(row=1,column=3)


	lb = Label(root,text = '测试结果：').grid(row=2)
	ta = Text(root, width=50, height=10).grid(row=3, columnspan=4)
	


	
	
	root.mainloop()
	

if __name__=="__main__":
	main()
	