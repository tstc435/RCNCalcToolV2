import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import tifffile as tiff
from skimage import data
from skimage import filters
from skimage import exposure
from skimage import img_as_ubyte
import datetime
import time
#the major class to do the calculation of noise
class RCNCalctor(object):
	imgNum = 0
	imgWidth = 0
	imgHeight = 0
	src_path = ''
	cutoffx = 0
	binx = 1
	images = []
	channelNumPerChip = 256
	roiWidth = channelNumPerChip
	roiHeight = channelNumPerChip
	numOfColRoi = imgWidth / channelNumPerChip 

	def __init__(self, *args, **kwargs):
		return super().__init__(*args, **kwargs)

	def load(self, src_path, binx, cutoffx):
		self.src_path = src_path
		self.binx = binx
		self.cutoffx = cutoffx
		self.images=tiff.imread(src_path)
		self.imgNum = self.images.shape[0]
		self.imgWidth = self.images.shape[1]
		self.imgHeight = self.images.shape[2]
		self.roiWidth = self.roiWidth // binx
		self.roiHeight = self.roiHeight // binx
		self.numOfColRoi = int(self.imgWidth / self.channelNumPerChip * binx)
		print(self.images.shape)
		print('read image successfully!')

	def testNPDimension(self, src_path):
		img0 = tiff.imread(src_path)
		print(img0.shape)
		roiImg = img0[0:32, 1:65]
		print(roiImg.shape)

	def calc(self):
		if self.imgNum <= 1:
			raise AssertionError("image number should larger than 1!")
		self.numOfColRoi = int(self.imgWidth / self.roiWidth)
		darkAvgs = np.zeros((self.imgNum, 1), dtype=np.float64);
		avgImg = np.zeros((self.imgHeight, self.imgWidth), dtype=np.float64)
		stdImg = np.zeros((self.imgHeight, self.imgWidth), dtype=np.float64)
		#calc average image
		for idx in range(self.imgNum):
			darkAvgs[idx, 0] = cv2.mean(self.images[idx, :, :])[0]
			avgImg = avgImg + np.array(self.images[idx, :,:], dtype=np.float)
			stdImg = stdImg + np.array(self.images[idx, :,:], dtype=np.float) * np.array(self.images[idx, :,:], dtype=np.float)
		avgImg = avgImg / self.imgNum
		stdImg = (stdImg - avgImg * avgImg * self.imgNum)/(self.imgNum - 1)
		stdImg = np.sqrt(stdImg)
		stdAvg = cv2.mean(stdImg)[0]
		stdAvgPerChip = np.zeros((self.numOfColRoi, 1), dtype=np.float)
		for i in range(self.numOfColRoi):
			roiImg = stdImg[:, i*self.roiWidth:(i+1)*self.roiWidth]
			stdAvgPerChip[i] = cv2.mean(roiImg)[0]
		
		minValue, maxValue, minLoc, maxLoc = cv2.minMaxLoc(darkAvgs)
		meanValue = np.mean(darkAvgs, 0)
		#print('minValue: {0}, maxValue:{1}, minLoc: {2}, maxLoc: {3}, meanValue: {4}'.format(minValue, maxValue, minLoc, maxLoc, meanValue))
		stability = abs(maxValue - minValue)/meanValue;
		print('for image sequences {0}, binSize: {1}, cutoff:{2}\n'.format(self.src_path, self.binx, self.cutoffx))
		print('stability:%.2f%%' % (stability*100))
		print('average random noise: %.2f' %(stdAvg))
		strStdAvgPerChips = ['%.2f' % (value) for value in stdAvgPerChip]
		print('random noise per chip:')
		print(' '.join(strStdAvgPerChips))
		#define roi for each chip
		offY = []
		if self.roiHeight > self.imgHeight or self.roiWidth > self.imgWidth:
			raise Exception('roi size ({0}x{1}) is larger than image size ({0}x{1})'.format(self.roiWidth, self.roiHeight, self.imgWidth, self.imgHeight))
		if self.roiHeight < self.imgHeight / 2: 
			offY = [int(self.imgHeight/4 - self.roiHeight /2), int(self.imgHeight*3/4 - self.roiHeight/2)]
		else:
			offY = [int(self.imgHeight - self.roiHeight)/2]
		avgRCN = np.zeros((len(offY), self.numOfColRoi), dtype=np.float64)
		avgRN = np.zeros((len(offY), self.numOfColRoi), dtype=np.float64)
		avgCN = np.zeros((len(offY), self.numOfColRoi), dtype=np.float64)

		chipAvgRCN = np.zeros((1, self.numOfColRoi), dtype=np.float64)
		chipAvgRN = np.zeros((1, self.numOfColRoi), dtype=np.float64)
		chipAvgCN = np.zeros((1, self.numOfColRoi), dtype=np.float64)

		chipMedRCN = np.zeros((1, self.numOfColRoi), dtype=np.float64)
		chipMedRN = np.zeros((1, self.numOfColRoi), dtype=np.float64)
		chipMedCN = np.zeros((1, self.numOfColRoi), dtype=np.float64)
		 
		medRCN = np.zeros((len(offY), self.numOfColRoi), dtype=np.float64)
		medRN = np.zeros((len(offY), self.numOfColRoi), dtype=np.float64)
		medCN = np.zeros((len(offY), self.numOfColRoi), dtype=np.float64)

		RCN2D = np.zeros((self.imgNum, len(offY), self.numOfColRoi), dtype=np.float64)
		RN2D = np.zeros((self.imgNum, len(offY), self.numOfColRoi), dtype=np.float64)
		CN2D = np.zeros((self.imgNum, len(offY), self.numOfColRoi), dtype=np.float64)

		for idx in range(self.imgNum):
			noiseImg = np.array(self.images[idx,:,:], dtype=np.float) - avgImg
			noiseImg[0:self.cutoffx, :] = noiseImg[self.cutoffx:2*self.cutoffx, :]
			noiseImg[(self.imgHeight-self.cutoffx):self.imgHeight, :] = noiseImg[(self.imgHeight-2*self.cutoffx):(self.imgHeight-self.cutoffx), :]
			noiseImg[:, 0:self.cutoffx] = noiseImg[:, self.cutoffx:2*self.cutoffx]
			noiseImg[:, (self.imgWidth-self.cutoffx):self.imgWidth] = noiseImg[:, (self.imgWidth-2*self.cutoffx):(self.imgWidth-self.cutoffx)]
			
			for h in range(len(offY)):
				for w in range(self.numOfColRoi):
					roiImg = noiseImg[offY[h]:(offY[h]+self.roiHeight), w*self.roiWidth:(w*self.roiWidth+self.roiWidth)];
					rowAvgs = np.mean(roiImg, 1);
					rmed = np.median(rowAvgs);
					tmpRNInRow = np.std(rowAvgs);
					rowIdxs = [i for i in range(len(rowAvgs)) if abs(rowAvgs[i] - rmed) >= 3.5*tmpRNInRow]
					rowAvgs[rowIdxs] = rmed
					RNInRow = np.std(rowAvgs)

					colAvgs = np.mean(roiImg, 0)
					cmed = np.median(colAvgs)
					tmpCNInCol = np.std(colAvgs)
					colIdxs = [i for i in range(len(colAvgs)) if abs(colAvgs[i] - cmed) >= 3.5*tmpCNInCol]
					colAvgs[colIdxs] = cmed
					RNInCol = np.std(colAvgs)

					RN2D[idx][h][w] = RNInRow
					CN2D[idx][h][w] = RNInCol
					RCN2D[idx][h][w] = RNInRow / max(RNInCol, 1e-6)

		avgRCN = np.mean(RCN2D, 0)
		avgRN = np.mean(RN2D, 0)
		avgCN = np.mean(CN2D, 0)
		chipAvgRCN = np.mean(avgRCN, 0)
		chipAvgRN = np.mean(avgRN, 0)
		chipAvgCN = np.mean(avgCN, 0)
		medRCN = np.median(RCN2D, 0)
		medRN = np.median(RN2D, 0)
		medCN = np.median(CN2D, 0)
		chipMedRCN = np.mean(medRCN, 0)
		chipMedRN = np.mean(medRN, 0)
		chipMedCN = np.mean(medCN, 0)
		
		logDir = 'logs'
		if not os.path.exists(logDir):
			os.makedirs(logDir)
		dt = datetime.datetime.now()
		strTime = dt.strftime('%Y-%m-%d')
		fid = open(os.path.join(logDir, strTime+'.log'), 'a')
		fid.write('for image sequences {0}, binSize: {1}, cutoff:{2}\n'.format(self.src_path, self.binx, self.cutoffx))

		fid.write('stablility: %.2f%%\n' % (stability*100))

		fid.write('average random noise: %.2f\n' %(stdAvg))
		fid.write('random noise per chip:\n')
		fid.write(' '.join(strStdAvgPerChips)+'\n')

		fid.write('average of row noise per chip:\n')
		strAvgRNs = (['%.2f' % (value) for value in chipAvgRCN])
		strAvgRN = ' '.join(strAvgRNs)
		fid.write(strAvgRN+'\n')

		fid.write('average of column noise per chip:\n')
		strAvgCN = ['%.2f' % (value) for value in chipAvgCN]
		fid.write(' '.join(strAvgCN)+'\n')

		fid.write('median of row noise per chip:\n')
		strMedRN = ['%.2f' % (value) for value in chipMedRN]
		fid.write(' '.join(strMedRN)+'\n')

		fid.write('median of column noise per chip:\n')
		strMedCN = ['%.2f' % (value) for value in chipMedCN]
		fid.write(' '.join(strMedCN)+'\n')

		fid.write('average of RCN per chip:\n')
		strAvgRCN = ['%.2f' % (value) for value in chipAvgRCN]
		fid.write(' '.join(strAvgRCN)+'\n')

		fid.write('median of RCN per chip:\n')
		
		strMedRCN = ['%.2f' % (value) for value in chipMedRCN]
		fid.write(' '.join(strMedRCN)+'\n')

		fid.write('\n')
		fid.close()

		
		print('median of RCN per chip:')
		print(' '.join(strMedRCN))

		return stability, stdAvg, chipMedRCN, chipAvgRCN


if __name__=="__main__":
	calctor = RCNCalctor()
	calctor.load("E:\\Workspace\\matlabWorkspace\\RCN\\RCN\\1536x1536_Nref=32_side=dual.tif", 1, 32)
	#calctor.testNPDimension('E:\\Workspace\\pyWorkspace\\PyVSProjects\\imgproc\\RCNCalcTool\\images\\dark-1.tif')
	calctor.calc();




