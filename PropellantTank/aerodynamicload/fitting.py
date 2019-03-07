# -*- coding: utf-8 -*-
# 荷重計算の際に上に凸、原点を通る曲げモーメントの分布が得られる
# 曲げモーメント分布をノーズを固定端とした片持はり複数の集中荷重による近似をする。
# ここでは6点の集中荷重として曲線を直線群によってフィッティングを行なう。

import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
import warnings

def fitting_6lines(xdata, ydata, x1, x2, x3, x4, x5, x6, x_end,
 				   plot_flag=False, print_flag=False, savefig_flag=False,
				   output_name = u"", title_name = u""):
	# @input
	# xdata : 位置[mm]のデータ列 list or np.array
	# ydata : 曲げモーメント[Nm]のデータ列 list or np.array
	# x1~x6 : 集中荷重を受ける位置、機体オシリ側から位置[mm]
	# x_end : 機体の長さ（オシリからノーズまでの距離）[mm]
	# xxxx_flag : 出力の有無（plot、標準出力文字列、plot保存） True or False
	# xxxx_name : 保存、plotタイトルの名前
	# @output
	# P1 ~ P6 : 位置x1~x6にかかる集中荷重[N] コメントアウトをいじってP9まで対応可
	def func1(x, P1):
		return P1 * x - P1 * x1
	def func2(x, P2):
		return (P1 + P2) * x - (P1 * x1 + P2 * x2)
	def func3(x, P3):
		return (P1 + P2 + P3) * x - (P1 * x1 + P2 * x2 + P3 * x3)
	def func4(x, P4):
		return (P1 + P2 + P3 + P4) * x - (P1 * x1 + P2 * x2 + P3 * x3 + P4 * x4)
	def func5(x, P5):
		return (P1 + P2 + P3 + P4 + P5) * x \
		       - (P1 * x1 + P2 * x2 + P3 * x3 + P4 * x4 + P5 * x5)
	def func6(x, P6):
		return (P1 + P2 + P3 + P4 + P5 + P6) * x \
		       - (P1 * x1 + P2 * x2 + P3 * x3 + P4 * x4 + P5 * x5 + P6 * x6)
	def func7(x, P7):
		return (P1 + P2 + P3 + P4 + P5 + P6 + P7) * x \
		       - (P1 * x1 + P2 * x2 + P3 * x3 + P4 * x4 + P5 * x5 + P6 * x6 + P7 * x7)
	def func8(x, P8):
		return (P1 + P2 + P3 + P4 + P5 + P6 + P7 + P8) * x \
		       - (P1 * x1 + P2 * x2 + P3 * x3 + P4 * x4 + P5 * x5 + P6 * x6 + P7 * x7 + P8 * x8)
	def func9(x, P9):
		return (P1 + P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9) * x \
		       - (P1 * x1 + P2 * x2 + P3 * x3 + P4 * x4 + P5 * x5 + P6 * x6 + P7 * x7 + P8 * x8 + P9 * x9)

	with warnings.catch_warnings():
		warnings.simplefilter("ignore") # 不必要な警告が出るので抑制

		df = pd.DataFrame({"x" : xdata,
						   "BM" : ydata})
		parameter_initial = np.array([0.0])
		P1, pcov = scipy.optimize.curve_fit(func1, df[df.x < x2].x, df[df.x < x2].BM, p0=parameter_initial)
		P2, pcov = scipy.optimize.curve_fit(func2, df[df.x > x2][df.x < x3].x, df[df.x > x2][df.x < x3].BM, p0=parameter_initial)
		P3, pcov = scipy.optimize.curve_fit(func3, df[df.x > x3][df.x < x4].x, df[df.x > x3][df.x < x4].BM, p0=parameter_initial)
		P4, pcov = scipy.optimize.curve_fit(func4, df[df.x > x4][df.x < x5].x, df[df.x > x4][df.x < x5].BM, p0=parameter_initial)
		P5, pcov = scipy.optimize.curve_fit(func5, df[df.x > x5][df.x < x6].x, df[df.x > x5][df.x < x6].BM, p0=parameter_initial)
		P6, pcov = scipy.optimize.curve_fit(func6, df[df.x > x6][df.x < x_end].x, df[df.x > x6][df.x < x_end].BM, p0=parameter_initial)
		# P7, pcov = scipy.optimize.curve_fit(func7, df[df.x > x7][df.x < x8].x, df[df.x > x7][df.x < x8].BM, p0=parameter_initial)
		# P8, pcov = scipy.optimize.curve_fit(func8, df[df.x > x8][df.x < x_end].x, df[df.x > x8][df.x < x_end].BM, p0=parameter_initial)
		# P9, pcov = scipy.optimize.curve_fit(func9, df[df.x > x9][df.x < x_end].x, df[df.x > x9][df.x < x_end].BM, p0=parameter_initial)

		if (plot_flag == True):
			plt.figure()
			plt.figure().subplots_adjust(left=0.15)
			plt.plot(df.x, df.BM, "--", label=u"入力値")
			plt.plot(df[df.x < x2].x, func1(df[df.x < x2].x, P1), label=u"近似 区間12 x1 = %d mm, P1 = %d N" % (x1, P1*1000))
			plt.plot(df[df.x > x2][df.x < x3].x, func2(df[df.x > x2][df.x < x3].x, P2), label=u"近似 区間23 x2 = %d mm, P2 = %d N" % (x2, P2*1000))
			plt.plot(df[df.x > x3][df.x < x4].x, func3(df[df.x > x3][df.x < x4].x, P3), label=u"近似 区間34 x3 = %d mm, P3 = %d N" % (x3, P3*1000))
			plt.plot(df[df.x > x4][df.x < x5].x, func4(df[df.x > x4][df.x < x5].x, P4), label=u"近似 区間45 x4 = %d mm, P4 = %d N" % (x4, P4*1000))
			plt.plot(df[df.x > x5][df.x < x6].x, func5(df[df.x > x5][df.x < x6].x, P5), label=u"近似 区間56 x5 = %d mm, P5 = %d N" % (x5, P5*1000))
			plt.plot(df[df.x > x6][df.x < x_end].x, func6(df[df.x > x6][df.x < x_end].x, P6), label=u"近似 区間6E x6 = %d mm, P6 = %d N" % (x6, P6*1000))
			# plt.plot(df[df.x > x7][df.x < x8].x, func7(df[df.x > x7][df.x < x8].x, P7), label=u"近似 区間78 x7 = %d mm, P7 = %d N" % (x7, P7*1000))
			# plt.plot(df[df.x > x8][df.x < x_end].x, func8(df[df.x > x8][df.x < x_end].x, P8), label=u"近似 区間8E x8 = %d mm, P8 = %d N" % (x8, P8*1000))
			# plt.plot(df[df.x > x9][df.x < x_end].x, func9(df[df.x > x9][df.x < x_end].x, P9), label=u"近似 区間9E x9 = %d mm, P9 = %d N" % (x9, P9*1000))
			plt.axvline(x=x1, color = "k", linestyle="--", alpha = 0.2)
			plt.axvline(x=x2, color = "k", linestyle="--", alpha = 0.2)
			plt.axvline(x=x3, color = "k", linestyle="--", alpha = 0.2)
			plt.axvline(x=x4, color = "k", linestyle="--", alpha = 0.2)
			plt.axvline(x=x5, color = "k", linestyle="--", alpha = 0.2)
			plt.axvline(x=x6, color = "k", linestyle="--", alpha = 0.2)
			# plt.axvline(x=x7, color = "k", linestyle="--", alpha = 0.2)
			# plt.axvline(x=x8, color = "k", linestyle="--", alpha = 0.2)
			# plt.axvline(x=x9, color = "k", linestyle="--", alpha = 0.2)
			plt.axvline(x=x_end, color = "k", linestyle="--", alpha = 0.2)
			
			plt.xlabel(u"STA mm")
			plt.ylabel(u"曲げモーメント Nm")
			plt.title(u"%s 曲げモーメントと近似直線" % (title_name))
			plt.legend(loc="best")
			if (savefig_flag == True):
				plt.savefig(output_name + u"_fitting.png")

	if (print_flag == True):
		print("==== 曲げモーメントの近似のための集中荷重計算の結果 ====")
		print("曲線名 : %s" % (output_name))
		print("位置 x1 = %.1f mm,\t荷重 P1 = %.1f N" % (x1, P1*1000))
		print("位置 x2 = %.1f mm,\t荷重 P2 = %.1f N" % (x2, P2*1000))
		print("位置 x3 = %.1f mm,\t荷重 P3 = %.1f N" % (x3, P3*1000))
		print("位置 x4 = %.1f mm,\t荷重 P4 = %.1f N" % (x4, P4*1000))
		print("位置 x5 = %.1f mm,\t荷重 P5 = %.1f N" % (x5, P5*1000))
		print("位置 x6 = %.1f mm,\t荷重 P6 = %.1f N" % (x6, P6*1000))
		# print("位置 x7 = %.1f mm,\t荷重 P7 = %.1f N" % (x7, P7*1000))
		# print("位置 x8 = %.1f mm,\t荷重 P8 = %.1f N" % (x8, P8*1000))
		# print("位置 x9 = %.1f mm,\t荷重 P9 = %.1f N" % (x9, P9*1000))
		print("ノーズ位置 x_N = %d" % (x_end))

	return P1 * 1000, P2 * 1000, P3 * 1000, P4 * 1000, P5 * 1000, P6 * 1000

if __name__ == '__main__':
	plt.close("all")
	df = pd.read_csv("test.csv", skiprows=1, names=("x","BM"))

	#x1 = 9888
	#x2 = 8932
	#x3 = 6104
	#x4 = 5574
	#x5 = 3496
	#x6 = 1096
	#x_end = 0

	x1 = 0
	x2 = 1096
	x3 = 3496
	x4 = 5574
	x5 = 6104
	x6 = 8932
	x_end = 9888

	fitting_6lines(df.x,df.BM, x1, x2, x3, x4, x5, x6, x_end)
