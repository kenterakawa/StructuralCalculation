# -*- coding: utf-8 -*-
# MOMOの大気中飛行時の荷重計算
# 軸力と曲げモーメントを計算する
# 参考文献：液体ロケットの構造システム設計
from __future__ import print_function

import sys
import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
import time as tm
from fitting import fitting_6lines

plt.close("all")
process_start = tm.time()

class Component:
	def __init__(self, length, weight_dry, prop_init, prop_end, burntime, press=0.0):
		self.length     = length
		self.weight_dry = weight_dry
		self.prop_init  = prop_init
		self.prop_end   = prop_end
		self.burntime   = burntime
		self.prop       = np.linspace(prop_init, prop_end, burntime)
		self.weight_wet = np.linspace(weight_dry, weight_dry, burntime) + self.prop
		self.press      = press # NOT USED NOW

		# イナートセット
		self.density = self.weight_dry/self.length*np.ones([burntime,self.length])

		# 推進剤セット
		if self.prop_init != 0.0:
			density_prop_init = prop_init/self.length # 初期推進剤線密度
			for time in range(self.burntime):
				num_block_prop = int(self.prop[time]/density_prop_init+1e-8)
				array_current_prop = density_prop_init*np.ones(self.length)
				array_current_prop[0:self.length-num_block_prop]=0
				self.density[time] = self.density[time] + array_current_prop
				amari = self.weight_wet[time]-sum(self.density[time])
				if amari < -1e-8:
					print("ERROR: NEGATIVE MASS")
					print(time,self.burntime,num_block_prop,self.length,array_current_prop,amari)
					exit(1)

				if abs(amari)>1e-8:
				    self.density[time][-num_block_prop-1] = self.density[time][-num_block_prop-1] + amari
					
		# 整合性確認
		for time in range(self.burntime):
			
			if abs(sum(self.density[time])-self.weight_wet[time])>1e-8:
				print(u"ERROR: 質量と密度積分値が一致しません。")
				print(self.weight_wet[time]-sum(self.density[time]),density_prop_init)
				exit(1)

	def show(self):
		print(u"[%d mm,\t%d kg,\t%d kg,\t%d kg]" %(self.length, self.weight_dry, self.prop_init, self.prop_end))

class Rocket:
	def __init__(self, burntime, diameter):
		# 以下はFinalizeが必要
		# dmdx
		self.burntime   = burntime
		self.length     = 0
		self.weight_dry = 0
		self.weight_wet = 0
		self.prop    = np.zeros(self.burntime)
		self.dmdx    = np.zeros([self.burntime, self.length])
		self.mass    = np.zeros(self.burntime)
		self.inertia = np.zeros(self.burntime)
		self.x_CG    = np.zeros(self.burntime)
		self.x_CP    = np.zeros(self.burntime)
		self.x       = np.arange(self.length)
		#self.inertia_distribution = np.zeros([self.burntime, self.length+1])
		self.diameter= diameter
		self.area = 1.0/4 * np.pi * diameter**2 # 面積 m2
		self.divid = [0]

	def add_component(self, component):
		# 部品を加えてロケットのパラメータを更新
		length_start = self.length

		self.length     = self.length     + component.length
		self.weight_dry = self.weight_dry + component.weight_dry
		self.prop       = self.prop       + component.prop
		self.weight_wet = self.weight_wet + component.weight_wet

		self.divid.append(self.length)

		self.dmdx = np.concatenate((self.dmdx,np.zeros([self.burntime, component.length])),axis=1)
		#self.inertia_distribution = np.zeros([self.burntime, self.length])
		for time in range(self.burntime):
			self.dmdx[time][length_start:self.length] = component.density[time]

	def set_C_A(self, C_A):
		# 部品を加えた後に軸力係数をセット
		# C_A:全機の軸力係数、今後は入力を分布にできるように関数だけ用意
		self.C_A = C_A
		self.dC_Adx = np.zeros([self.burntime, self.length+1])
		for time in range(self.burntime):
			self.dC_Adx[time][0] = C_A

	def set_C_N(self, C_N, nose, fin, engine):
		# 部品を加えた後に法線力係数をセット
		# C_N：全機の法線力係数、今後は入力を分布にできるように関数を用意
		self.dC_Ndx = np.zeros([self.burntime, self.length+1])
		pos_nose = int(nose.length/2)
		pos_fin  = int(self.length - (engine.length + fin.length / 2))
		for time in range(self.burntime):
			self.C_N_nose = C_N * (self.x_CP[time] - pos_fin ) / (pos_nose - pos_fin)
			self.C_N_tail = C_N * (self.x_CP[time] - pos_nose) / (pos_fin - pos_nose)
			self.dC_Ndx[time][pos_nose] = self.C_N_nose
			self.dC_Ndx[time][pos_fin]  = self.C_N_tail

	def set_x_CP(self, x_CP):
		self.x_CP = x_CP*np.ones(self.burntime)

	def finalize(self):
		# dmdxの境界値処理(機体後端を0)
		self.x = np.arange(rocket.length+1)
		self.dmdx = np.concatenate((self.dmdx,np.zeros([self.burntime, 1])),axis=1)

		# 慣性諸元計算
		for time in range(self.burntime):
			self.mass[time]    = np.sum(self.dmdx[time])
			self.x_CG[time]    = np.sum(self.dmdx[time] * self.x)
			self.inertia[time] = np.sum(self.dmdx[time] * self.x ** 2)

			self.x_CG[time]    = self.x_CG[time] / self.mass[time] # [mm]
			self.inertia[time] = self.inertia[time] - self.mass[time] * self.x_CG[time] ** 2 # [kg*mm2]


def calc_axial_load(rocket,thrust_a,q_a,rating_time):
	drag     = np.zeros([rocket.burntime, rocket.length+1])
	force_A  = np.zeros([rocket.burntime, rocket.length+1])
	X_dotdot = np.zeros(rocket.burntime)
	#for time in range(burntime): # for all the duration.
	for time in rating_time:
		X_dotdot[time] = (thrust_a[time] - rocket.area * q_a[time] * rocket.C_A) / rocket.mass[time] # [m/s2]
		drag[time] = rocket.area * q_a[time] * np.cumsum(rocket.dC_Adx[time])
		force_A[time] = - drag[time] - X_dotdot[time] * np.cumsum(rocket.dmdx[time])
	return [X_dotdot,drag,force_A]

def calc_bending_moment(rocket,thrust_a,q_a,rating_time):
	M_a  = np.zeros([rocket.burntime, rocket.length+1])
	M_a1 = np.zeros([rocket.burntime, rocket.length+1])
	M_a2 = np.zeros([rocket.burntime, rocket.length+1])
	M_d  = np.zeros([rocket.burntime, rocket.length+1])
	M_d1 = np.zeros([rocket.burntime, rocket.length+1])
	M_d2 = np.zeros([rocket.burntime, rocket.length+1])
	M1   = np.zeros([rocket.burntime, rocket.length+1])
	M2   = np.zeros([rocket.burntime, rocket.length+1])
	Z_dotdot_d  = np.zeros(rocket.burntime)
	Z_dotdot_a  = np.zeros(rocket.burntime)
	Omega_dot_d = np.zeros(rocket.burntime)
	Omega_dot_a = np.zeros(rocket.burntime)
	#for time in range(burntime): # for all the duration 
	for time in rating_time:
		#if(time%10==0):print(u"曲げモーメント計算：燃焼時間 %d 秒"% (time)) # for all the duration
		## Symbol:
		# *_d: Moment by Gimbal
		# *_a: Moment by Air Force

		# gimbal
		Z_dotdot_d[time]  = T_g_a[time] / rocket.mass[time] # [m/s2]
		Omega_dot_d[time] = T_g_a[time] * rocket.x_CG[time] / rocket.inertia[time] # [m/mm/s2]
		for pos in range(rocket.length+1):
			M_d1[time][pos] =   T_g_a[time] * pos * 1e-3 # [Nm]
			M_d2[time][pos] = - np.sum(\
					      (Z_dotdot_d[time] + Omega_dot_d[time]*(rocket.x_CG[time]- rocket.x[:pos]))\
					    * rocket.dmdx[time][:pos]\
					    * (pos - rocket.x[:pos])) * 1e-3 # [Nm]

		# air force
		Z_dotdot_a[time]  = rocket.area * q_a[time] * np.sum(rocket.dC_Ndx[time]) / rocket.mass[time] # [m/s2]
		Omega_dot_a[time] = rocket.area * q_a[time] * np.sum(rocket.dC_Ndx[time]*(rocket.x_CG[time]-rocket.x)) / rocket.inertia[time] # [m/mm/s2]

		for pos in range(rocket.length+1):
			M_a1[time][pos] = + rocket.area * q_a[time] * np.sum(rocket.dC_Ndx[time][:pos] * (pos - rocket.x[:pos])) * 1e-3 # [Nm]
			M_a2[time][pos] = - np.sum(\
					      (Z_dotdot_a[time] + Omega_dot_a[time]*(rocket.x_CG[time]- rocket.x[:pos]))\
					    * rocket.dmdx[time][:pos]\
					    * (pos - rocket.x[:pos])) * 1e-3 # [Nm]
	M_a = M_a1 + M_a2
	M_d = M_d1 + M_d2
	M1 = M_a + M_d
	M2 = M_a - M_d
	M_max = np.maximum (abs(M1),abs(M2))
	
	return [M_a, M_d, M1, M2, M_max, Z_dotdot_d, Z_dotdot_a, Omega_dot_d, Omega_dot_a]

def calc_equivalent_axial_force(force_A,M_max,diameter):
	F_eq_comp = force_A-M_max*4/diameter
	F_eq_tens = force_A+M_max*4/diameter

	return [F_eq_comp, F_eq_tens]

def calc_rating_load(load,divid,rating_time,sign):
	load_rating = []
	for i in range(len(divid)-1):
		i_start = divid[i]
		i_end   = divid[i+1]+1
		val = 0
		for j,time in enumerate(rating_time):
			cur = max(load[time][i_start:i_end]*sign)
			val = max(val,cur)
		load_rating.append(val*sign)
	return load_rating

if __name__ == "__main__":
	print(u"****** IST 荷重計算プログラム ******")
	g0 = 9.80665

	#諸元入力ゾーン==========================================================================================================================


	# 軌道情報 MOMO dynamics, ZERO Phase6E軌道設計 参考
	burntime = 163 # 燃焼時間 秒
	rating_label = ["LiftOff"	, "MaxQ"	, "MaxDrag"	, "MECO"	]
	rating_time  = [0		, 55		, 60		, 162		] # 評定となる時刻 秒
	thrust       = [500400		, 570600		, 570600		, 571000		] # 推力 N
	q            = [6.442E+00	, 2.6755E+04	, 2.6148E+04	, 1.686E+03	] # 動圧 Pa
	max_gimbal_angle = 8 * np.pi/180 # 最大舵角[rad]
	T_g = [v*np.sin(max_gimbal_angle) for v in thrust]# ジンバルによる横推力 N

	# 空力諸元 源泉:MOMO初号機軌道設計10_Missile DATCOMでの結果.pdf
	dia  = 2.0 # 機体直径 m
	C_A  =    0.746 # 軸力係数 ND
	C_N  =      2.1 # 法線力係数 ND
	x_CP =     6600 # ノーズからの風圧中心位置 mm
	
	savefig_flag = True # 出力を保存するかどうか
	savepdf_flag = True # PDF出力するかどうか
	save_name = u"ZERO_Ph6F_NP_Case1"
	
	if(savepdf_flag):pdf = PdfPages(save_name + u"_plot.pdf")
	sys.stdout = sys.__stdout__
	
	print(u"コンポーネント設定開始")
	# === コンポーネント ====
	# comp        = Component(length_mm, weight_kg, prop_init_kg, prop_end_kg, burntime_sec, press_MPa)
	nose          = Component(3000, 235, 0, 0, burntime, 0.0)
	tank_2nd_LOx	= Component(500, 142, 3967, 3967, burntime, 0.5)
	tank_2nd_inter = Component(2180, 364, 0, 0, burntime, 0.0)
	tank_2nd_fuel = Component(900, 80, 1853, 1853, burntime, 0.5)
	tank_interstage = Component(3000, 300, 0, 0, burntime, 0.0)
	tank_1st_LOx       = Component(5400, 686, 19477, 109, burntime, 0.5)
	tank_1st_inter    = Component(1500, 2457, 0, 0, burntime, 0.0)
	tank_1st_fuel      = Component(3800, 517, 10223, 97, burntime, 0.5)
	fin          = Component(1000, 200, 0, 0, burntime, 0.0)
	engine           = Component(1000, 1080, 0, 0, burntime, 0.0)
	
	# === 曲げモーメントの曲線フィッティング ===
	fitting_flag = False # 関数フィッティングするかどうか
	## x1~6 : 集中荷重を受ける機体頭からの距離 mm
	#x1 = 0
	#x2 = 1096
	#x3 = 3497
	#x4 = 5605
	#x5 = 6105
	#x6 = 8905
	#x_end = 9884
	# ==== 入力ここまで =====================================================================================================================
	
	# === コンポーネントをRocketクラスにAdd ====
	rocket = Rocket(burntime, dia)
	rocket.add_component(nose)
	rocket.add_component(tank_2nd_LOx)
	rocket.add_component(tank_2nd_inter)
	rocket.add_component(tank_2nd_fuel)
	rocket.add_component(tank_interstage)
	rocket.add_component(tank_1st_LOx)
	rocket.add_component(tank_1st_inter)
	rocket.add_component(tank_1st_fuel)
	rocket.add_component(fin)
	rocket.add_component(engine)
	
	rocket.set_x_CP(x_CP)
	rocket.set_C_A(C_A)
	rocket.set_C_N(C_N,nose,fin,engine)
	
	rocket.finalize()
	
	print(u"コンポーネント設定終了")
	print(u"入力値出力開始")
	# ==== 補間 ====
	time_a = np.linspace(0,burntime, burntime+1)
	T_g_f    = interp1d(rating_time, T_g,    fill_value="extrapolate")
	thrust_f = interp1d(rating_time, thrust, fill_value="extrapolate")
	q_f      = interp1d(rating_time, q,      fill_value="extrapolate")
	
	T_g_a    = T_g_f(time_a)
	thrust_a = thrust_f(time_a)
	q_a      = q_f(time_a)
	
	#  ==== 重量分布のプロット ====
	plt.figure(1)
	for (i, time) in enumerate(rating_time):
		plt.plot(range(rocket.length+1),rocket.dmdx[time], label= "%s" % (rating_label[i]))
	for j in rocket.divid:
		plt.axvline(x=j, color = "k", linestyle="--", alpha = 0.1)
	plt.xlabel("STA mm")
	plt.ylabel("mass distribution kg/mm")
	plt.title("mass distribution")
	plt.legend(loc="best")
	if(savefig_flag):plt.savefig(save_name + "_load_calculation_mass distribution.png")
	if(savepdf_flag):pdf.savefig()
	
	# ==== 軸力係数＆法線力係数のプロット =====
	time_force = 0
	plt.figure(2)
	plt.plot(range(rocket.length+1),rocket.dC_Adx[time_force], linewidth=5)
	for j in rocket.divid:
		plt.axvline(x=j, color = "k", linestyle="--", alpha = 0.2)
	plt.xlabel("STA [mm]")
	plt.ylabel("dC_A/dx [kg/mm]")
	plt.xlim([-100, rocket.length+100])
	plt.title(r"$\frac{dC_A}{dx}$")
	if(savefig_flag):plt.savefig(save_name + "_load_calculation_dCAdx.png")
	if(savepdf_flag):pdf.savefig()
	
	plt.figure(3)
	plt.plot(range(rocket.length+1),rocket.dC_Ndx[time_force], linewidth=5)
	for j in rocket.divid:
		plt.axvline(x=j, color = "k", linestyle="--", alpha = 0.2)
	plt.xlabel("STA [mm]")
	plt.ylabel("dC_N/dx [kg/mm]")
	plt.xlim([-100, rocket.length+100])
	plt.title(r"$\frac{dC_N}{dx}$")
	if(savefig_flag):plt.savefig(save_name + "_load_calculation_dCNdx.png")
	if(savepdf_flag):pdf.savefig()
	print(u"入力値出力終了")
	
	# ==== 軸力 ====
	print(u"軸力計算中...")
	[X_dotdot,drag,force_A] = calc_axial_load(rocket,thrust_a,q_a,rating_time)
	print(u"軸力計算終了")

	print(u"軸力出力開始")
	plt.figure(4)
	X_dotdot_a =[]
	for (i, time) in enumerate(rating_time):
		plt.plot(range(rocket.length+1),force_A[time], label = "%s" % (rating_label[i]))
		X_dotdot_a.append(X_dotdot[time])
	for j in rocket.divid:
		plt.axvline(x=j, color = "k", linestyle="--", alpha = 0.2)
	plt.xlabel("STA mm")
	plt.ylabel("Axial load N")
	plt.ylim(ymin=-18000)
	plt.title("Axial load")
	plt.legend(loc="best")
	if(savefig_flag):plt.savefig(save_name + "_load_calculation_axial_load.png")
	if(savepdf_flag):pdf.savefig()
	print(u"軸力出力終了")
	
	# ==== 曲げモーメント ====
	print(u"曲げモーメント計算中...")
	[M_a, M_d, M1, M2, M_max, Z_dotdot_d, Z_dotdot_a, Omega_dot_d, Omega_dot_a] = calc_bending_moment(rocket,thrust_a,q_a,rating_time)
	print(u"曲げモーメント計算終了")
	
	# ==== 曲げモーメントのPLOT ====
	print(u"曲げモーメント出力開始")
	Z_dotdot_d_a  = []
	Z_dotdot_a_a  = []
	Omega_dot_d_a = []
	Omega_dot_a_a = []
	for (i, time) in enumerate(rating_time):
		Z_dotdot_d_a.append(Z_dotdot_d[time])
		Z_dotdot_a_a.append(Z_dotdot_a[time])

		# 集約表作成
		plt.figure(5)
		plt.plot(range(rocket.length+1),M1[time], label="%s" % (rating_label[i]))
		plt.figure(6)
		plt.plot(range(rocket.length+1),M2[time], label="%s" % (rating_label[i]))
		plt.figure(7)
		plt.plot(range(rocket.length+1),M_max[time], label="%s" % (rating_label[i]))

		# 曲げモーメント内訳
		plt.figure()
		#plt.plot(range(rocket.length+1),M_a1[time], label = "Air Force")
		#plt.plot(range(rocket.length+1),M_a2[time], label = "Air Force Inertia")
		#plt.plot(range(rocket.length+1),M_d1[time], label = "Gimbal")
		#plt.plot(range(rocket.length+1),M_d2[time], label = "Gimbal Inertia")
		plt.plot(range(rocket.length+1),M_a[time],  label = "Air Force")
		plt.plot(range(rocket.length+1),M_d[time],  label = "Gimbal")
		plt.axhline(y=0, color = "k", linestyle="--", alpha = 0.2)
		for j in rocket.divid:
			plt.axvline(x=j, color = "k", linestyle="--", alpha = 0.2)
		plt.legend(loc = "best")
		plt.xlabel("STA mm")
		plt.ylabel("Bending Moment Nm")
		plt.title("%s Breakdown of BMD" % (rating_label[i]))
		if(savefig_flag):plt.savefig(save_name + "_load_calculation_bending_moment_%s.png" % (rating_label[i]))
		if(savepdf_flag):pdf.savefig()
	
		if(fitting_flag): # 曲げモーメント曲線のフィッティング
			fitting_6lines(rocket.x, M1[time], x1, x2, x3, x4, x5, x6, x_end,True, True, True,
			 			   "%s_M++_%s" % (save_name, rating_label[i]), "%s_M++_%s" % (save_name, rating_label[i]))
			if(savepdf_flag):pdf.savefig()
			fitting_6lines(rocket.x, M2[time], x1, x2, x3, x4, x5, x6, x_end,True, False, True,
			 			   "%s_M+-_%s" % (save_name, rating_label[i]), "%s_M+-_%s" % (save_name, rating_label[i]))
			if(savepdf_flag):pdf.savefig()
	
	plt.figure(5)
	plt.axhline(y=0, color = "k", linestyle="--", alpha = 0.2)
	for j in rocket.divid:
		plt.axvline(x=j, color = "k", linestyle="--", alpha = 0.2)
	plt.xlabel("STA mm")
	plt.ylabel("Bending Moment Nm")
	plt.title("Bending Moment (Airforce + Gimbal)")
	plt.legend(loc="best")
	if(savefig_flag):plt.savefig(save_name + u"_load_calculation_BendingMoment_same_sign.png")
	if(savepdf_flag):pdf.savefig()

	plt.figure(6)
	plt.axhline(y=0, color = "k", linestyle="--", alpha = 0.2)
	for j in rocket.divid:
		plt.axvline(x=j, color = "k", linestyle="--", alpha = 0.2)
	plt.xlabel("STA mm")
	plt.ylabel("Bending Moment Nm")
	plt.title("Bending Moment (Airforce - Gimbal)")
	plt.legend(loc="best")
	if(savefig_flag):plt.savefig(save_name + u"_load_calculation_BendingMoment_different_sign.png")
	if(savepdf_flag):pdf.savefig()
	
	plt.figure(7)
	plt.axhline(y=0, color = "k", linestyle="--", alpha = 0.2)
	for j in rocket.divid:
		plt.axvline(x=j, color = "k", linestyle="--", alpha = 0.2)
	plt.xlabel("STA mm")
	plt.ylabel("Bending Moment Nm")
	plt.title("Max Bending Moment(abs)")
	plt.legend(loc="best")
	if(savefig_flag):plt.savefig(save_name + u"_load_calculation_BendingMoment_Max.png")
	if(savepdf_flag):pdf.savefig()

	print(u"曲げモーメント出力終了")
	

	# ==== 等価軸力 ====
	print(u"等価軸力計算中...")
	[F_eq_comp, F_eq_tens]=calc_equivalent_axial_force(force_A,M_max,dia)
	print(u"等価軸力計算終了")

	print(u"等価軸力出力開始")

	plt.figure()
	F_eq_comp_a =[]
	for (i, time) in enumerate(rating_time):
		plt.plot(range(rocket.length+1),-F_eq_comp[time], label = "%s" % (rating_label[i]))
		F_eq_comp_a.append(F_eq_comp[time])
	for j in rocket.divid:
		plt.axvline(x=j, color = "k", linestyle="--", alpha = 0.2)
	plt.xlabel("STA mm")
	plt.ylabel("Axial force N")
	plt.ylim(ymin=0)
	plt.title("Equivalent Axial Force (Compression)")
	plt.legend(loc="best")
	if(savefig_flag):plt.savefig(save_name + "_load_calculation_equivalent_axial_force_compressive.png")
	if(savepdf_flag):pdf.savefig()


	#plt.figure()
	#F_eq_tens_a =[]
	#for (i, time) in enumerate(rating_time):
	#	plt.plot(range(rocket.length+1),F_eq_tens[time], label = "%s" % (rating_label[i]))
	#	F_eq_tens_a.append(F_eq_tens[time])
	#for j in rocket.divid:
	#	plt.axvline(x=j, color = "k", linestyle="--", alpha = 0.2)
	#plt.xlabel("STA mm")
	#plt.ylabel("Axial force N")
	#plt.ylim(ymin=0)
	#plt.title("Equivalent Axial Force (Tensile)")
	#plt.legend(loc="best")
	#if(savefig_flag):plt.savefig(save_name + "_load_calculation_equivalent_axial_force_tensile.png")
	#if(savepdf_flag):pdf.savefig()

	print(u"等価軸力出力終了")

	# ==== 評定荷重 ====
	print(u"評定荷重計算中...")
	force_A_rating   = calc_rating_load(force_A,  rocket.divid,rating_time,-1)
	M_max_rating     = calc_rating_load(M_max,    rocket.divid,rating_time, 1)
	F_eq_tens_rating = calc_rating_load(F_eq_tens,rocket.divid,rating_time, 1)
	F_eq_comp_rating = calc_rating_load(F_eq_comp,rocket.divid,rating_time,-1)
	print(u"評定荷重計算終了")

	print(u"評定荷重出力開始")

	fp = open(save_name + "_rating.csv", "w") # 出力先をファイルに変更
	rocket_divid_sta = [v for v in rocket.divid]
	fp.write("開始位置[mm],"       + ",".join(map(str,rocket_divid_sta[0:-1]))+"\n")
	fp.write("終了位置[mm],"       + ",".join(map(str,rocket_divid_sta[1:]))+"\n")
	fp.write("軸力[N],"            + ",".join(map(str,force_A_rating))+"\n")
	fp.write("曲げモーメント[Nm]," + ",".join(map(str,M_max_rating))+"\n")
	#fp.write("等価軸引張力[N],"    + ",".join(map(str,F_eq_tens_rating))+"\n")
	fp.write("等価軸圧縮力[N],"    + ",".join(map(str,F_eq_comp_rating))+"\n")
	fp.close()

	print(u"評定荷重出力終了")

	# ==== 後処理====
	# plt.show()
	
	if(savepdf_flag):pdf.close()
	
	print(u"処理時間:%.1f sec" % (tm.time() - process_start))
	
	# ==== 文字出力 ====
	sys.stdout = open(save_name + u"_output.txt", "w") # 出力先をファイルに変更
	print(u"==== 結果出力 ====")
	print(u"★ 入力値")
	print(u"機体直径 = %d mm,\t機体断面積 = %.3f m2" % (dia*1e3, rocket.area))
	print(u"燃焼時間 = %d 秒" % (burntime))
	# print(u"")
	print(u"凡例 :\t\t\t[LiftOff, MaxQ, MaxDrag, MECO]")
	print(u"時刻 :\t\t\t[%d sec, %d sec, %d sec, %d sec]" % (rating_time[0], rating_time[1], rating_time[2], rating_time[3]))
	print(u"動圧 :\t\t\t[%.1f Pa, %.1f Pa, %.1f Pa, %.1f Pa]" % (q[0], q[1], q[2], q[3]))
	print(u"推力 :\t\t\t[%.1f N, %.1f N, %.1f N, %.1f N]" % (thrust[0], thrust[1], thrust[2], thrust[3]))
	print(u"ジンバル横推力 :\t[%.1f N, %.1f N, %.1f N, %.1f N]" % (T_g[0], T_g[1], T_g[2], T_g[3]))
	print(u"軸力係数C_A :\t\t[%.1f , %.1f , %.1f , %.1f ]" % (C_A, C_A, C_A, C_A))
	print(u"法線力係数C_N :\t\t[%.1f , %.1f , %.1f , %.1f ]" % (C_N, C_N, C_N, C_N))
	print(u"風圧中心x_CP :\t\t[%.1f mm, %.1f mm, %.1f mm, %.1f mm]" % (rocket.x_CP[rating_time[0]], rocket.x_CP[rating_time[1]], rocket.x_CP[rating_time[2]], rocket.x_CP[rating_time[3]]))
	print(u"重心x_CG（参考） :\t[%.1f mm, %.1f mm, %.1f mm, %.1f mm]" % (rocket.x_CG[rating_time[0]], rocket.x_CG[rating_time[1]], rocket.x_CG[rating_time[2]], rocket.x_CG[rating_time[3]]))
	print(u"分割点[mm] :\t\t", end="")
	print(rocket.divid)
	print(u"")
	print(u"★ 計算結果")
	print(u"軸方向加速度 :\t\t\t[%.1f m/s2, %.1f m/s2, %.1f m/s2, %.1f m/s2]" % (X_dotdot_a[0], X_dotdot_a[1], X_dotdot_a[2], X_dotdot_a[3]))
	print(u"垂直方向加速度（空気力） :\t[%.1f m/s2, %.1f m/s2, %.1f m/s2, %.1f m/s2]" %   (Z_dotdot_a_a[0], Z_dotdot_a_a[1], Z_dotdot_a_a[2], Z_dotdot_a_a[3]))
	print(u"垂直方向加速度（ジンバル） :\t[%.1f m/s2, %.1f m/s2, %.1f m/s2, %.1f m/s2]" % (Z_dotdot_d_a[0], Z_dotdot_d_a[1], Z_dotdot_d_a[2], Z_dotdot_d_a[3]))
	print(u"")
	print(u"==== コンポーネント ====")
	print(u"[長さ mm,\tドライ重量 kg,\t推進剤重量 kg,\t推進剤空時 kg]")
	print(u"fairing :\t", end="")
	nose.show()
	print(u"2nd_LOx_tank :\t", end="")
	tank_2nd_LOx.show()
	print(u"2nd_fuel_tank :\t", end="")
	tank_2nd_fuel.show()
	print(u"1st_LOx_tank :\t", end="")
	tank_1st_LOx.show()
	print(u"1st_fuel_tank :\t", end="")
	tank_1st_fuel.show()
	print(u"fin :\t", end="")
	fin.show()
	print(u"engine :\t", end="")
	engine.show()
	
	# 一端ファイルに出力させたものを標準出力に呼び出している
	sys.stdout.close()
	sys.stdout = sys.__stdout__
	print(open(save_name + u"_output.txt","r").read())
