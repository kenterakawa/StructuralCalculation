# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 Interstellar Technologies Inc. All Rights Reserved.
# Authors : Takahiro Inagawa
# All rights Reserved


"""
ロケット概念検討時の
・タンク内圧と曲げモーメントによる引張応力を計算します
・軸力と曲げモーメントによる座屈応力を計算します

"""



import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import imp
from scipy import interpolate
from scipy import optimize
import configparser
from matplotlib.font_manager import FontProperties

plt.close('all')


class Tank:
    def __init__(self, setting_file, reload = False):
        if (reload):  #再読込の際はself.settingの値をそのまま使う
            pass
        else:
            self.setting_file = setting_file
            self.setting = configparser.ConfigParser()
            self.setting.optionxform = str  # 大文字小文字を区別するおまじない
            self.setting.read(setting_file, encoding='utf8')
        setting = self.setting

        self.name = setting.get("全体", "名前")
        self.is_save_fig = setting.getboolean("全体", "図を保存する？")

        self.diameter = setting.getfloat("タンク", "直径[m]")
        self.thickness = setting.getfloat("タンク", "タンク厚み[mm]")
        self.press = setting.getfloat("タンク", "内圧[MPa]")
        self.length = setting.getfloat("タンク", "NP平行部長さ[m]")
        self.aspect = setting.getfloat("タンク", "タンク鏡板縦横比")
        self.fueldensity = setting.getfloat("タンク", "NP密度[kg/m3]") #追加
        self.loxdensity = setting.getfloat("タンク", "LOx密度[kg/m3]") #追加
        self.obyf = setting.getfloat("タンク", "質量酸燃比O/F") #追加
        self.propflowrate = setting.getfloat("タンク", "推進剤質量流量 [kg/s]") #追加
        self.material_name = setting.get("材料", "材料名")
        self.rupture = setting.getfloat("材料", "引張破断応力[MPa]")
        self.proof = setting.getfloat("材料", "耐力[MPa]")
        self.safety_ratio = setting.getfloat("材料", "安全率")
        self.density = setting.getfloat("材料", "密度[kg/m3]")
        self.poisson_ratio = setting.getfloat("材料","ポアソン比") #追加
        self.youngmodulus = setting.getfloat("材料","ヤング率E[GPa]") #追加
        self.welding_eff = setting.getfloat("材料", "溶接継手効率[%]")

        self.moment_bend = setting.getfloat("外力", "曲げモーメント[N・m]")

        self.radius = self.diameter / 2

        # 重量の計算
        self.volume_hemisphere = 4 / 3 * np.pi * (self.radius**3 * self.aspect - (self.radius - self.thickness/1000)**2 * (self.radius * self.aspect - self.thickness/1000))
        #self.volume_hemisphere = self.thickness * 2 * np.pi * (self.radius**2 + (self.radius*self.aspect)**2 * math.atan(self.aspect) / self.aspect #追加楕円体の計算方法別式
        self.volume_straight = np.pi * (self.radius**2 - (self.radius - self.thickness/1000)**2) * self.length
        self.weight = self.density * (self.volume_hemisphere + self.volume_straight)

        

        # (追加) NP容量の計算
        self.content_volume_head = 4 / 3 * np.pi * ((self.radius - self.thickness/1000)**3 * self.aspect)
        self.content_volume_body = np.pi * (self.radius - self.thickness/1000)**2 * self.length
        self.content_volume = self.content_volume_head + self.content_volume_body
        self.content_weight = self.content_volume * self.fueldensity 

        # (追加) LOx容量の計算 (肉厚、径はNPタンクと同一と仮定)
        self.content_loxweight = self.content_weight * self.obyf
        self.content_loxvolume = self.content_loxweight / self.loxdensity
        self.content_volume_loxbody =  self.content_loxvolume - self.content_volume_head
        self.loxlength =  self.content_volume_loxbody / (np.pi * (self.radius - self.thickness/1000)**2)

        # (追加) LOxタンク重量の計算 (肉厚、径はNPタンクと同一と仮定)
        self.loxvolume_straight = np.pi * (self.radius**2 - (self.radius - self.thickness/1000)**2) * self.loxlength
        self.loxweight = self.density * (self.volume_hemisphere + self.loxvolume_straight)
        self.content_totalweight = self.content_weight + self.content_loxweight
        self.totalweight = self.weight + self.loxweight
        

        # 内圧の計算
        self.stress_theta = self.press * self.radius / (self.thickness / 1000)  # [MPa]
        self.stress_longi = 0.5 * self.stress_theta
        s1 = self.stress_theta
        s2 = self.stress_longi
        self.stress_Mises = np.sqrt(0.5 * (s1**2 + s2**2 + (s1 - s2)**2))

        # (update)曲げモーメントの計算
        d1 = self.diameter
        d2 = self.diameter - (self.thickness / 1000) * 2
        self.I = np.pi / 64 * (d1**4 - d2**4)
        self.stress_bend = self.moment_bend / self.I * 1e-6
        s2 = self.stress_longi + self.stress_bend
        self.stress_total_p = np.sqrt(0.5 * (s1**2 + s2**2 + (s1 - s2)**2))
        self.stress_total_m = np.sqrt(0.5 * (s1**2 + s2**2 + (s1 + s2)**2))

        #座屈評定(Bruhn式)
        #Bruhn Fig8.9近似曲線 Kc = aZ^2+bZ+c
        # Z>100, 100<r/t<500のみ有効
        self.BruhnCa = float(5.6224767 * 10**-8)
        self.BruhnCb = float(0.2028736)
        self.BruhnCc = float(-2.7833319)
        self.BruhnEtha = float(0.9)  #Fcr>20MPaにおいて0.9付近に飽和
        self.BruhnZ = self.length**2 / (self.radius * self.thickness / 10**3) * np.sqrt(1 - self.poisson_ratio**2)
        self.BruhnKc = self.BruhnCa * self.BruhnZ**2 + self.BruhnCb * self.BruhnZ + self.BruhnCc
        self.Fcr = self.BruhnEtha * np.pi**2 * self.youngmodulus * 10**3 * self.BruhnKc / 12 / (1 - self.poisson_ratio**2) * (self.thickness / 10**3 / self.length)**2
        self.tankarea = np.pi * (self.diameter**2 - (self.diameter - self.thickness / 10**3)**2) / 4
        self.bucklingforce = self.Fcr * self.tankarea * 10**3

        #Bruhn Fig8.8a近似曲線
        self.Fcrratio = self.Fcr / self.youngmodulus / 10**3
        

    #タンク内圧計算結果
    def display(self):
        #NPタンク諸元
        print("タンク鏡重量 :\t\t%.1f [kg]" %(self.volume_hemisphere * self.density)) #add
        print("NPタンク重量 :\t\t%.1f [kg]" %(self.weight))
        print("NPタンク内圧 :\t\t%.1f [MPa]" % (self.press))
        print("NPタンク直径 :\t\t%d [mm]" % (self.diameter * 1000))
        print("NPタンク肉厚 :\t\t%.1f [mm]" % (self.thickness))
        print("NPタンク平行部長さ :\t%.1f [m]" % (self.length))
        print("NPタンク鏡部容積 :\t%.1f [m3]" % (self.content_volume_head)) #add
        print("NPタンク平行部容積 :\t%.1f [m3]" % (self.content_volume_body)) #add
        print("NPタンク総容積: \t\t%.3f [m3]" % (self.content_volume))
        print("NP重量: \t\t%.1f [kg]" % (self.content_weight))
        print()
        #LOxタンク諸元(自動計算)
        print("LOxタンク重量 :\t\t%.1f [kg]" %(self.loxweight))
        print("LOxタンク平行部長さ :\t%.1f [m]" % (self.loxlength))
        print("LOxタンク鏡部容積 :\t%.1f [m3]" % (self.content_volume_head)) #add
        print("LOxタンク平行部容積 :\t%.1f [m3]" % (self.content_volume_loxbody)) #add
        print("LOxタンク総容積: \t\t%.3f [m3]" % (self.content_loxvolume))
        print("LOx重量: \t\t%.1f [kg]" % (self.content_loxweight))
        print()
        print("タンク総重量: \t\t%.1f [kg]" % (self.totalweight)) 
        print("推進剤総重量: \t\t%.1f [kg]" % (self.content_totalweight))
        print("燃焼時間: \t\t%.2f [s]" % (self.content_totalweight / self.propflowrate))
        print()
        #応力
        print("内圧 半径方向応力 :\t%.1f [MPa]" % (self.stress_theta))
        print("内圧 長手方向応力 :\t%.1f [MPa]" % (self.stress_longi))
        print("内圧 ミーゼス応力 :\t%.1f [MPa]" % (self.stress_Mises))
        print()
        print("断面二次モーメント :\t%.3f " % (self.I))
        print("曲げモーメント応力 :\t%.4f [MPa]" % (self.stress_bend))
        print("合計 ミーゼス応力圧縮 :\t%.1f [MPa]" % (self.stress_total_p))
        print("合計 ミーゼス応力引張 :\t%.1f [MPa]" % (self.stress_total_m))
        #タンク座屈計算結果
        print("係数Z :\t\t\t%.2f " % (self.BruhnZ))
        print("係数Kc :\t\t%.2f " % (self.BruhnKc))

        print("座屈応力Fcr(90%%確度) :\t%.2f [MPa]" % (self.Fcr))
        print("座屈限界軸力(90%%確度) :\t%.2f [kN]" % (self.bucklingforce))
        print()
        print("99%%確度座屈評価 Fig.8.8aにて評価をしてください。合格条件:Fcr/E計算値>Fcr/Eグラフ読み取り値")
        print("(評価用)r/t :\t\t%.0f " % (self.radius / self.thickness * 10**3))
        print("(評価用)L/r :\t\t%.1f " % (self.length / self.radius))        
        print("計算値Fcr/E :\t\t%.6f " % (self.Fcrratio))

    def print(self):
        with open("tankout_S1.out","w") as output:
            print("タンク鏡重量:\t%.1f [kg]" %(self.volume_hemisphere * self.density),file=output) #add
            print("NPタンク重量:\t%.1f [kg]" %(self.weight),file=output)
            print("NPタンク内圧:\t%.1f [MPa]" % (self.press),file=output)
            print("NPタンク直径:\t%d [mm]" % (self.diameter * 1000),file=output)
            print("NPタンク肉厚:\t%.1f [mm]" % (self.thickness),file=output)
            print("NPタンク平行部長さ:\t%.2f [m]" % (self.length),file=output)
            print("NPタンク鏡部容積:\t%.2f [m3]" % (self.content_volume_head),file=output) #add
            print("NPタンク平行部容積:\t%.2f [m3]" % (self.content_volume_body),file=output) #add
            print("NPタンク平行部重量:\t%.1f [kg]" % (self.volume_straight * self.density),file=output) #add
            print("NPタンク容積: \t%.3f [m3]" % (self.content_volume),file=output)
            print("NP重量: \t%.1f [kg]" % (self.content_weight),file=output)
            print()
            #LOxタンク諸元(自動計算)
            print("LOxタンク重量:\t%.1f [kg]" %(self.loxweight),file=output)
            print("LOxタンク平行部長さ:\t%.2f [m]" % (self.loxlength),file=output)
            print("LOxタンク鏡部容積:\t%.2f [m3]" % (self.content_volume_head),file=output) #add
            print("LOxタンク平行部容積:\t%.2f [m3]" % (self.content_volume_loxbody),file=output) #add
            print("LOxタンク平行部重量:\t%.1f [kg]" % (self.loxvolume_straight * self.density),file=output) #add
            print("LOxタンク容積: \t%.3f [m3]" % (self.content_loxvolume),file=output)
            print("LOx重量: \t%.1f [kg]" % (self.content_loxweight),file=output)
            print()
            print("タンク総重量: \t%.1f [kg]" % (self.totalweight),file=output)
            print("推進剤総重量: \t%.1f [kg]" % (self.content_totalweight),file=output)
            print("燃焼時間: \t%.2f [s]" % (self.content_totalweight / self.propflowrate),file=output)
            print()
            #応力
            print("内圧半径方向応力:\t%.1f [MPa]" % (self.stress_theta),file=output)
            print("内圧長手方向応力:\t%.1f [MPa]" % (self.stress_longi),file=output)
            print("内圧ミーゼス応力:\t%.1f [MPa]" % (self.stress_Mises),file=output)
            print()
            print("断面二次モーメント:\t%.3f [m4] " % (self.I),file=output)
            print("曲げモーメント応力:\t%.4f [MPa]" % (self.stress_bend),file=output)
            print("合計ミーゼス応力圧縮:\t%.1f [MPa]" % (self.stress_total_p),file=output)
            print("合計ミーゼス応力引張:\t%.1f [MPa]" % (self.stress_total_m),file=output)
            #タンク座屈計算結果
            print("係数Z:\t%.2f [ND]" % (self.BruhnZ),file=output)
            print("係数Kc:\t%.2f [ND]" % (self.BruhnKc),file=output)
            print("座屈応力Fcr(90%%確度):\t%.2f [MPa]" % (self.Fcr),file=output)
            print("座屈限界軸力(90%%確度):\t%.2f [kN]" % (self.bucklingforce),file=output)
            print("(評価用)r/t:\t%.0f [ND]" % (self.radius / self.thickness * 10**3),file=output)
            print("(評価用)L/r:\t%.1f [ND]" % (self.length / self.radius),file=output)
            print("計算値Fcr/E:\t%.6f [ND]" % (self.Fcrratio),file=output)
       

    def change_setting_value(self, section, key, value):
        """設定ファイルの中身を変更してファイルを保存しなおす
        Args:
            sectiojn (str) : 設定ファイルの[]で囲まれたセクション
            key (str) : 設定ファイルのkey
            value (str or float) : 書き換える値（中でstringに変換）
        """
        self.setting.set(section, key, str(value))
        self.__init__(self.setting_file, reload=True)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        setting_file = 'setting_S1.ini'
    else:
        setting_file = sys.argv[1]
        assert os.path.exists(setting_file), "ファイルが存在しません"
    plt.close("all")
    plt.ion()
    t = Tank(setting_file)
    t.display()
    t.print()
    # t.plot()

    def calc_moment(vel, rho, diameter, length, CN):
        """機体にかかる曲げモーメントを概算
        Args:
            vel (float) : 速度 [m/s]
            rho (float) : 空気密度 [kg/m3]
            diameter (float) : 直径 [m]
            length (float) : 機体長さ [m]
            CN (float) : 法戦力係数
        Return:
            法戦力、
            法戦力が全部先端にかかったとしたと仮定した最大の曲げモーメント、
            真ん中重心の際の曲げモーメント
        """
        A = diameter **2 * np.pi / 4
        N = 0.5 * rho * vel**2 * A * CN
        moment_max = length * N
        return N, moment_max, moment_max/2


    """ 速度 420m/s, 空気密度0.4kg/m3, 機体長さ14m, 法戦力係数0.4 """
    N, moment_max, moment_nominal = calc_moment(420, 0.4, t.diameter, 14, 0.4)


    moment_a = np.linspace(0, moment_max * 1.1)
    press_l = [0.3, 0.4, 0.5, 0.6, 0.7]
    eff = t.welding_eff / 100






    

    """ グラフを書く (使わない) """
    """
    plt.figure(0)
    plt.figure(1)
    for p in press_l:
        t.change_setting_value("タンク", "内圧[MPa]", p)
        temp0 = []
        temp1 = []
        temp2 = []
        temp3 = []
        for i in moment_a:
            t.change_setting_value("外力", "曲げモーメント[N・m]", i)
            temp0.append(t.stress_total_p)
            temp3.append(t.stress_total_m)
            temp1.append(t.stress_bend)
            temp2.append(t.stress_theta)
        plt.figure(0)
        plt.plot(moment_a/1e3, temp0, label="内圧 %.1f [MPa]" % (p))
        plt.figure(1)
        plt.plot(moment_a/1e3, temp3, label="内圧 %.1f [MPa]" % (p))
        plt.figure(3)
        plt.plot(moment_a/1e3, temp2, label="内圧 %.1f [MPa]" % (p))
    plt.figure(0)
    # plt.axhline(y=t.rupture / eff, label="%s 破断応力" % (t.material_name), color="C6")
    # plt.axhline(y=t.proof / eff, label="%s 0.2%%耐力" % (t.material_name), color="C7")
    # plt.axhline(y=t.rupture / t.safety_ratio / eff, label="安全率= %.2f 破断応力" % (t.safety_ratio), color="C8")
    plt.axhline(y=t.proof / t.safety_ratio / eff, label="%s, 安全率= %.2f 耐力" % (t.material_name, t.safety_ratio), color="C6")
    plt.axvline(x=moment_max / 1e3, label="飛行時最大曲げM（高見積り）", color="C7")
    plt.axvline(x=moment_nominal / 1e3, label="飛行時最大曲げM（低見積り）", color="C8")
    plt.grid()
    plt.xlabel("曲げモーメント [kN・m]")
    plt.ylabel("引張側 最大ミーゼス応力")
    plt.title(t.name + " タンク応力, 肉厚 = %.1f[mm], 直径 = %.1f[m], 溶接効率 = %d[%%]" % (t.thickness, t.diameter, t.welding_eff))
    plt.legend()
    if(t.is_save_fig):plt.savefig("stress_tank_" + t.name + "_1.png")

    plt.figure(1)
    # plt.axhline(y=t.rupture / eff, label="%s 破断応力" % (t.material_name), color="C6")
    # plt.axhline(y=t.proof / eff, label="%s 0.2%%耐力" % (t.material_name), color="C7")
    # plt.axhline(y=t.rupture / t.safety_ratio / eff, label="安全率= %.2f 破断応力" % (t.safety_ratio), color="C8")
    plt.axhline(y=t.proof / t.safety_ratio / eff, label="%s, 安全率= %.2f 耐力" % (t.material_name, t.safety_ratio), color="C6")
    plt.axvline(x=moment_max / 1e3, label="飛行時最大曲げM（高見積り）", color="C7")
    plt.axvline(x=moment_nominal / 1e3, label="飛行時最大曲げM（低見積り）", color="C8")
    plt.grid()
    plt.xlabel("曲げモーメント [kN・m]")
    plt.ylabel("圧縮側 最大ミーゼス応力")
    plt.title(t.name + " タンク応力, 肉厚 = %.1f[mm], 直径 = %.1f[m], 溶接効率 = %d[%%]" % (t.thickness, t.diameter, t.welding_eff))
    plt.legend()
    if(t.is_save_fig):plt.savefig("stress_tank_" + t.name + "_2.png")

    plt.figure(2)
    plt.plot(moment_a/1e3, temp1, label="法戦力による曲げモーメント")
    # plt.axhline(y=t.rupture / eff, label="%s 破断応力" % (t.material_name), color="C6")
    # plt.axhline(y=t.proof / eff, label="%s 0.2%%耐力" % (t.material_name), color="C7")
    # plt.axhline(y=t.rupture / t.safety_ratio / eff, label="安全率= %.2f 破断応力" % (t.safety_ratio), color="C8")
    plt.axhline(y=t.proof / t.safety_ratio / eff, label="%s, 安全率= %.2f 耐力" % (t.material_name, t.safety_ratio), color="C6")
    plt.axvline(x=moment_max / 1e3, label="飛行時最大曲げM（高見積り）", color="C7")
    plt.axvline(x=moment_nominal / 1e3, label="飛行時最大曲げM（低見積り）", color="C8")
    plt.grid()
    plt.xlabel("曲げモーメント [kN・m]")
    plt.ylabel("長手方向応力")
    plt.title(t.name + " タンク応力, 肉厚 = %.1f[mm], 直径 = %.1f[m], 溶接効率 = %d[%%]" % (t.thickness, t.diameter, t.welding_eff))
    plt.legend()
    if(t.is_save_fig):plt.savefig("stress_tank_" + t.name + "_3.png")

    plt.figure(3)
    plt.axhline(y=t.proof / t.safety_ratio / eff, label="%s, 安全率= %.2f 耐力" % (t.material_name, t.safety_ratio), color="C6")
    plt.axvline(x=moment_max / 1e3, label="飛行時最大曲げM（高見積り）", color="C7")
    plt.axvline(x=moment_nominal / 1e3, label="飛行時最大曲げM（低見積り）", color="C8")
    plt.grid()
    plt.xlabel("曲げモーメント [kN・m]")
    plt.ylabel("タンク　フープ応力")
    plt.title(t.name + " タンク　フープ応力, 肉厚 = %.1f[mm], 直径 = %.1f[m], 溶接効率 = %d[%%]" % (t.thickness, t.diameter, t.welding_eff))
    plt.legend()
    if(t.is_save_fig):plt.savefig("stress_tank_" + t.name + "_4.png")
    """
    
