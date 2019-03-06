# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 Interstellar Technologies Inc. All Rights Reserved.
# Authors : Takahiro Inagawa
# ALl right Reserved

"""
ロケット概念検討時のタンク内圧と曲げモーメントによる引張応力の関係について
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
        self.length = setting.getfloat("タンク", "平行部長さ[m]")
        self.aspect = setting.getfloat("タンク", "タンク鏡板縦横比")

        self.material_name = setting.get("材料", "材料名")
        self.rupture = setting.getfloat("材料", "引張破断応力[MPa]")
        self.proof = setting.getfloat("材料", "耐力[MPa]")
        self.safety_ratio = setting.getfloat("材料", "安全率")
        self.density = setting.getfloat("材料", "密度[kg/m3]")
        self.welding_eff = setting.getfloat("材料", "溶接継手効率[%]")

        self.moment_bend = setting.getfloat("外力", "曲げモーメント[N・m]")

        self.radius = self.diameter / 2

        # 重量の計算
        self.volume_hemisphere = 4 / 3 * np.pi * (self.radius**3 * self.aspect - (self.radius - self.thickness/1000)**2 * (self.radius * self.aspect - self.thickness/1000))
        self.volume_straight = np.pi * (self.radius**2 - (self.radius - self.thickness/1000)**2) * self.length
        self.weight = self.density * (self.volume_hemisphere + self.volume_straight)

        # 内圧の計算
        self.stress_theta = self.press * self.radius / (self.thickness / 1000)  # [MPa]
        self.stress_longi = 0.5 * self.stress_theta
        s1 = self.stress_theta
        s2 = self.stress_longi
        self.stress_Mises = np.sqrt(0.5 * (s1**2 + s2**2 + (s1 - s2)**2))

        # 曲げモーメントの計算
        d1 = self.diameter
        d2 = self.diameter - (self.thickness / 1000) * 2
        self.I = np.pi / 64 * (d1**4 - d2**4)
        self.stress_bend = self.moment_bend / self.I * 1e-6
        s2 = self.stress_longi + self.stress_bend
        self.stress_total_p = np.sqrt(0.5 * (s1**2 + s2**2 + (s1 - s2)**2))
        self.stress_total_m = np.sqrt(0.5 * (s1**2 + s2**2 + (s1 + s2)**2))

    def display(self):
        print("タンク重量 :\t\t%.1f [kg]" %(self.weight))
        print("タンク内圧 :\t\t%.1f [MPa]" % (self.press))
        print("タンク直径 :\t\t%d [mm]" % (self.diameter * 1000))
        print("タンク肉厚 :\t\t%.1f [mm]" % (self.thickness))
        print("タンク平行部長さ :\t%.1f [m]" % (self.length))
        print()
        print("内圧 半径方向応力 :\t%.1f [MPa]" % (self.stress_theta))
        print("内圧 長手方向応力 :\t%.1f [MPa]" % (self.stress_longi))
        print("内圧 ミーゼス応力 :\t%.1f [MPa]" % (self.stress_Mises))
        print()
        print("断面二次モーメント :\t%.3f " % (self.I))
        print("曲げモーメント応力 :\t%.1f [MPa]" % (self.stress_bend))
        print("合計 ミーゼス応力圧縮 :\t%.1f [MPa]" % (self.stress_total_p))
        print("合計 ミーゼス応力引張 :\t%.1f [MPa]" % (self.stress_total_m))

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
        setting_file = 'setting.ini'
    else:
        setting_file = sys.argv[1]
        assert os.path.exists(setting_file), "ファイルが存在しません"
    plt.close("all")
    plt.ion()
    t = Tank(setting_file)
    t.display()
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
