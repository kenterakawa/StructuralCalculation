# -*- coding: utf-8 -*-
# ======
# 多段ロケットのサイジング問題の計算
# 任意の拘束条件を入れた初期重量最小のロケットサイジングを行なう。
#
# 使い方：
# > python Sizing.py (設定ファイル)
# 引数に入力する設定ファイルを指定するとそのファイルを読み込む
# デフォルトは"setting.ini"
#
# Copyright (c) 2016-2017 Takahiro Inagawa
# This code is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
# ======

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import sys
import os
# import imp
import re
import numpy as np
# import scipy.interpolate
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd
try:
    import configparser
except ImportError:  # for python2
    import ConfigParser as configparser


class Stage:
    def __init__(self,
                 stracture_ratio,
                 propellant_consumption_rate=100,
                 jettison=0,
                 Isp=300,
                 thrust=10,
                 number_of_engine=1,
                 nozzle_exit_area=0,
                 use_SeaLevel = False):
        """ロケットの各段クラス
        Args:
            stracture_ratio (float) : 構造効率 (0.0~1.0) , (s)
            propellant_consumption_rate (float, optional) : 推進剤消費割合[%](0~100), (pc_rate)
            jettison (float, optional) : 投棄物質量 [kg]（例えばフェアリング等）
            Isp (float, optional) : 真空中比推力[sec]
            thrust (float, optional) : 真空中推力 [kN/基]
            number_of_engine (int, optional) : エンジン基数[基] (num_engine)
            nozzle_exit_area (float, optional) : エンジン1基あたりのノズル出口面積 [m2/基]
            use_SeaLevel (bool, optional) : 点火時に海面上推力を考慮するかどうか (default : False)
        Attributes:
            m0 : 各段の点火時重量 [kg]
            mf : 各段の終端重量 [kg]
            mass_stracture : 各段の構造重量 [kg]
            upper_mass : 各段の上段部分重量の合計 [kg]
            burn_time : 燃焼時間 [秒]
            acc_ignition : 点火時の加速度 [G]
            acc_cutoff : 燃焼終了時の加速度 [G]
            consumpsion_propellant : 消費推進剤 [kg]
            residual_propellant : 残留推進剤 [kg]
            thrust_SL : 海面上推力 [kN]
            Isp_SL : 海面上比推力 [秒]
            mdot : エンジンの質量流量 [(kg/s)/基]
        """
        self.s = stracture_ratio
        self.pc_rate = propellant_consumption_rate / 100
        self.jettison = jettison
        self.Isp = Isp
        self.thrust = thrust
        self.num_engine = number_of_engine
        self.nozzle_exit_area = nozzle_exit_area
        self.use_SeaLevel = use_SeaLevel
        self.m0 = 0
        self.mf = 0
        self.mass_stracture = 0
        self.upper_mass = 0
        self.burn_time = 0
        self.acc_ignition = 0
        self.acc_cutoff = 0
        self.consumpsion_propellant = 0
        self.residual_propellant = 0
        self.thrust_SL = 0
        self.mdot = 0

    def set_upper_mass(self, upper_mass):
        self.upper_mass = upper_mass

    def get_deltaV(self, propellant):
        """有効推進剤[kg]を入れて⊿V[km/s]を計算
        Args:
            propellant (float) : 有効推進剤[kg]
        """
        g0 = 9.80665
        Pe = 101300  # 大気圧 [Pa]
        self.mass_prop = propellant
        self.consumpsion_propellant = propellant * self.pc_rate
        self.residual_propellant = propellant - self.consumpsion_propellant
        self.mdot = (self.thrust * 1000) / self.Isp / g0
        self.burn_time = self.consumpsion_propellant / (self.mdot * self.num_engine)
        self.mass_stracture = propellant * (1 - self.s) / self.s
        self.mf = self.upper_mass + self.mass_stracture \
                  + self.residual_propellant + self.jettison
        self.m0 = self.mf + self.consumpsion_propellant
        self.deltaV = self.Isp * g0 * np.log(self.m0 / self.mf)

        if self.use_SeaLevel:  # 大気中で点火する際は海面上推力を設定
            self.thrust_SL = self.thrust - self.nozzle_exit_area * Pe / 1000
        else:
            self.thrust_SL = self.thrust  # [kN]
        self.Isp_SL = (self.thrust_SL * 1000) / self.mdot / g0
        self.acc_ignition = (self.num_engine * self.thrust_SL * 1000) / (self.m0 * g0)
        self.acc_cutoff = (self.num_engine * self.thrust * 1000) / (self.mf * g0)

        return self.deltaV

    def display_summary(self):
        print("delta_V\t=\t %d [m/s]" % (self.deltaV))
        print("質量m0\t=\t %.1f [kg]" % (self.m0))
        print("質量mf\t=\t %.1f [kg]" % (self.mf))

    def display_mass(self):
        print("各段質量m0\t=\t %.1f [kg]" % (self.m0 - self.upper_mass))
        print("各段質量mf\t=\t %.1f [kg]" % (self.mf - self.upper_mass))
        print("上段質量mu\t=\t %.1f [kg]" % (self.upper_mass))
        print("有効推進剤質量\t=\t %.1f [kg]" % (self.mass_prop))
        print("推進剤消費率\t=\t %.1f [%%]" % (self.pc_rate * 100))
        print("消費推進剤質量\t=\t %.1f [kg]" % (self.mass_prop * self.pc_rate))
        print("残渣推進剤質量\t=\t %.1f [kg]" % (self.residual_propellant))
        print("構造効率\t=\t %.3f" % (self.s))
        print("構造重量\t=\t %.1f [kg]" % (self.mass_stracture))
        print("投棄物\t=\t %.1f [kg]" % (self.jettison))
        print("Isp(vac)\t=\t %d [s]" % (self.Isp))

    def display_propulsion(self):
        print("エンジン基数\t=\t %d [基]" % (self.num_engine))
        print("推力(vac)\t=\t %.1f [kN/基]" % (self.thrust))
        print("燃焼時間\t=\t %d [s]" % (self.burn_time))
        print("質量流量\t=\t %.1f [(kg/s)/基]" % (self.mdot))
        print("推力(点火時)\t=\t %.1f [N]" % (self.thrust_SL))
        print("Isp(点火時)\t=\t %d [秒]" % (self.Isp_SL))
        print("出口面積\t=\t %.3f [m2/基]" %(self.nozzle_exit_area))
        print("加速度@点火\t=\t %.2f [G]" % (self.acc_ignition))
        print("加速度@CutOff\t=\t %.2f [G]" % (self.acc_cutoff))

    def print(self):
        with open("Sizing.out","w") as output:
            print("delta_V\t=\t %d [m/s]" % (self.deltaV),file=output)
            print("質量m0\t=\t %.1f [kg]" % (self.m0),file=output)
            print("質量mf\t=\t %.1f [kg]" % (self.mf),file=output)
            print("各段質量m0\t=\t %.1f [kg]" % (self.m0 - self.upper_mass),file=output)
            print("各段質量mf\t=\t %.1f [kg]" % (self.mf - self.upper_mass),file=output)
            print("上段質量mu\t=\t %.1f [kg]" % (self.upper_mass),file=output)
            print("有効推進剤質量\t=\t %.1f [kg]" % (self.mass_prop),file=output)
            print("推進剤消費率\t=\t %.1f [%%]" % (self.pc_rate * 100),file=output)
            print("消費推進剤質量\t=\t %.1f [kg]" % (self.mass_prop * self.pc_rate),file=output)
            print("残渣推進剤質量\t=\t %.1f [kg]" % (self.residual_propellant),file=output)
            print("構造効率\t=\t %.3f" % (self.s),file=output)
            print("構造重量\t=\t %.1f [kg]" % (self.mass_stracture),file=output)
            print("投棄物\t=\t %.1f [kg]" % (self.jettison),file=output)
            print("Isp(vac)\t=\t %d [s]" % (self.Isp),file=output)
            print("エンジン基数\t=\t %d [基]" % (self.num_engine),file=output)
            print("推力(vac)\t=\t %.1f [kN/基]" % (self.thrust),file=output)
            print("燃焼時間\t=\t %d [s]" % (self.burn_time),file=output)
            print("質量流量\t=\t %.1f [(kg/s)/基]" % (self.mdot),file=output)
            print("推力(点火時)\t=\t %.1f [N]" % (self.thrust_SL),file=output)
            print("Isp(点火時)\t=\t %d [秒]" % (self.Isp_SL),file=output)
            print("出口面積\t=\t %.3f [m2/基]" %(self.nozzle_exit_area),file=output)
            print("加速度@点火\t=\t %.2f [G]" % (self.acc_ignition),file=output)
            print("加速度@CutOff\t=\t %.2f [G]" % (self.acc_cutoff),file=output)

    def display(self):
        self.display_summary()
        self.display_mass()
        self.display_propulsion()
        self.print()

class Rocket:
    """ロケットクラス、各ステージを内包
    """
    def __init__(self, setting_file, reload = False):
        # print("読み込み設定ファイル : %s" % (setting_file))
        if (reload):  #再読込の際はself.settingの値をそのまま使う
            pass
        else:
            self.setting_file = setting_file
            self.setting = configparser.ConfigParser()
            self.setting.optionxform = str  # 大文字小文字を区別するおまじない
            self.setting.read(setting_file, encoding='utf8')
        setting = self.setting
        # print("読み込みセクション: ", end="")
        # print(setting.sections())

        # 何段あるのかの読み込み、設定ファイルの[x段]のところの最大値を段数に設定
        self.num_stage = 0
        for section in setting.sections():
            m = re.match(r"[0-9]+段", section)
            if m:
                self.num_stage = max(self.num_stage, int(m.group()[:-1]))
        # print("%d段ロケット" % (self.num_stage))

        # 全体セクションの値の読み込み
        self.name = setting.get("全体", "ロケット名")
        self.payload = setting.getfloat("全体", "ペイロード[kg]")
        self.target_deltaV = setting.getfloat("全体", "目標deltaV[km/s]") * 1000

        # 各段の値の読み込み
        self.stages = []
        for i in range(self.num_stage):
            stage = i + 1
            p1 = setting.getfloat("%d段" % (stage), "構造効率[-]")
            p2 = setting.getfloat("%d段" % (stage), "推進剤消費率[%]")
            p3 = setting.getfloat("%d段" % (stage), "投棄物[kg]")
            p4 = setting.getfloat("%d段" % (stage), "Isp(vac)[秒]")
            p5 = setting.getfloat("%d段" % (stage), "推力(vac)[kN/基]")
            p6 = setting.getfloat("%d段" % (stage), "エンジン数[基]")
            p7 = setting.getfloat("%d段" % (stage), "エンジン出口面積[m2/基]")
            p8 = setting.getboolean("%d段" % (stage), "点火時が大気圏内か？")
            self.stages.append(Stage(p1,p2,p3,p4,p5,p6,p7,p8))

        self.col = [("deltaV", "m/s"),
                    ("点火時質量m0", "kg"),
                    ("終了時質量mf", "kg"),
                    ("各段質量m0", "kg"),
                    ("各段質量mf", "kg"),
                    ("上段質量mu", "kg"),
                    ("有効推進剤質量", "kg"),
                    ("推進剤消費率", "%"),
                    ("消費推進剤質量", "kg"),
                    ("残渣推進剤質量", "kg"),
                    ("構造効率", "-"),
                    ("構造重量", "kg"),
                    ("投棄物", "kg"),
                    ("ペイロード", "kg"),
                    ("Isp(vac)", "秒"),
                    ("エンジン基数", "基"),
                    ("推力(vac)", "kN/基"),
                    ("燃焼時間", "秒"),
                    ("質量流量", "(kg/s)/基"),
                    ("質量流量", "kg/s"),
                    ("Isp（点火時）", "秒"),
                    ("推力（点火時）", "kN"),
                    ("出口面積", "m2/基"),
                    ("加速度＠点火時", "G"),
                    ("加速度＠CutOff", "G"),
                    ]

    def calc(self, prop):
        """推進剤を入れた際の各段パラメータ計算
        """
        assert len(prop) == self.num_stage, "引数は段数と一致するリスト"
        self.output_a = []
        self.index_a = []
        self.deltaV_sum = 0
        for stage in range(len(self.stages)-1, -1, -1):
            # 最上段ではペイロード重量を上段重量に足して、下の段では上段のm0を足す
            if (stage == len(self.stages) - 1):
                self.stages[stage].upper_mass = self.payload
            else:
                self.stages[stage].upper_mass = self.stages[stage+1].m0
            dV = self.stages[stage].get_deltaV(prop[stage])
            self.deltaV_sum += dV
            # 出力のためにデータ整理
            sta = self.stages[stage]
            output = [np.around(sta.deltaV),
                      np.around(sta.m0, 1),
                      np.around(sta.mf, 1),
                      np.around(sta.m0 - sta.upper_mass, 1),
                      np.around(sta.mf - sta.upper_mass, 1),
                      np.around(sta.upper_mass, 1),
                      np.around(sta.mass_prop, 1),
                      np.around(sta.pc_rate * 100, 1),
                      np.around(sta.consumpsion_propellant, 1),
                      np.around(sta.residual_propellant, 1),
                      np.around(sta.s, 3),
                      np.around(sta.mass_stracture, 1),
                      np.around(sta.jettison, 1),
                      "",
                      np.around(sta.Isp, 1),
                      sta.num_engine,
                      np.around(sta.thrust, 1),
                      np.around(sta.burn_time, 1),
                      np.around(sta.mdot, 1),
                      np.around(sta.mdot * sta.num_engine, 1),
                      np.around(sta.Isp_SL, 1),
                      np.around(sta.thrust_SL, 1),
                      np.around(sta.nozzle_exit_area, 3),
                      np.around(sta.acc_ignition, 2),
                      np.around(sta.acc_cutoff, 2),
                      ]
            self.output_a.append(output)
            self.index_a.append("%d段" % (stage+1))

        # 出力の段数の順番を整える
        self.output_a.reverse()
        self.index_a.reverse()
        # 全体項目を追加する
        output = [np.around(self.deltaV_sum),
                  "","","","","","","","","","","","",
                  self.payload,"","","","","","","","","","",""]
        self.output_a.append(output)
        self.index_a.append("全体")
        self.col_multi = pd.MultiIndex.from_tuples(self.col, names=['項目', '単位'])

    def display(self):
        """標準出力に出力する"""
        self.df = pd.DataFrame(self.output_a, columns=self.col_multi, index = self.index_a)
        print(self.df.T)

    def to_excel(self, savefile = "", sheet = ""):
        """EXCELに計算結果を出力する。
        Args:
            savefile (str, optional) : 保存ファイル名、ここで指定しなければ設定ファイルで指定した名前
        """
        self.df = pd.DataFrame(self.output_a, columns=self.col_multi, index = self.index_a)
        if savefile == "":
            filename = self.name
        if sheet == "":
            sheetname = self.name
        writer = pd.ExcelWriter("rocket_sizing_" + filename + ".xlsx")
        self.df.T.to_excel(writer, sheetname)
        writer.save()

    def change_setting_value(self, section, key, value):
        """設定ファイルの中身を変更してファイルを保存しなおす
        Args:
            sectiojn (str) : 設定ファイルの[]で囲まれたセクション
            key (str) : 設定ファイルのkey
            value (str or float) : 書き換える値（中でstringに変換）
        """
        self.setting.set(section, key, str(value))
        self.__init__(self.setting_file, reload=True)
        # with open(self.setting_file, 'w') as sf:
        #     self.setting.write(sf)

    def deltaV(self, prop):
        """⊿Vの合計と目標⊿Vの差分の絶対値
        """
        self.calc(prop)
        self.deltaV_sum = 0
        for i, stage in enumerate(self.stages):
            self.deltaV_sum += stage.get_deltaV(prop[i])
        return abs(self.deltaV_sum - self.target_deltaV)

    """
    以下は⊿Vを変化させたときのペイロード重量を求めるための目的関数
    """
    def get_deltaV_from_payload(self, payload, prop):
        self.payload = payload
        # self.deltaV(prop)
        # return self.deltaV_sum
        return self.deltaV(prop)

    """
    以下は最適化のための目的関数
    """
    def initial_mass(self, prop):
        """初段のm0を出力"""
        self.calc(prop)  # 計算しておく
        return self.stages[0].m0

    def limit_stracture1(self, prop, limit):
        """1段目の構造効率"""
        self.calc(prop)
        return self.stages[0].mass_stracture - limit

    def limit_stracture(self, prop, limit):
        """2段目の構造効率"""
        self.calc(prop)
        return self.stages[1].mass_stracture - limit

    def limit_acceralation1(self, prop, limit_acc):
        """1段目の加速度上限"""
        self.calc(prop)
        return limit_acc - self.stages[0].acc_cutoff

    def limit_acceralation2(self, prop, limit_acc):
        """2段目の加速度上限"""
        self.calc(prop)
        return limit_acc - self.stages[1].acc_cutoff

    def limit_V2(self, prop, limit):
        """2段目の⊿V"""
        self.calc(prop)
        return self.stages[1].deltaV - limit


if __name__ == '__main__':
    print("==== 多段ロケットの最適質量配分問題 ====")
    if len(sys.argv) == 1:
        setting_file = 'setting.ini'
    else:
        setting_file = sys.argv[1]
        assert os.path.exists(setting_file), "ファイルが存在しません"
    rocket = Rocket(setting_file)
    prop0 = [1000, 100]  # 初期値のおまじない
    bounds = [[1, np.inf]] * len(rocket.stages)  # 推進剤の上限下限設定のおまじない
    mass_limit_2nd = 821  # ２段目の構造重量の下限
    cons = ({'type': 'eq',
             'fun': lambda prop: rocket.deltaV(prop)},
            {'type': 'ineq',  # 不等式拘束条件 (returnの中身) > 0
             'fun': lambda prop, limit: rocket.limit_stracture(prop, limit),
             'args': (mass_limit_2nd,)})
    result = optimize.minimize(rocket.initial_mass, prop0, method="SLSQP",
                               constraints=cons,
                               bounds=bounds)
    print(result)
    print(result.x)
    rocket.display()
    rocket.to_excel()
