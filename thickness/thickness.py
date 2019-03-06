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

class Material:
    """
    材料クラス、強度計算するための材料特性の値を保持するクラス
    Args:
        modulus_Young (float) : ヤング率 [GPa]
        ratio_Poisson (float) : ポアソン比 [-]
        proof_stress (float) : 耐力 [MPa]
        breaking_stress (float, optional) : 破断応力 [MPa]
    """
    def __init__(self, modulus_Young, ratio_Poisson, proof_stress, breaking_stress=None):
        self.modulus_Young = modulus_Young
        self.ratio_Poisson = ratio_Poisson
        self.proof_stress = proof_stress
        self.breaking_stress = breaking_stress

def thickness_ceil(num, decimals=0):
    """任意の桁数での切り上げ decimals>0の整数
    Args:
        num (float) : 入力値
        decimals (int) : 桁数
    Return:
        (float) 切り上げられた値
    """
    digits = 10. ** decimals
    return np.ceil(num * digits) / digits

class class_thickness:
    def __init__(self):
        #print("==== class thickness ====")
        """
        ひとまずモノコック構造での厚みを出すことを考える。
        ・Bruhnのfig.C8.2~fig.C8.4およびfig.C8.11, figC8.28を目視で関数化
        ・ロケット外径が2000mmで固定値であることと、肉厚が1mm以上必要であろうという想定から
        figC8.5(r/t over 2000)については関数化していない。
        """
        # fig_c8_2 =[[1,10,20,30,100,1000,10000,20000],[4,4,5,6,20,200,2000,4000]]
        fig_c8_2 =[[0,3,4,5,8,10,13,19,27,35,48,70,94,126,179,232,303,414,556,814,1069,1535,1970,2850,2274,4043,5716,7796,10789,14034,19567,28111,36171,25361,43765,50126],
                   [1.0,1.0,1.1,1.2,1.5,2.0,2.6,3.8,5.3,7.0,9.4,14,18,25,35,46,60,81,110,163,210,306,393,572,450,815,1168,1579,2153,2850,3999,5788,7396,5163,8997,10296]]
        fig_c8_3 =[[1,10,28,30,40,100,1000,10000,20000],[4,4,5,5.2,6,15,150,1500,3000]]
        fig_c8_4 =[[1,10,20,30,50,60,100,1000,10000,20000],[4,4,4.1,4.7,6,6.7,10,100,1000,2000]]
        # fig_c8_11 =[[0.01,0.03,0.1,0.2,0.3,0.9,2,20,100],[0.017,0.04,0.09,0.14,0.16,0.2,0.22,0.22,0.22]]
        fig_c8_28 =[[2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 2000, 4000, 8000, 10000, 20000, 40000, 80000, 100000, 200000, 400000, 1000000],
                    [4.2, 4.35, 4.45, 4.6, 4.75, 4.8, 4.95, 5.05, 5.25, 6.2, 7.1, 7.9, 8.6, 9.25, 9.85, 10.5, 11, 11.5, 15.75, 19, 21.75, 24, 26, 28, 29.5, 32.2, 42.2, 55.3, 72.5, 79.2, 103.8, 136.1, 178.4, 194.7, 255.3, 334.7, 478.8]]

        self.func_Bruhn_C8_2 = interpolate.interp1d(fig_c8_2[0],fig_c8_2[1])
        self.func_Bruhn_C8_3 = interpolate.interp1d(fig_c8_3[0],fig_c8_3[1])
        self.func_Bruhn_C8_4 = interpolate.interp1d(fig_c8_4[0],fig_c8_4[1])
        # self.func_Bruhn_C8_11 = interpolate.interp1d(fig_c8_11[0],fig_c8_11[1])
        self.func_Bruhn_C8_28 = interpolate.interp1d(fig_c8_28[0],fig_c8_28[1])

        self.AL5056 = Material(70, 0.3, 140)

        self.list_material = {}

    def calc_Pcr_cylinder_Bruhn_90_prob(self, thickness, radius, length, ratio_Poisson, modulus_Young, eta=1.0):
        """
        Args:
            thickness (float) : 円筒厚み [m]
            radius (float) : 円筒半径 [m]
            length (float) : 円筒長さ [m]
            ratio_Poisson (float) : ポアソン比 [-]
            modulus_Young (float) : 縦弾性係数（ヤング率） [GPa]
            eta (float) : 塑性補正項(plasticity correction factor)、弾性座屈の場合はη=1
        Returns:
            Pcr (float) : 90%確度曲線を用いた軸圧縮力下での円筒の座屈荷重（臨界圧縮軸力） [N]
        Note:
            self.Z (flaot) : Bruhn本C8.4に出てくる幾何係数Z [-]
            self.Kc (float) : Bruhn本に出てくる係数、Zの関数 [-]
            self.Fcr (float) : 90%確度曲線を用いた軸圧縮力下での座屈応力 [MPa]
            self.area (float) : 円筒の断面積 [m2]
        """
        self.Z = (length**2) / radius / thickness * np.sqrt(1 - ratio_Poisson**2)
        if self.Z > 20000:
            self.Pcr = 1e-8 # thicknessが小さすぎてデータ外のため、座屈0で返す。
            return self.Pcr
        rt = radius / thickness
        if (rt < 500): # r / t < 100についても使用する(安全側のため。)
            func_interpolate = self.func_Bruhn_C8_2
        elif (rt >= 500 and rt < 1000):
            func_interpolate = self.func_Bruhn_C8_3
        elif (rt >= 1000 and rt <= 2000):
            func_interpolate = self.func_Bruhn_C8_4
        else:
            print("r/t =", rt)
            raise ValueError("r/t is out of range, error!")
        self.Kc = func_interpolate(self.Z)
        self.Fcr = self.Kc * (np.pi**2) * (modulus_Young * 1e3) / (12 * (1 - ratio_Poisson**2)) \
              * (thickness / length)**2 * eta
        self.area = np.pi * ((radius + thickness)**2 - radius**2)  # 断面積 [m2]
        self.Pcr = self.Fcr * self.area * 1e6
        return self.Pcr

    def coef_conical_Bruhn_C8_25(self, L_rho):
        """Bruhn 薄肉コニカル壁に関するC8.25のグラフの傾きや切片を取得する
        両対数グラフより、Fc = c * (L/ρ)**a のように近似できるとしてa,cを出力
        Args:
            L_rho (float) :　L/ρ
        Return:
            a, c (float, float) : 係数 a,c （意味は上記）
        """
        if (L_rho < 4 and L_rho > 2):
            a = -1.589
            c = 8.71
        elif (L_rho <= 2 and L_rho > 1):
            a = -1.571
            c = 8.75
        elif (L_rho <=1 and L_rho > 0.5):
            a = -1.564
            c = 8.85
        elif (L_rho <= 0.5):
            a = -1.541
            c = 8.861
        else:
            print(L_rho)
            raise ValueError("L/ρ is out of range, error!")
        return a, c

    def calc_Pcr_conical_Bruhn(self, thickness, length, vertical_angle_rad, radius_start, radius_end, modulus_Young):
        """Bruhnの円錐座屈の応力の計算を行う。円錐構造の5°<α<11°のときのみ適用可能
        Args:
            thickness (float) : 円錐の材料厚み[m]
            length (float) : 円錐の垂直長さ [m] (斜辺長さLとは異なるので注意)
            vertical_angle_rad (float) : 円錐の頂角（斜辺傾斜角） [rad]
            radius_start (float) : STA開始位置の半径 [m]
            radius_end (float) : STA終了位置の半径 [m]
            modulus_Young (float) : 材料のヤング率 [GPa]
        Returns:
            Pcr (float) : 薄肉コニカル壁の座屈荷重（臨界圧縮軸力） [N]
        """
        L = length / np.cos(vertical_angle_rad) # 円錐斜辺長さ [m]
        radius_min = min([radius_start, radius_end])
        rho = radius_min / np.cos(vertical_angle_rad)  # 最小曲率半径
        a, c = self.coef_conical_Bruhn_C8_25(L/rho)
        Fcr = (np.e**c) * ((rho / thickness) ** a) * modulus_Young / 1000 # [GPa]
        Pcr = Fcr * 2 * np.pi * radius_min * thickness * 1e9 # [N]
        return Pcr

    def calc_qcr_conical_Bruhn(self, thickness, length, vertical_angle_rad, radius_start, radius_end, ratio_Poisson, modulus_Young):
        """
        Args:
            thickness (float) : 円錐の材料厚み[m]
            length (float) : 円錐の垂直長さ [m] (斜辺長さLとは異なるので注意)
            vertical_angle_rad (float) : 円錐の頂角（斜辺傾斜角） [rad]
            radius_start (float) : STA開始位置の半径 [m]
            radius_end (float) : STA終了位置の半径 [m]
            ratio_Poisson (float) : ポアソン比 [-]
            modulus_Young (float) : 材料のヤング率 [GPa]
        Returns:
            qcr (float) : 外圧下での薄肉コニカル壁の座屈荷重 [Pa]
        Note:
            Z (flaot) : Bruhn本C8.28に出てくる幾何係数Z [-]
            Ky (float) : Bruhn本に出てくる係数、Zの関数 [-]
        """
        L = length / np.cos(vertical_angle_rad) # 円錐斜辺長さ [m]
        radius_ave = (radius_start + radius_end) / 2.0
        rho_ave = radius_ave / np.cos(vertical_angle_rad)  # 平均曲率半径
        Z = L**2 / (rho_ave * thickness) * (1.0 - ratio_Poisson**2)**0.5
        Ky = self.func_Bruhn_C8_28(Z)
        qcr = Ky * modulus_Young * thickness**3 * np.pi**2 / (rho_ave * L**2 * 12 * (1 - ratio_Poisson**2))
        qcr = qcr * 1e9
        return qcr

    def calc_Pcr_conical_NASA_SP8019(self, thickness, vertical_angle_rad, radius_start, radius_end, modulus_Young, ratio_Poisson):
        """NASA SP-8019 "Buckling of thin-walled truncated cones"の章4.2にある式から
        臨界圧縮軸力を計算する
        Args:
            thickness (float) : 円錐の材料厚み [m]
            vertical_angle_rad (float) : 円錐の頂角（斜辺傾斜角） [rad]
            radius_start (float) : STA開始位置の半径 [m]
            radius_end (float) : STA終了位置の半径 [m]
            modulus_Young (float) : 縦弾性係数（ヤング率） [GPa]
            ratio_Poisson (float) : ポアソン比 [-]
        Return:
            Fcr (float) : 臨界圧縮軸力Fcr [N]
        Note:
            10°<α<75° のとき以外の弱体化係数γが文献にないので、計算は不適。
            弱体化係数γ=0.33(10°<α<75°)と決めて計算する
            Pcr (float) : 臨界圧縮軸力Pcr [GPa]
        """
        gamma = 0.33
        Pcr = 2 * np.pi * modulus_Young * (thickness**2) * (np.cos(vertical_angle_rad)**2) \
              / np.sqrt(3 * (1 - ratio_Poisson**2)) * gamma
        Fcr = Pcr * 2 * np.pi * min(radius_start, radius_end) * thickness * 1e9
        return Fcr

    def calc_pcr_conical_NASA_SP8019(self, thickness, length, vertical_angle_rad, radius_start, radius_end, modulus_Young):
        """NASA SP-8019 "4.2.3 Uniform Hydrostatic Pressure"にある式から臨界外圧を計算する
        Args:
            thickness (float) : 円錐の材料厚み [m]
            length (float) : 円錐の垂直長さ [m] (斜辺長さLとは異なるので注意)
            vertical_angle_rad (float) : 円錐の頂角（斜辺傾斜角） [rad]
            radius_start (float) : STA開始位置の半径 [m]
            radius_end (float) : STA終了位置の半径 [m]
            modulus_Young (float) : 縦弾性係数（ヤング率） [GPa]
        Return:
            pcr (float) : 臨界圧縮軸力Fcr [Pa]
        """
        L = length / np.cos(vertical_angle_rad) # 円錐斜辺長さ [m]
        radius_ave = (radius_start + radius_end) / 2.0
        rho_ave = radius_ave / np.cos(vertical_angle_rad)  # 平均曲率半径
        gamma = 0.75
        pcr = 0.92 * modulus_Young * gamma / ((L / rho_ave) * (rho_ave / thickness) ** (5./2)) * 1e9
        return pcr

    def define_thickness(self, rated_force, Q, STA_start, STA_end, dia_start, dia_end, material):
        """load_ratingとrating_sectionの中身から必要な材料厚みを計算する
        1. 仮の厚みを決めて（初期値厚みt0=0.1mm）
        2. 座屈応力を求める
        3. 評定荷重から圧縮強度応力を求め、座屈応力より大きな場合は厚みを増やしていく（0.1mmずつ）
        4. 板厚を実際に作れる形に調整。
            (板厚)=((必要板厚)+0.6mm)を1mm単位で切り上げたもの
            https://istellar.slack.com/archives/C2PBTPBSS/p1530687699000271
        Args:
            rated_force (float) : 評定となる力（等価軸圧縮力） [N]
            Q (float) : 動圧 [N]
            STA_start (float) : 位置STAの開始位置[m]
            STA_end (float) : 位置STAの終了位置 [m]
            dia_start (float) : STA_startでの機体直径 [m]
            dia_end (float) : STA_endでの機体直径 [m]
            material (Material class) : 材料、Materialクラスのインスタンス
        Returns:
            thickness (float) : 各セクションの必要厚み [m]
        Notes:
            value_to_evaluate: 評価値。1以下でOK。                
            rated_pressure (float) : 評定となる力（外部圧力） [Pa]
        """
        length = STA_end - STA_start
        radius_start = dia_start / 2
        radius_end = dia_end / 2
        vertical_angle_rad = np.arctan2(abs(radius_end - radius_start), length)  # 円錐角度α [rad]
        vertical_angle_deg = np.rad2deg(vertical_angle_rad)
        rated_pressure = Q * np.sin(vertical_angle_rad) ** 2

        thickness = 0.001  # 初期厚み 1mm
        value_to_evaluate = 1e9 # 初期の座屈荷重は0としてループに入る
        while value_to_evaluate > 1:
            thickness = thickness + 0.0001
            if (vertical_angle_deg == 0):  # 円筒の場合
                Pcr = self.calc_Pcr_cylinder_Bruhn_90_prob(thickness, radius_start, length,
                                                           material.ratio_Poisson, material.modulus_Young)
                value_to_evaluate = rated_force / Pcr
            elif (vertical_angle_deg > 0 and vertical_angle_deg <= 10):  # 円錐の場合 Bruhn
                Pcr = self.calc_Pcr_conical_Bruhn(thickness, length, vertical_angle_rad,
                                                  radius_start, radius_end, material.modulus_Young)
                qcr = self.calc_qcr_conical_Bruhn(thickness, length, vertical_angle_rad,
                                                  radius_start, radius_end, material.ratio_Poisson, 
                                                  material.modulus_Young)
                value_to_evaluate = (rated_force / Pcr) ** 1.2 + (rated_pressure / qcr) ** 1.2 # Bruhn本 C8.26
            elif (vertical_angle_deg > 10 and vertical_angle_deg < 75):  # 円錐の場合 NASA
                Pcr = self.calc_Pcr_conical_NASA_SP8019(thickness, vertical_angle_rad,
                            radius_start, radius_end, material.modulus_Young, material.ratio_Poisson)
                pcr = self.calc_pcr_conical_NASA_SP8019(thickness, length, vertical_angle_rad,
                                                  radius_start, radius_end, material.modulus_Young)
                value_to_evaluate = rated_force / Pcr + rated_pressure / pcr # NASA SP 8019 4.2.5.4. eq(19)
            else:
                print("length = ", length, "radius_end = ", radius_end, "radius_start = ", radius_start)
                print("vertical_angle_deg = ", vertical_angle_deg)
                raise ValueError("vertical_angle_deg is out of range.")

        # 溶接分の厚み考慮, および
        thickness_welding = 0.0006  # 溶接分の必要厚み [m]
        thickness = thickness_ceil(thickness + thickness_welding, 3)

        return thickness

    def get_MS(self, rated_force, Q, STA_start, STA_end, dia_start, dia_end, material, name, thickness):
        """load_ratingとrating_section, thickness_matrixの中身からM.S.を計算する
        板厚は公差最小分(-0.6mm)とする。
            https://istellar.slack.com/archives/C2PBTPBSS/p1530687699000271
        Args:
            rated_force (float) : 評定となる力（等価軸圧縮力） [N]
            Q (float) : 動圧 [N]
            STA_start (float) : 位置STAの開始位置[m]
            STA_end (float) : 位置STAの終了位置 [m]
            dia_start (float) : STA_startでの機体直径 [m]
            dia_end (float) : STA_endでの機体直径 [m]
            material (Material class) : 材料、Materialクラスのインスタンス
            name(string): コンポーネント名
            thickness (float) : 各セクションの必要厚み [m]
        Returns:
            MS: 1D-list of [name, thickness, F, Fcr, MS]
        """
        length = STA_end - STA_start
        radius_start = dia_start / 2
        radius_end = dia_end / 2
        vertical_angle_rad = np.arctan2(abs(radius_end - radius_start), length)  # 円錐角度α [rad]
        vertical_angle_deg = np.rad2deg(vertical_angle_rad)
        rated_pressure = Q * np.sin(vertical_angle_rad) ** 2

        # 溶接分の厚み考慮
        thickness_welding = 0.0006  # 溶接分の必要厚み [m]
        thickness = thickness - thickness_welding

        if (vertical_angle_deg == 0):  # 円筒の場合
            Pcr = self.calc_Pcr_cylinder_Bruhn_90_prob(thickness, radius_start, length,
                                                       material.ratio_Poisson, material.modulus_Young)
            MS = [[name, thickness + thickness_welding, rated_force, Pcr, Pcr / rated_force -1 ]]
        elif (vertical_angle_deg > 0 and vertical_angle_deg <= 10):  # 円錐の場合 Bruhn
            Pcr = self.calc_Pcr_conical_Bruhn(thickness, length, vertical_angle_rad,
                                              radius_start, radius_end, material.modulus_Young)
            qcr = self.calc_qcr_conical_Bruhn(thickness, length, vertical_angle_rad,
                                              radius_start, radius_end, material.ratio_Poisson, 
                                              material.modulus_Young)
            value_to_evaluate = (rated_force / Pcr) ** 1.2 + (rated_pressure / qcr) ** 1.2 # Bruhn本 C8.26
            MS = [[name + " compression", thickness + thickness_welding, rated_force, Pcr, Pcr / rated_force -1 ]]
            MS.append([name + " external pressure", thickness + thickness_welding, rated_pressure, qcr, qcr/ rated_pressure -1 ])
        elif (vertical_angle_deg > 10 and vertical_angle_deg < 75):  # 円錐の場合 NASA
            Pcr = self.calc_Pcr_conical_NASA_SP8019(thickness, vertical_angle_rad,
                        radius_start, radius_end, material.modulus_Young, material.ratio_Poisson)
            pcr = self.calc_pcr_conical_NASA_SP8019(thickness, length, vertical_angle_rad,
                                              radius_start, radius_end, material.modulus_Young)
            value_to_evaluate = rated_force / Pcr + rated_pressure / pcr # NASA SP 8019 4.2.5.4. eq(19)
            MS = [[name + " compression", thickness + thickness_welding, rated_force, Pcr, Pcr / rated_force -1 ]]
            MS.append([name + " external pressure", thickness + thickness_welding, rated_pressure, pcr, pcr/ rated_pressure -1 ])
        else:
            print("length = ", length, "radius_end = ", radius_end, "radius_start = ", radius_start)
            print("vertical_angle_deg = ", vertical_angle_deg)
            raise ValueError("vertical_angle_deg is out of range.")

        return MS

    def set_material(self, name, modulus_Young, ratio_Poisson, proof_stress, breaking_stress=None):
        self.list_material[name] = Material(modulus_Young, ratio_Poisson, proof_stress, breaking_stress)

    def main(self, load_rating, rating_sections, Q):
        """
        各セクション毎に計算された評定荷重と各セクションの直径から、
        必要十分な材料厚みを出力する。
        Args:
            load_rating: list of [string, float]
                第0要素: 名称。文字列。
                第1要素: 区間最大等価軸力[N]
            rating_sections : list of [string, [float, float], [float, float], string]
                荷重評定区間。
                第0要素: 名称。文字列。
                第1要素: 位置STA[m]
                    第0要素: 開始位置
                    第1要素: 終了位置
                第2要素: 機体径[m]
                    第0要素: 開始位置
                    第1要素: 終了位置
                機体径が同じ場合は円筒モノコック、違う場合は円錐モノコックを前提とする。
                第3要素: 材料名。文字列。
            Q: float
                動圧[Pa]
        Returns:
            thickness_matrix: list of [string, float] 厚みのリスト
                    第0要素: 名称。文字列。
                    第1要素: 厚み[m]
        """
        thickness_matrix = []

        for i, load in enumerate(load_rating):
            name = load[0]
            rated_force = load[1]
            if name != rating_sections[i][0]:
                raise ValueError("The rating_sections and the load_rating are not consistent.")
            STA = rating_sections[i][1]
            dia = rating_sections[i][2]
            material_name = rating_sections[i][3]
            thickness = self.define_thickness(rated_force, Q, STA[0], STA[1], dia[0], dia[1], self.list_material[material_name])
            thickness_matrix.append([name, thickness])
        return thickness_matrix

    def get_MS_matrix(self, load_rating, rating_sections, Q, thickness_matrix):
        """
        各セクション毎に計算された評定荷重と各セクションの直径から、
        必要十分な材料厚みを出力する。
        Args:
            load_rating: list of [string, float]
                第0要素: 名称。文字列。
                第1要素: 区間最大等価軸力[N]
            rating_sections : list of [string, [float, float], [float, float], string]
                荷重評定区間。
                第0要素: 名称。文字列。
                第1要素: 位置STA[m]
                    第0要素: 開始位置
                    第1要素: 終了位置
                第2要素: 機体径[m]
                    第0要素: 開始位置
                    第1要素: 終了位置
                機体径が同じ場合は円筒モノコック、違う場合は円錐モノコックを前提とする。
                第3要素: 材料名。文字列。
            Q: float
                動圧[Pa]
            thickness_matrix: list of [string, float] 厚みのリスト
                    第0要素: 名称。文字列。
                    第1要素: 厚み[m]
        Returns:
            MS_matrix: list of [string, float] 
                    第0要素: 名称。文字列。
                    第1要素: 厚み[m]
                    第2要素: 荷重[N]
                    第3要素: 座屈荷重[N]
                    第4要素: M.S.[-]
        """
        MS_matrix = []
        for i, load in enumerate(load_rating):
            name = load[0]
            rated_force = load[1]
            if name != rating_sections[i][0] or name != thickness_matrix[i][0]:
                raise ValueError("The rating_sections, the load_rating and the thickness_matrix are not consistent.")
            STA = rating_sections[i][1]
            dia = rating_sections[i][2]
            material_name = rating_sections[i][3]
            thickness = thickness_matrix[i][1]
            MS = self.get_MS(rated_force, Q, STA[0], STA[1], dia[0], dia[1], self.list_material[material_name], name, thickness)
            MS_matrix.extend(MS)

        return MS_matrix

    def print_MS_matrix(self, load_rating, rating_sections, Q, thickness_matrix):
        print("MS matrix")
        print("{0:50s},{1:>15s},{2:>20s},{3:>25s},{4:>10s}".format("component name", "thickness[mm]", "load[kN or kPa]", "critical load[kN or kPa]", "MS[-]"))
        MS_matrix = self.get_MS_matrix(load_rating, rating_sections, Q, thickness_matrix)
        for v in MS_matrix:
            print("{0:50s},{1:15.1f},{2:20.0f},{3:25.0f},{4:10.3f}".format(v[0], v[1]*1e3, v[2]*1e-3, v[3]*1e-3, v[4]))
        
if __name__ == '__main__':
    print("==== thickness.py ====")

    load_rating = [["hoge", 100000], ["hogehoge", 200000]]
    rating_sections = [["fuga", [1, 2], [1.2, 1.2]], ["fugafuga", [2, 3], [2, 2]]]
    Q = 0
    E = 0
    mu = 0

    instance = class_thickness()
    thickness_matrix = instance.main(load_rating, rating_sections)
