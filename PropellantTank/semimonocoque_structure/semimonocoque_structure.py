# -*- coding: utf-8 -*-
"""

Rev.A Created on Fri May 18 16:48:50 2018 @author: murohara
Rev.B Aug-22 2018 kterakawa


セミモノコック構造の座屈応力を計算するスクリプトです。
計算フローは以下。
1.ストリンガがないものとして軸圧縮力がすべてパネル部にかかるとしてパネル肉厚を計算(設定された安全率を満たすように）。
2.設定ファイルから希望ストリンガ肉厚、フランジ幅を読み取り
3.パネル部+ストリンガ部で軸圧縮力をもつようにストリンガウェブ高さを計算（ストリンガ部が設定された安全率を満たすように）。
4.再度パネル部の安全率を計算し、諸元をcsvファイルに出力。

肉厚に対して座屈応力と等価軸圧縮力、耐力の関係をグラフに出力する。
等価軸圧縮力<耐力かつ座屈応力の範囲で設計を行う。
詳細についてはChapter9.docxを参照のこと。源泉はBruhn本9章。
弾性座屈のみを想定してη=１で計算

注意点:
**曲げモーメントは等価軸圧縮力に変換されて計算されているP=4M/Dの等式を用いている。
**リング枚数は計算結果関係ない。
**収束しない場合はfor文の範囲を変えてより初期肉厚を厚くするとうまくいく。
"""

import numpy as np
pi=np.pi
sqrt=np.sqrt
import configparser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate
import csv


class semimonocoque:
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
        
        #ストリンガ物性値読み込み
        self.STR_material = setting.get('ストリンガ物性値','材料名')
        self.STR_E = setting.getfloat('ストリンガ物性値','ヤング率[GPa]')
        self.STR_v = setting.getfloat('ストリンガ物性値','ポアソン比')
        self.STR_y = setting.getfloat('ストリンガ物性値', '耐力[MPa]')
        self.STR_density = setting.getfloat('ストリンガ物性値', '密度[kg/m3]') #未使用

        #外板物性値読み込み
        self.FRP_material = setting.get('外板物性値','材料名')
        self.FRP_E = setting.getfloat('外板物性値','ヤング率[GPa]')
        self.FRP_v = setting.getfloat('外板物性値','ポアソン比')
        self.FRP_y = setting.getfloat('外板物性値', '引張強さ[MPa]')
        self.FRP_density = setting.getfloat('外板物性値', '密度[kg/m3]') #未使用

        #ストリンガ寸法初期値読み込み
        self.STR_Fl_b = setting.getfloat('ストリンガ初期寸法','フランジ幅b[mm]')
        self.STR_Fl_s = setting.getfloat('ストリンガ初期寸法','フランジ肉厚s[mm]')
        self.STR_WE_t = setting.getfloat('ストリンガ初期寸法','ウェブ初期肉厚t[mm]')
        
        #計算条件読み込み
        self.f=setting.getfloat('計算条件', '等価圧縮軸力[kN]')
        self.safety_factor=setting.getfloat('計算条件', '全体安全率')
        self.safety_factor_FRP=setting.getfloat('計算条件', '外板単体安全率')
        self.external_diameter=setting.getfloat('計算条件', '外径[mm]')
        self.length_all=setting.getfloat('計算条件', '全長[mm]')
        self.num_ring=setting.getfloat('計算条件', 'リング枚数')
        self.pressure=setting.getfloat('計算条件', '内圧[MPa]')
        self.num_stringer=setting.getfloat('計算条件', '桁数')
        #self.constraint=setting.getfloat('計算条件', '支持条件')    未使用
        self.length = self.length_all/(self.num_ring+1)
        
        #外力読み込み
        self.compressive_stress=setting.getfloat('外力', '軸圧縮力[MPa]')
        self.bending_moment=setting.getfloat('外力', '曲げモーメント[MN*m]')
        
        #Bruhnのfig.C9.1を目視で関数化
        #ロケット外径2000mmで固定、肉厚tは1mm以上必要との想定からr/t over 2000はコメントアウト
        fig_c9_1_500 =[[1,5,15,35,50,100,190,600,1200,2400,4800],[4,5,8,15,21,42,80,250,500,1000,2000]]
        fig_c9_1_700 =[[1,13,30,50,80,400,800,1600,3200,6400],[4,7,11,17,25,125,250,500,1000,2000]]
        fig_c9_1_1000 =[[1,13,30,60,100,200,400,800,5400,10800],[4,7,10.5,16,23,40,76,150,1000,2000]]
        #fig_c9_1_2000 =[[1,5,40,60,100,300,600,900,9000],[4,5,12,15,21,50,90,130,1000]]
        #fig_c9_1_3000 =[[1,5,40,60,100,200,400,800,1000,5000,11000],[4,5,12,15,20.5,33,55,100,120,500,1000]]

        self.function500 = interpolate.interp1d(fig_c9_1_500[0],fig_c9_1_500[1])
        self.function700 = interpolate.interp1d(fig_c9_1_700[0],fig_c9_1_700[1])
        self.function1000 = interpolate.interp1d(fig_c9_1_1000[0],fig_c9_1_1000[1])
        #self.function2000 = interpolate.interp1d(fig_c9_1_2000[0],fig_c9_1_2000[1])
        #self.function3000 = interpolate.interp1d(fig_c9_1_3000[0],fig_c9_1_3000[1])
        
    def panel_designe(self):
        #座屈応力=パネル荷重応力となるようなパネル肉厚を求める
        Fcr_list = []
        stress_panel_list =[]
        s_ratio_panel = [] 
        x = []
        result_panel = [0,0,0,0]
        for i in range(1,30,1): #rangeで板圧の条件を設定する
            t = 0.1*i
            
            #座屈応力Fcr [MPa]の計算
            b = pi*self.external_diameter/self.num_stringer  #パネル周方向長さ
            Z = b**2/((self.external_diameter/2-t)*t)*sqrt(1-self.FRP_v**2)
            r_over_t = (self.external_diameter/2-t)/t
            if r_over_t >=100 and r_over_t<600:
                Kc = self.function500(Z)
                #print(500)
            elif r_over_t>=600 and r_over_t<850:
                Kc = self.function700(Z)
                #print(700)
            elif r_over_t>=850 and r_over_t<2000:
                Kc = self.function1000(Z)
                #print(1000)
            else:
                continue
            Fcr = Kc*pi**2*self.FRP_E/(12*sqrt(1-self.FRP_v**2))*(t/b)**2*1000
            
            #曲面パネルへの荷重応力stress_panel [MPa]の計算
            area_panel = pi*((self.external_diameter/2)**2-(self.external_diameter/2-t)**2)/self.num_stringer
            stress_panel = self.f/(area_panel*self.num_stringer)*10**3+self.pressure+4*self.bending_moment/(self.external_diameter*10**(-3))
            
            Fcr_list.append(Fcr)
            stress_panel_list.append(stress_panel)
            s_ratio_panel.append(Fcr/stress_panel)
            x.append(t)
            
            if Fcr/stress_panel > self.safety_factor_FRP and result_panel[0] == 0:
            #result[パネル肉厚、パネル荷重応力、パネル座屈荷重]
                result_panel[0]=t
                result_panel[1]=area_panel
                result_panel[2]= stress_panel
                result_panel[3] = Fcr
                
        plt.figure()
        title = "パネル肉厚の決定"
        plt.subplot(2,1,1)
        plt.plot(x,Fcr_list, color="red", label = "座屈応力")
        plt.plot(x,stress_panel_list, color="blue", label = "荷重応力")
        plt.hlines(self.FRP_y, 1, 2, color="red",label="耐力")
        #plt.ylim(1300,1500)
        #plt.xlabel("パネル肉厚 [mm]")
        plt.ylabel("応力 [MPa]")
        plt.legend()
        plt.title(title)  
        
        plt.subplot(2,1,2)
        plt.plot(x,s_ratio_panel,label = "安全率")
        plt.hlines(1, 1, 2, color="red")
        plt.ylim(0,2)
        plt.xlabel("パネル肉厚[mm]")
        plt.ylabel("安全率")
        plt.legend()
        plt.close
        
        return result_panel
    
    def stringer_design(self,t_panel,t_stringer,area_panel,stress_panel,Fcr_panel):
        #ストリンガのウェブ高さを計算する。
        #ストリンガも含めた形状の安全率を再度計算する。
        #ストリンガのフランジ部は固定値とする。
        #MOMO2を参考にT字形状とする。
        self.t_panel = t_panel
        self.t_stringer = self.STR_WE_t  #self.STR_WE_t: ウェブ肉厚初期値 (setting.iniから取得)
        self.area_panel = area_panel
        self.stress_panel = stress_panel
        self.Fcr_panel = Fcr_panel

        #self.STR_Fl_b: フランジ幅固定値 (setting.iniから取得)
        #self.STR_Fl_s: フランジ肉厚固定値 (setting.iniから取得)
       
        
        stress_list=[]
        Fcr_stringer_list = []
        s_ratio_panel = []
        s_ratio_stringer =[]
        H_list = []
        result_H = 0
        for i in range(10,1000,1):
            H = self.STR_Fl_s + i*0.1
            
            #ストリンガ荷重応力の計算
            area_stringer = (H-self.STR_Fl_s)*self.t_stringer+self.STR_Fl_s*self.STR_Fl_b        
            stress = self.f/((area_stringer + self.area_panel)*self.num_stringer)*10**3+self.pressure+4*self.bending_moment/(self.external_diameter*10**(-3))
            
            #座屈応力の計算
            e1 = H-(H**2*self.t_stringer+self.STR_Fl_s**2*(self.STR_Fl_b-self.t_stringer))/(2*(self.STR_Fl_b*self.STR_Fl_s+self.t_stringer*(H-self.STR_Fl_s)))
            e2 = (H**2*self.t_stringer+self.STR_Fl_s**2*(self.STR_Fl_b-self.t_stringer))/(2*(self.STR_Fl_b*self.STR_Fl_s+self.t_stringer*(H-self.STR_Fl_s)))
            I = (self.t_stringer*e1**3+self.STR_Fl_b*e2**3-(self.STR_Fl_b-self.t_stringer)*(e2-self.STR_Fl_s)**3)/3
            Fcr_stringer = pi**2*self.STR_E*10**3*I/(self.length**2*area_stringer)

            stress_list.append(stress)
            Fcr_stringer_list.append(Fcr_stringer)
            s_ratio_panel.append(self.Fcr_panel/stress)
            s_ratio_stringer.append(Fcr_stringer/stress)
            H_list.append(H)
            
            if Fcr_stringer/stress >self.safety_factor and result_H==0:
               result_s_ratio_panel=self.Fcr_panel/stress   
               result_s_ratio_stringer=Fcr_stringer/stress
               result_t_panel=self.t_panel
               result_t_stringer=self.t_stringer
               result_H=H
               result_Fcr_panel = self.Fcr_panel
               result_Fcr_stringer=Fcr_stringer
               result_stress=stress
               
               print("----------パネル諸元-------------")
               print("長さ[mm] " + str(self.length))
               print("外径[mm] " + str(self.external_diameter))
               print("肉厚[mm] " + str(result_t_panel))
               print("荷重応力[MPa] " + str(result_stress))
               print("座屈荷重[MPa] " + str(result_Fcr_panel))
               print("安全率　　　　　" + str(result_s_ratio_panel))
               
               print("----------ストリンガ諸元-------------") 
               print("長さ[mm]　" + str(self.length))
               print("ウェブ肉厚[mm] " + str(result_t_stringer))
               print("フランジ肉厚[mm] " + str(self.STR_Fl_s))
               print("フランジ幅[mm] " + str(self.STR_Fl_b))
               print("ウェブ高さ[mm] " + str(result_H)) 
               print("荷重応力[MPa] " + str(result_stress))
               print("座屈荷重[MPa] " + str(result_Fcr_stringer))
               print("座屈荷重[kN] " + str(result_Fcr_stringer*area_stringer/10**3))
               print("安全率 " + str(result_s_ratio_stringer))
               print("縦貫材本数 " + str(self.num_stringer))
               print("リング枚数 " + str(self.num_ring))
               
               #csVファイルに出力
               with open('セミモノコック構造設計.csv','w',newline="") as f:
                   writer = csv.writer(f)
                   writer.writerow(["パネル諸元",""])
                   writer.writerow(["長さ[mm]",self.length])
                   writer.writerow(["外径[mm]",self.external_diameter])
                   writer.writerow(["肉厚[mm]",result_t_panel])
                   writer.writerow(["荷重応力[MPa]",result_stress])
                   writer.writerow(["座屈荷重[MPa]",result_Fcr_panel])
                   writer.writerow(["安全率",result_s_ratio_panel])
                   writer.writerow(["",""])
                   writer.writerow(["ストリンガ諸元",""])
                   writer.writerow(["長さ[mm]",self.length])
                   writer.writerow(["ウェブ肉厚[mm]",result_t_stringer])
                   writer.writerow(["フランジ肉厚[mm] ",self.STR_Fl_s])
                   writer.writerow(["フランジ幅[mm] ",self.STR_Fl_b])
                   writer.writerow(["ウェブ高さ[mm]",result_H])
                   writer.writerow(["荷重応力[MPa]",result_stress])
                   writer.writerow(["座屈荷重[MPa]",result_Fcr_stringer])
                   writer.writerow(["安全率",result_s_ratio_stringer])
                   writer.writerow(["リング枚数",self.num_ring])
               
        if result_H ==0:
            print("指定の外板、ストリンガ範囲内では座屈します。")
            
        title = "ストリンガウェブ高さの決定"
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(H_list,Fcr_stringer_list, color="red", label = "ストリンガ座屈応力")
        plt.plot(H_list,stress_list, color="blue", label = "荷重応力")
        plt.ylabel("応力 [MPa]")
        plt.legend()
        plt.title(title)

        plt.subplot(2,1,2)
        plt.plot(H_list,s_ratio_stringer,label = "ストリンガ安全率")
        plt.plot(H_list,s_ratio_panel,label = "パネル安全率")
        plt.hlines(1, 1, 30, color="red")
        plt.ylim(0,1.2)
        plt.xlabel("H [mm]")
        plt.ylabel("安全率 [mm]")
        plt.legend()

if __name__ == '__main__':
    setting_file = 'setting.ini'
    semimonocoque = semimonocoque(setting_file)
    result_panel = semimonocoque.panel_designe()
    semimonocoque.stringer_design(result_panel[0],int(result_panel[0]+1),result_panel[1],result_panel[2],result_panel[3])
