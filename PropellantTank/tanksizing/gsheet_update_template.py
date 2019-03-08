import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']

credentials = ServiceAccountCredentials.from_json_keyfile_name('My Project-d77f51940158.json', scope)
gc = gspread.authorize(credentials)
wb = gc.open('ISTDN-RD-RP-TC-OR-2018-006C_ZERO 基本設計段階 質量計画値')
sh = wb.worksheet("tankout")


#出力ファイル書き込み
f = open('tankout_S1.out')
line = f.readline()
words = line.split()

while line:
    if line == "\n":
        print ("Empty")
    else:
        for i in range(35):
            sh.update_cell(i+1,1,words[0])
            sh.update_cell(i+1,4,words[2])
            line = f.readline()
            words = line.split()

f.close


