# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:08:12 2020

@author: Mario
"""



import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import itertools


def ct_advantage(alive):
    #SPLITING ALIVES PLAYER INTO TEAMS
    ct = list(filter(lambda x: x['team']=='CT',(alive[i] for i in range(len(alive)))))
    tt = list(filter(lambda x: x['team']=='Terrorist',(alive[i] for i in range(len(alive)))))
    
    #SUMING HP, ARMOR, UTILITES, WEAPON VALUE FOR CT AND TT
    ct_hp = sum(map(lambda x: x['health'],ct))
    ct_armor = sum(map(lambda x: x['armor'],ct))
    ct_has_defuser = int(sum(map(lambda x: x['has_defuser'],ct))>0)
    ct_helmets = sum(map(lambda x: x['has_helmet'],ct))
    ### weapons
    ct_weapons = list(map(lambda x: x['inventory'],ct))
    ct_weapons = list(itertools.chain(*ct_weapons))
    ct_weapons_worth = 0
    for weapon in ct_weapons:
        if weapon['clip_ammo'] + weapon['reserve_ammo']>0:            
            ct_weapons_worth += weapon_dict(weapon['item_type'])
                
    ct_nades = list(filter(lambda x: weapon_dict(x['item_type']) == 0,ct_weapons))
    ct_nades_ammount = sum(map(lambda x:x['clip_ammo'],ct_nades)) 
    
    tt_hp = sum(map(lambda x: x['health'],tt))
    tt_armor = sum(map(lambda x: x['armor'],tt))
    tt_helmets = sum(map(lambda x: x['has_helmet'],tt))
    tt_weapons = list(map(lambda x: x['inventory'],tt))
    tt_weapons = list(itertools.chain(*tt_weapons))
    tt_weapons_worth = 0
    for weapon in tt_weapons:
        if weapon['clip_ammo'] + weapon['reserve_ammo']>0:            
            tt_weapons_worth += weapon_dict(weapon['item_type'])
                
    tt_nades = list(filter(lambda x: weapon_dict(x['item_type']) == 0,tt_weapons))
    tt_nades_ammount = sum(map(lambda x:x['clip_ammo'],tt_nades)) 
    
    # COMPARING HP, ARMOR, WEAPONS VALUE, NADES
    ct_advantage_health = np.sign(ct_hp - tt_hp)
    ct_advantage_armor = np.sign((ct_armor + 30 * ct_helmets) - (tt_armor + 30 * tt_helmets))
    ct_advantage_weapons = np.sign(ct_weapons_worth - tt_weapons_worth)
    ct_advantage_nades = np.sign(ct_nades_ammount - tt_nades_ammount)
    return [ct_advantage_health,ct_advantage_armor,ct_advantage_weapons,ct_advantage_nades,ct_has_defuser]


def weapon_dict(weapon):
    d = {#PISTOLS
        "P2000":200,"UspS":200,"Glock":200,"P250":300,"Elite":400,
         "FiveSeven":500,"Tec9":500,"Cz75Auto":500,"R8Revolver":600,"Deagle":700,
         #SMG
         "Mp5sd":1500,"Mp9":1200,"Mac10":1050,"P90":2350,"Ump45":1200,"Mp7":1500,"Bizon":1600,
         #SHOTGUNS
         "Nova":1050,"Sawedoff":1100,"Mag7":1300,"Xm1014":2000,
         #HEAVY
         "M249":1500,"Negev":1700,
         #RIFTLES
         "Famas":2050,"GalilAr":1800,"Ak47":2700,"Aug":3300,"Sg553":3000,"M4a1S":2900,
         #SNIPERS
         "Ssg08":1700,"Awp":4900,"Scar20":5000,"G3sg1":5000,
         #OTHERS
         "ZeusX27":200,"C4":1 }
    if weapon in d.keys():
        return d[weapon]
    else:
        return 0




    
def prep_data(data):   
    df = pd.DataFrame().from_dict(data)    
    df=df.drop(columns=['map','current_score','previous_kills'],axis=1) #DROPING IRRELEVANT COLS
    planted_bomb = []
    for d in data:
        planted_bomb.append(1 if d['planted_bomb'] is not None else 0) #DROPING BOMB LOCATION
    df  = df.drop(columns=['planted_bomb'],axis=1)
    df.insert(loc=len(df.columns),column='planted_bomb',value=planted_bomb)
    
    #CHANGING WINNER VALUE FROM STRING TO LOGICAL
    df.round_winner = LabelBinarizer().fit_transform(df.round_winner) 
    df.round_winner = 1-df.round_winner
    
    #DROPING DATA FROM FREEZETIME ROUNDS
    df=df[df['round_status'] != "FreezeTime"]  
    df = df.drop(columns=['round_status'],axis=1)
    
    #DROPING NADES LOCATION, NOT USED ATM
    df = df.drop(columns=['active_molotovs','active_smokes'],axis=1)
    
    #COMPUTING WHO HAS ADVANTAGE AND INSERTING DATA
    df.alive_players = df.alive_players.map(ct_advantage)    
    df.insert(loc=len(df.columns),column='ct_advantage_health',value=df.alive_players.map(lambda x:x[0]))
    df.insert(loc=len(df.columns),column='ct_advantage_armor',value=df.alive_players.map(lambda x:x[1]))
    df.insert(loc=len(df.columns),column='ct_advantage_weapons',value=df.alive_players.map(lambda x:x[2]))
    df.insert(loc=len(df.columns),column='ct_advantage_nades',value=df.alive_players.map(lambda x:x[3]))
    df.insert(loc=len(df.columns),column='ct_has_defuser',value=df.alive_players.map(lambda x:x[4]))
    df = df.drop(columns=['alive_players'])
    
    #INSERTING COLS BASED ON ROUND TIME LEFT
    df['is_time_gt_20'] = df['round_status_time_left']>20
    df.is_time_gt_20 = LabelBinarizer().fit_transform(df.is_time_gt_20)
    df['is_no_time_to_defuse'] = df['planted_bomb'] * (df['round_status_time_left']<(10-5*df['ct_has_defuser']))    
    df=df.drop(columns=['round_status_time_left'],axis=1)
    
    return df


data = []
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4) #XGB MODEL
X_test = pd.DataFrame()
Y_test = pd.DataFrame()

#READING ALL 24 FILES
files = list('0'+str(x) for x in range(0,25))
files = list(x[-2:] for x in files)
for f in files:
    name = 'dataset_'+f+'.json'
    with open(name) as json_file:
        data = json.load(json_file)
    df = prep_data(data)
    
    ### LEARNING
    Y = df.round_winner
    df = df.drop(columns=['round_winner'],axis=1)
    X = df
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, train_size=0.95, test_size=0.05,    
    random_state=0)    
    if len(X_test)>0:        
        X_test = pd.concat([X_test,X_valid],ignore_index=True)
        Y_test = pd.concat([Y_test,y_valid],ignore_index=True)
    else:
        X_test = X_valid
        Y_test = y_valid
        
    my_model.fit(X_train, y_train, 
                  early_stopping_rounds=5, 
                  eval_set=[(X_valid, y_valid)], 
                  verbose=False)
#PREDICT
predictions = my_model.predict(X_test)>0.5
hit = Y_test - predictions
acc = len(hit[hit ==  0])/len(hit)
print('Acc: ',round(acc,4)*100,"%")