""" 
rules_python.py
Simple rule-based stroke predictor (IF-THEN rules).
"""
import pandas as pd
import numpy as np

def apply_rules(df_row):
    try:
        age = float(df_row.get('age', 0))
    except:
        age = 0.0
    hd = df_row.get('heart_disease', 0)
    ht = df_row.get('hypertension', 0)
    ag = df_row.get('avg_glucose_level', 0)
    bmi_v = df_row.get('bmi', 0)
    smoking = str(df_row.get('smoking_status','')).lower()
    ever_married = str(df_row.get('ever_married','')).lower()
    work_type = str(df_row.get('work_type','')).lower()
    gender = str(df_row.get('gender','')).lower()
    hd = 1 if str(hd).strip() in ['1','yes','true','True','Y','y'] else 0
    ht = 1 if str(ht).strip() in ['1','yes','true','True','Y','y'] else 0
    if age > 80 and hd==1:
        return 1
    if ag is not None and float(ag) > 200:
        return 1
    if ht==1 and hd==1:
        return 1
    if ('formerly' in smoking or 'smokes' in smoking) and age>60:
        return 1
    if bmi_v is not None and float(bmi_v) > 35 and age>65:
        return 1
    if 'yes' in ever_married and age>75:
        return 1
    if 'private' in work_type and ht==1 and age>60:
        return 1
    if 'male' in gender and hd==1 and age>70:
        return 1
    return 0
