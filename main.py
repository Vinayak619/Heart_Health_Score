import pandas as pd
import numpy as np

node_weights = {
    "Age": 0.8,
    "Gender": 0.5,
    "Family history of heart disease": 0.7,
    "Physical Activity": 0.6,
    "Diet (High Salt)": 0.7,
    "Diet (High Sugar)": 0.7,
    "Diet (High Saturated and Trans Fats)": 0.8,
    "Diet (Fiber)": 0.6,
    "Diet (Healthy Fats - Omega-3)": 0.6,
    "Diet (Antioxidants)": 0.5,
    "Smoking": 0.9,
    "Alcohol Consumption": 0.5,
    "Obesity": 0.8,
    "Body Mass Index (BMI)": 0.6,
    "Stress": 0.5,
    "Blood Pressure (Hypertension)": 0.9,
    "Cholesterol Levels (LDL, HDL, Triglycerides)": 0.8,
    "Blood Sugar and Diabetes": 0.8,
    "Insulin Resistance": 0.7,
    "Inflammation (CRP, Homocysteine)": 0.6,
    "Heart Rate (Arrhythmias)": 0.6,
    "Coronary Artery Disease (CAD)": 1.0,
    "Cardiomyopathy (Dilated, Hypertrophic, Restrictive)": 0.8,
    "Sleep Apnea": 0.7,
    "Hormonal Imbalances (Thyroid, Cortisol, Estrogen)": 0.7,
    "Air Pollution": 0.5,
    "Second-hand Smoke": 0.6,
    "Socioeconomic Status": 0.4,
    "Chronic Kidney Disease": 0.8,
    "Chronic Inflammation (Rheumatoid Arthritis, Lupus, etc.)": 0.7,
    "Steroids (Medications)": 0.6,
    "NSAIDs (Medications)": 0.5,
    "Antihypertensive Medications": 0.5,
    "Stent Placement (Post-Surgery)": 0.8
}

column_aliases = {
    "age": "Age",
    "gender": "Gender",
    "family history": "Family history of heart disease",
    "physical activity": "Physical Activity",
    "diet high salt": "Diet (High Salt)",
    "diet high sugar": "Diet (High Sugar)",
    "diet high saturated and trans fats": "Diet (High Saturated and Trans Fats)",
    "diet fiber": "Diet (Fiber)",
    "diet healthy fats": "Diet (Healthy Fats - Omega-3)",
    "diet antioxidants": "Diet (Antioxidants)",
    "smoking": "Smoking",
    "alcohol consumption": "Alcohol Consumption",
    "obesity": "Obesity",
    "bmi": "Body Mass Index (BMI)",
    "stress": "Stress",
    "blood pressure": "Blood Pressure (Hypertension)",
    "cholesterol": "Cholesterol Levels (LDL, HDL, Triglycerides)",
    "blood sugar": "Blood Sugar and Diabetes",
    "insulin resistance": "Insulin Resistance",
    "crp": "Inflammation (CRP, Homocysteine)",
    "heart rate": "Heart Rate (Arrhythmias)",
    "cad history": "Coronary Artery Disease (CAD)",
    "cardiomyopathy": "Cardiomyopathy (Dilated, Hypertrophic, Restrictive)",
    "sleep apnea": "Sleep Apnea",
    "hormonal imbalances": "Hormonal Imbalances (Thyroid, Cortisol, Estrogen)",
    "air pollution": "Air Pollution",
    "second-hand smoke": "Second-hand Smoke",
    "socioeconomic status": "Socioeconomic Status",
    "chronic kidney disease": "Chronic Kidney Disease",
    "chronic inflammation": "Chronic Inflammation (Rheumatoid Arthritis, Lupus, etc.)",
    "steroids": "Steroids (Medications)",
    "nsaids": "NSAIDs (Medications)",
    "antihypertensive meds": "Antihypertensive Medications",
    "stent placement": "Stent Placement (Post-Surgery)"
}

binary_mapping = {
    "Gender": {"male":1,"m":1,"female":0,"f":0},
    "Smoking": None,  
    "Alcohol Consumption": None,  
    "Obesity": {"yes":1,"y":1,"no":0,"n":0},
    "Family history of heart disease": {"yes":1,"y":1,"no":0,"n":0},
    "Sleep Apnea": {"yes":1,"y":1,"no":0,"n":0},
    "Coronary Artery Disease (CAD)": {"yes":1,"y":1,"no":0,"n":0},
    "Cardiomyopathy (Dilated, Hypertrophic, Restrictive)": {"yes":1,"y":1,"no":0,"n":0},
    "Chronic Kidney Disease": {"yes":1,"y":1,"no":0,"n":0},
    "Chronic Inflammation (Rheumatoid Arthritis, Lupus, etc.)": {"yes":1,"y":1,"no":0,"n":0},
    "Hormonal Imbalances (Thyroid, Cortisol, Estrogen)": {"yes":1,"y":1,"no":0,"n":0},
    "Second-hand Smoke": {"yes":1,"y":1,"no":0,"n":0},
    "Steroids (Medications)": {"yes":1,"y":1,"no":0,"n":0},
    "NSAIDs (Medications)": {"yes":1,"y":1,"no":0,"n":0},
    "Antihypertensive Medications": {"yes":1,"y":1,"no":0,"n":0},
    "Stent Placement (Post-Surgery)": {"yes":1,"y":1,"no":0,"n":0},
    "Stress": None,
    "Diet (High Salt)": None,
    "Diet (High Sugar)": None,
    "Diet (High Saturated and Trans Fats)": None,
    "Diet (Fiber)": None,
    "Diet (Healthy Fats - Omega-3)": None,
    "Diet (Antioxidants)": None
}

numeric_ranges = {
    "Age": (0,120),
    "Physical Activity": (0,24),
    "Body Mass Index (BMI)": (10,50),
    "Blood Pressure (Hypertension)": (60,300),
    "Cholesterol Levels (LDL, HDL, Triglycerides)": (100,300),
    "Blood Sugar and Diabetes": (50,400),
    "Insulin Resistance": (0,10),
    "Inflammation (CRP, Homocysteine)": (0,20),
    "Heart Rate (Arrhythmias)": (30,200),
    "Socioeconomic Status": (0,10),
    "Air Pollution": (0,300),
    "Smoking": (0,50),
    "Alcohol Consumption": (0,50)
}

def normalize_and_map(df):
    df = df.copy()
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_object_dtype(s):
            s = s.astype(str).str.strip().str.lower()
        # Binary mapping
        if col in binary_mapping and binary_mapping[col] is not None:
            s = s.replace(binary_mapping[col])
        # Blood Pressure parsing
        if col == "Blood Pressure (Hypertension)":
            def parse_bp(x):
                try:
                    if '/' in str(x): x = str(x).split('/')[0]
                    val = float(x)
                    low, high = numeric_ranges[col]
                    if val < low or val > high: return np.nan
                    return (val - low)/(high-low)
                except: return np.nan
            s = s.apply(parse_bp)
        # Numeric normalization
        elif col in numeric_ranges and col != "Blood Pressure (Hypertension)":
            low, high = numeric_ranges[col]
            def norm_val(x):
                try:
                    val = float(x)
                    if val < low or val > high: return np.nan
                    return (val - low)/(high-low)
                except: return np.nan
            s = s.apply(norm_val)
        df[col] = s
    return df

def compute_heart_health_score(row):
    weighted_sum = 0
    total_weight_used = 0
    total_weight_all = sum(node_weights.values())
    
    for col, weight in node_weights.items():
        val = row.get(col,np.nan)
        if pd.isna(val): continue
        weighted_sum += val * weight
        total_weight_used += weight
    
    if total_weight_used==0: 
        return 0,0

    raw_score = weighted_sum / total_weight_used  
    scaling_factor = 0.85   
    score = raw_score * 100 * scaling_factor
    score = round(min(max(score,0),100),2)
    
    confidence = round((total_weight_used/total_weight_all)*100,2)
    
    return score, confidence


def get_risk_category(score):
    if score<=30: return "Healthy"
    elif score<=60: return "Moderate"
    else: return "High Risk"

def main():
    print("\nHeart Health Risk Predictor\n")
    file_path = input("Enter your CSV file path: ").strip()
    try: df = pd.read_csv(file_path)
    except Exception as e:
        print("CSV read error:", e)
        return
    df.columns = df.columns.str.strip().str.lower()
    df.rename(columns=column_aliases,inplace=True)
    print(f"\nTotal patients: {len(df)}")
    try:
        patient_no = int(input("Enter patient number (starting from 1): "))-1
    except:
        print("Invalid input"); return
    if patient_no<0 or patient_no>=len(df):
        print("Patient number out of range"); return

    patient_data = df.iloc[[patient_no]]
    print(f"\nSelected Patient Data (Patient {patient_no + 1}):")
    print(patient_data.transpose())
    print("\nNormalizing and analyzing factors...")
    df_norm = ++normalize_and_map(df)
    row = df_norm.iloc[patient_no]

    missing_cols = [col for col in node_weights.keys() if pd.isna(row.get(col))]
    score, confidence = compute_heart_health_score(row)

    print("\nHeart Health Score:", score,"/ 100")
    print("Risk Category:", get_risk_category(score))
    if missing_cols:
        print("\nMissing Values Detected in the following factors:")
        for col in missing_cols: print(" -",col)
        print(f"Confidence based on available factors: ~{confidence}%")
    else:
        print("\nAll factors available. Confidence: 100%")
    print("\nAnalysis Complete.\n")

if __name__=="__main__":
    main()

