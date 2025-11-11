import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
from datetime import datetime

# NODE WEIGHTS
node_weights = {
    "Age": 0.8, "Gender": 0.5, "Family history of heart disease": 0.7,
    "Physical Activity": 0.6, "Diet (High Salt)": 0.7, "Diet (High Sugar)": 0.7,
    "Diet (High Saturated and Trans Fats)": 0.8, "Diet (Fiber)": 0.6,
    "Diet (Healthy Fats - Omega-3)": 0.6, "Diet (Antioxidants)": 0.5,
    "Smoking": 0.9, "Alcohol Consumption": 0.5, "Obesity": 0.8,
    "Body Mass Index (BMI)": 0.6, "Stress": 0.5, "Blood Pressure (Hypertension)": 0.9,
    "Cholesterol Levels (LDL, HDL, Triglycerides)": 0.8, "Blood Sugar and Diabetes": 0.8,
    "Insulin Resistance": 0.7, "Inflammation (CRP, Homocysteine)": 0.6,
    "Heart Rate (Arrhythmias)": 0.6, "Coronary Artery Disease (CAD)": 1.0,
    "Cardiomyopathy (Dilated, Hypertrophic, Restrictive)": 0.8, "Sleep Apnea": 0.7,
    "Hormonal Imbalances (Thyroid, Cortisol, Estrogen)": 0.7, "Air Pollution": 0.5,
    "Second-hand Smoke": 0.6, "Socioeconomic Status": 0.4, "Chronic Kidney Disease": 0.8,
    "Chronic Inflammation (Rheumatoid Arthritis, Lupus, etc.)": 0.7, "Steroids (Medications)": 0.6,
    "NSAIDs (Medications)": 0.5, "Antihypertensive Medications": 0.5, "Stent Placement (Post-Surgery)": 0.8
}

# Column aliases for CSV normalization
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

numeric_ranges = {
    "Age": (0,120), "Physical Activity": (0,24), "Body Mass Index (BMI)": (10,50),
    "Blood Pressure (Hypertension)": (60,300),
    "Cholesterol Levels (LDL, HDL, Triglycerides)": (100,300),
    "Blood Sugar and Diabetes": (50,400),
    "Insulin Resistance": (0,10), "Inflammation (CRP, Homocysteine)": (0,20),
    "Heart Rate (Arrhythmias)": (30,200), "Socioeconomic Status": (0,10),
    "Air Pollution": (0,300), "Smoking": (0,50), "Alcohol Consumption": (0,50)
}

# CORE FUNCTIONS
def normalize_value(col, val):
    if val == "" or val is None:
        return np.nan
    try:
        if col == "Blood Pressure (Hypertension)":
            if "/" in str(val): val = str(val).split("/")[0]
        val = float(val)
        if col in numeric_ranges:
            low, high = numeric_ranges[col]
            if val < low or val > high:
                return np.nan
            return (val - low) / (high - low)
        return val
    except:
        return np.nan

def normalize_row(row):
    norm_row = {}
    for col, val in row.items():
        if col in node_weights:
            norm_row[col] = normalize_value(col, val)
    return pd.Series(norm_row)

def compute_heart_health_score(row):
    weighted_sum, total_weight_used = 0, 0
    total_weight_all = sum(node_weights.values())
    for col, weight in node_weights.items():
        val = row.get(col, np.nan)
        if pd.isna(val): 
            continue
        weighted_sum += val * weight
        total_weight_used += weight
    if total_weight_used == 0:
        return 0, 0
    raw_score = weighted_sum / total_weight_used
    score = min(max(raw_score * 100 * 0.85, 0), 100)
    confidence = (total_weight_used / total_weight_all) * 100
    return round(score, 2), round(confidence, 2)

def get_risk_category(score):
    if score <= 30: return "Healthy 🟢"
    elif score <= 60: return "Moderate 🟡"
    else: return "High Risk 🔴"

def get_confidence_label(conf):
    if conf < 40: return f"{conf}% → Low (Incomplete Data)"
    elif conf < 70: return f"{conf}% → Moderate (Partial Data)"
    else: return f"{conf}% → High (Reliable)"

# GUI FUNCTIONS
def start_manual_mode():
    start_frame.pack_forget()
    manual_frame.pack(fill="both", expand=True)

def start_csv_mode():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        return
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower()
        df.rename(columns=column_aliases, inplace=True)

        total = len(df)
        if total == 0:
            messagebox.showerror("Error", "CSV is empty.")
            return
        patient_no = simpledialog.askinteger("Select Patient", f"Total patients: {total}\nEnter patient number (1–{total}):")
        if not patient_no or patient_no < 1 or patient_no > total:
            messagebox.showerror("Error", "Invalid patient number.")
            return
        
        row = df.iloc[patient_no - 1]

        # Normalize after alias mapping
        norm_row = normalize_row(row)

        score, conf = compute_heart_health_score(norm_row)
        risk = get_risk_category(score)
        conf_text = get_confidence_label(conf)
        name = row.get("Name", f"Patient {patient_no}")

        csv_result_label.config(
            text=f"Name: {name}\nHeart Health Score: {score}/100\nRisk: {risk}\nConfidence: {conf_text}"
        )
        start_frame.pack_forget()
        csv_frame.pack(fill="both", expand=True)
    except Exception as e:
        messagebox.showerror("Error", f"CSV read failed: {e}")

def calculate_score():
    name = name_entry.get().strip() or "Unknown"
    data = {}
    for col, entry in entries.items():
        data[col] = normalize_value(col, entry.get())
    row = pd.Series(data)
    score, conf = compute_heart_health_score(row)
    risk = get_risk_category(score)
    conf_text = get_confidence_label(conf)
    result_label.config(text=f"Name: {name}\nScore: {score}/100\nRisk: {risk}\nConfidence: {conf_text}")
    global last_result
    last_result = (name, data, score, risk, conf_text)

def save_result():
    if not last_result:
        messagebox.showerror("Error", "Please compute result first.")
        return
    name, data, score, risk, conf = last_result
    filename = f"Heart_Result_{name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w") as f:
        f.write(f"Patient Name: {name}\n")
        f.write(f"Heart Health Score: {score}/100\n")
        f.write(f"Risk Category: {risk}\n")
        f.write(f"Confidence: {conf}\n\nUser Input Data:\n")
        for k, v in data.items():
            f.write(f"{k}: {v}\n")
    messagebox.showinfo("Saved", f"✅ Result saved as {filename}")

def reset_form():
    name_entry.delete(0, tk.END)
    for e in entries.values():
        e.delete(0, tk.END)
    result_label.config(text="")
    global last_result
    last_result = None

# GUI LAYOUT
root = tk.Tk()
root.title("🫀 Heart Health Risk Predictor (Local GUI)")
root.geometry("950x750")
root.configure(bg="#eef2f3")

start_frame = tk.Frame(root, bg="#eef2f3")
tk.Label(start_frame, text="Heart Health Risk Predictor", font=("Helvetica", 22, "bold"), bg="#eef2f3", fg="#1b4965").pack(pady=40)
tk.Label(start_frame, text="Choose input method:", font=("Helvetica", 14), bg="#eef2f3").pack(pady=10)
tk.Button(start_frame, text="🗂 Load from CSV", bg="#2e86de", fg="white", font=("Helvetica", 13), width=20, command=start_csv_mode).pack(pady=10)
tk.Button(start_frame, text="🧍 Manual Input", bg="#00b894", fg="white", font=("Helvetica", 13), width=20, command=start_manual_mode).pack(pady=10)
start_frame.pack(fill="both", expand=True)

# Manual mode same as before
manual_frame = tk.Frame(root, bg="#eef2f3")
canvas = tk.Canvas(manual_frame, bg="#eef2f3", highlightthickness=0)
scrollbar = ttk.Scrollbar(manual_frame, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)
scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((475, 0), window=scrollable_frame, anchor="n")
canvas.configure(yscrollcommand=scrollbar.set)

tk.Label(scrollable_frame, text="Manual Data Entry", font=("Helvetica", 18, "bold"), bg="#eef2f3", fg="#1b4965").pack(pady=10)
tk.Label(scrollable_frame, text="Fill patient details (Press Enter to skip any):", font=("Helvetica", 11), bg="#eef2f3").pack()

tk.Label(scrollable_frame, text="Name:", bg="#eef2f3").pack()
name_entry = tk.Entry(scrollable_frame, width=40)
name_entry.pack(pady=4)

entries = {}
for col in node_weights.keys():
    range_text = f" ({numeric_ranges[col][0]}–{numeric_ranges[col][1]})" if col in numeric_ranges else ""
    tk.Label(scrollable_frame, text=f"{col}{range_text}:", bg="#eef2f3").pack()
    e = tk.Entry(scrollable_frame, width=35)
    e.pack(pady=3)
    entries[col] = e

tk.Button(scrollable_frame, text="Compute Result", bg="#00b894", fg="white", font=("Helvetica", 12), command=calculate_score).pack(pady=10)
result_label = tk.Label(scrollable_frame, text="", font=("Helvetica", 11), bg="#eef2f3", justify="left")
result_label.pack(pady=10)

btn_frame = tk.Frame(scrollable_frame, bg="#eef2f3")
btn_frame.pack(pady=5)
tk.Button(btn_frame, text="💾 Save Result", bg="#2e86de", fg="white", font=("Helvetica", 11), width=15, command=save_result).pack(side="left", padx=5)
tk.Button(btn_frame, text="🔄 Reset", bg="#d63031", fg="white", font=("Helvetica", 11), width=15, command=reset_form).pack(side="left", padx=5)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

csv_frame = tk.Frame(root, bg="#eef2f3")
tk.Label(csv_frame, text="CSV Mode Result", font=("Helvetica", 18, "bold"), bg="#eef2f3", fg="#1b4965").pack(pady=20)
csv_result_label = tk.Label(csv_frame, text="", font=("Helvetica", 12), bg="#eef2f3", justify="left")
csv_result_label.pack(pady=20)
tk.Button(csv_frame, text="🏠 Back to Home", bg="#00b894", fg="white", font=("Helvetica", 12), command=lambda: [csv_frame.pack_forget(), start_frame.pack(fill="both", expand=True)]).pack(pady=15)

last_result = None
root.mainloop()
