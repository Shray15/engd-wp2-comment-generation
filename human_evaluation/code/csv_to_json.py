import json
import pandas as pd
base = r"C:\Users\20245179\OneDrive - TU Eindhoven\WP2_Simulation_Tool\human_evaluation\subsets"

for i in range(1, 8):
    df = pd.read_csv(f"{base}\\annotator_{i:02d}.csv")
    json_str = json.dumps(df.to_dict(orient='records'), ensure_ascii=False, indent=2)
    
    with open(f"{base}\\annotator_{i:02d}.json", 'w', encoding='utf-8') as f:
        f.write(json_str)
    