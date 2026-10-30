import os
import pandas as pd
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.units import units

# import the functions defined in mixing height scripts
from mixing_height.GradientMethod import calculate_temperature_gradient, find_inversion_peaks, plot_gradient_profile
from mixing_height.holzworth_parcel import calculate_parcel_mixing_height, plot_holzworth_sounding 
from mixing_height.richardson_est import calculate_richardson_mixing_height  

def run_batch_analysis():
    start_day = 6
    end_day = 14
    
    # Column names as they appear in CSV file
    PRES = 'pressure_hPa'
    ALT  = 'geopotential height_m'
    TEMP = 'temperature_C'
    MIXR = 'mixing ratio_g/kg'
    WSPD = 'wind speed_m/s'
    WDIR = 'wind direction_degree'
    
    #Calculating new parameters, adding in CSV as new columns
    THT   = 'THT'    # Potential Temperature
    THT_V = 'THT_V'  # Virtual Potential Temperature
    
    results_list = []
    
    # Creating folders for results
    os.makedirs("plots/method1", exist_ok=True)
    os.makedirs("plots/method2", exist_ok=True)
    os.makedirs("plots/method3", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    print("🚀 Ξεκινάει η πλήρης αυτοματοποιημένη ανάλυση και για τις 3 μεθόδους (6–14 Αυγούστου)...")
    
    for day in range(start_day, end_day + 1):
        file_name = f"aug_{day:02d}.csv"
        file_path = os.path.join("data", file_name)
        
        if not os.path.exists(file_path):
            print(f"⚠️ Το αρχείο {file_path} δεν βρέθηκε. Συνεχίζουμε...")
            continue
            
        print(f"\n--- 📅 Επεξεργασία ημέρας: {day} Αυγούστου ({file_name}) ---")
        
        try:
            # 1. Loading data and clearing wrong values
            df = pd.read_csv(file_path, skipinitialspace=True)
            df.columns = df.columns.str.strip()
            
            # 2. Transforming data into floats for correct use
            df[PRES] = pd.to_numeric(df[PRES], errors='coerce')
            df[ALT]  = pd.to_numeric(df[ALT], errors='coerce')
            df[TEMP] = pd.to_numeric(df[TEMP], errors='coerce')
            df[MIXR] = pd.to_numeric(df[MIXR], errors='coerce')
            if WSPD in df.columns:
                df[WSPD] = pd.to_numeric(df[WSPD], errors='coerce')
            if WDIR in df.columns:
                df[WDIR] = pd.to_numeric(df[WDIR], errors='coerce')
            
            # 3. Using metpy for calculations, it need units
            p = df[PRES].values * units.hPa
            T = df[TEMP].values * units.degC
            r = df[MIXR].values * units('g/kg')
            
            df[THT]   = mpcalc.potential_temperature(p, T).m
            df[THT_V] = mpcalc.virtual_potential_temperature(p, T, r).m
            
            # Saving new df dataframe into CSV with the new columns
            df.to_csv(file_path, index=False)
            
            # =================================================================
            # --- ΜΕΘΟΔΟΣ 1: Μέγιστη Βαθμίδα Θερμοκρασίας (Gradient Method) ---
            # Inputs: file_path, Ύψος (ALT), Εικονική Δυνητική Θερμοκρασία (THT_V)
            # =================================================================
            df_res1 = calculate_temperature_gradient(file_path, ALT, THT_V)
            loc_h1, loc_g1, glob_h1, glob_g1 = find_inversion_peaks(df_res1, ALT, 'gradient')
            
            # Plot Μέθοδος 1
            plot_gradient_profile(df_res1, ALT, 'gradient', loc_h1, loc_g1, glob_h1, glob_g1, title=f"Method 1 (Gradient) - Aug {day:02d}" )
            plt.title(f"Method 1 (Gradient) - Aug {day:02d}")
            plt.savefig(f"plots/method1/gradient_aug_{day:02d}.png")
            plt.close()
            
            # =================================================================
            # --- ΜΕΘΟΔΟΣ 2: Αέριο Πακέτο (Parcel Method) ---
            # Inputs: Ύψος (ALT), Εικονική Δυνητική Θερμοκρασία (THT_V)
            # =================================================================
            df_res2, zi_parcel = calculate_parcel_mixing_height(file_path, ALT, THT_V)
            plot_holzworth_sounding(df_res2, ALT, THT_V, zi_parcel, title = f"Method 2 (Holzworth) - Aug {day:02d}" )
            plt.title(f"Method 2 (Parcel) - Aug {day:02d}")
            plt.savefig(f"plots/method2/parcel_aug_{day:02d}.png")
            plt.close()
            
            # =================================================================
            # --- ΜΕΘΟΔΟΣ 3: Virtual Bulk Richardson (Ri_b Method) ---
            # Inputs: Ύψος (ALT), Εικονική Δυνητική Θερμοκρασία (THT_V), Άνεμος (WSPD)
            # =================================================================
            results_ri, val_method3, ri_status = calculate_richardson_mixing_height(
            file_path, ALT, WSPD, THT_V, target_ri=0.25)
        
        # Printing status messages
            if ri_status != "Success":
             print(f"  ⚠️ Aug {day:02d} BulkRi: {ri_status}")
            else:
             print(f"  ✅ Aug {day:02d} BulkRi Height: {val_method3:.1f} m")
            
            
            # Saving per day results into a matrix
            results_list.append({
                'Day': f"Aug {day:02d}",
                'Method1_Gradient_H (m)': glob_h1,
                'Method2_Parcel_H (m)': zi_parcel,
                'Method3_BulkRi_H (m)': val_method3
            })
            
            print(f"  ✅ Ημέρα {day} Αυγούστου: Υπολογίστηκαν και οι 3 μέθοδοι επιτυχώς.")
            
        except Exception as e:
           print(f"  ❌ Σφάλμα στην ημέρα {day}: {e}")

            
    # Final results 
    if results_list:
        summary_df = pd.DataFrame(results_list)
        print("\n📊 ΤΕΛΙΚΟΣ ΣΥΓΚΕΝΤΡΩΤΙΚΟΣ ΠΙΝΑΚΑΣ (6–14 Αυγούστου - Όλες οι Μέθοδοι):")
        print("=" * 75)
        print(summary_df.to_string(index=False))
        print("=" * 75)
        
        # Save as CSV 
        summary_df.to_csv("results/all_methods_summary.csv", index=False)
        print("💾 Ο συγκεντρωτικός πίνακας αποθηκεύτηκε στο results/all_methods_summary.csv")

if __name__ == "__main__":
    run_batch_analysis()
