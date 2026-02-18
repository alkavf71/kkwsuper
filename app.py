import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter

# ============================================================================
# KONFIGURASI GLOBAL
# ============================================================================

ISO_LIMITS_VELOCITY = {
    "Zone A (Good)": 2.8,
    "Zone B (Acceptable)": 4.5,
    "Zone C (Unacceptable)": 7.1,
    "Zone D (Danger)": 11.0
}

ACCEL_BASELINE = {
    "Band1 (0.5-1.5kHz)": 0.3,
    "Band2 (1.5-5kHz)": 0.2,
    "Band3 (5-16kHz)": 0.15
}

FLUID_PROPERTIES = {
    "Pertalite (RON 90)": {"sg": 0.73, "vapor_pressure_kpa_38C": 52, "viscosity_cst_40C": 0.6, "risk_level": "High"},
    "Pertamax (RON 92)": {"sg": 0.74, "vapor_pressure_kpa_38C": 42, "viscosity_cst_40C": 0.6, "risk_level": "High"},
    "Diesel / Solar": {"sg": 0.84, "vapor_pressure_kpa_38C": 0.5, "viscosity_cst_40C": 3.0, "risk_level": "Moderate"}
}

ELECTRICAL_LIMITS = {
    "voltage_unbalance_warning": 2.0,
    "voltage_unbalance_critical": 3.5,
    "current_unbalance_warning": 5.0,
    "current_unbalance_critical": 10.0,
    "voltage_tolerance_low": 90,
    "voltage_tolerance_high": 110
}

# ============================================================================
# FUNGSI BANTUAN (Ringkas - Expand sesuai kebutuhan)
# ============================================================================

def get_mechanical_recommendation(diagnosis, location, severity="Medium"):
    return f"🔧 {location} - {diagnosis}\n• Severity: {severity}\n• Action: Jadwalkan inspection"

def get_hydraulic_recommendation(diagnosis, fluid_type, severity="Medium"):
    return f"💧 {fluid_type} - {diagnosis}\n• Severity: {severity}\n• Action: Monitor intensif"

def get_electrical_recommendation(diagnosis, severity="Medium"):
    return f"⚡ {diagnosis}\n• Severity: {severity}\n• Action: Verifikasi supply"

def calculate_hydraulic_parameters(suction, discharge, flow, power, sg):
    delta_p = discharge - suction
    head = delta_p * 10.2 / sg if sg > 0 else 0
    hyd_power = (flow * head * sg * 9.81) / 3600 if flow > 0 and head > 0 else 0
    efficiency = (hyd_power / power * 100) if power > 0 else 0
    return {"head_m": head, "efficiency_percent": efficiency, "delta_p_bar": delta_p}

def calculate_electrical_parameters(v1, v2, v3, i1, i2, i3, pf, rated_v, rated_i):
    v_avg = (v1 + v2 + v3) / 3
    i_avg = (i1 + i2 + i3) / 3
    v_dev = [abs(v - v_avg) for v in [v1, v2, v3]]
    i_dev = [abs(i - i_avg) for i in [i1, i2, i3]]
    v_unbal = (max(v_dev) / v_avg * 100) if v_avg > 0 else 0
    i_unbal = (max(i_dev) / i_avg * 100) if i_avg > 0 else 0
    load_est = (i_avg / rated_i * 100) if rated_i > 0 else 0
    v_in_tol = (ELECTRICAL_LIMITS["voltage_tolerance_low"] <= (v_avg/rated_v*100) <= ELECTRICAL_LIMITS["voltage_tolerance_high"])
    return {"voltage_unbalance_percent": v_unbal, "current_unbalance_percent": i_unbal, 
            "load_estimate_percent": load_est, "voltage_within_tolerance": v_in_tol, "v_avg": v_avg}

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(page_title="Pump Diagnostic Expert System", layout="wide")
    
    # ===== INISIALISASI SESSION STATE (Defensive) =====
    if "shared_context" not in st.session_state:
        st.session_state.shared_context = {
            "machine_id": "P-101",
            "rpm": 2950,
            "service_criticality": "Essential (Utility)",
            "fluid_type": "Diesel / Solar"
        }
    
    # Header
    st.title("🔧💧⚡ Pump Diagnostic Expert System")
    st.caption("Integrated Mechanical • Hydraulic • Electrical | Pertamina Patra Niaga")
    
    # ===== SIDEBAR: Shared Context =====
    with st.sidebar:
        st.subheader("📍 Shared Context")
        
        machine_id = st.text_input("Machine ID", value=st.session_state.shared_context["machine_id"])
        rpm = st.number_input("RPM", min_value=600, max_value=3600, value=st.session_state.shared_context["rpm"], step=10)
        service_type = st.selectbox("Criticality", ["Critical (Process)", "Essential (Utility)", "Standby"], 
                                   index=["Critical (Process)", "Essential (Utility)", "Standby"].index(st.session_state.shared_context["service_criticality"]))
        fluid_type = st.selectbox("Fluid Type (BBM)", list(FLUID_PROPERTIES.keys()),
                                 index=list(FLUID_PROPERTIES.keys()).index(st.session_state.shared_context["fluid_type"]))
        
        st.session_state.shared_context.update({
            "machine_id": machine_id, "rpm": rpm, 
            "service_criticality": service_type, "fluid_type": fluid_type
        })
        
        props = FLUID_PROPERTIES[fluid_type]
        st.info(f"**{fluid_type}**: SG={props['sg']} | Risk: {props['risk_level']}")
        st.divider()
        st.caption("💡 Gunakan tab di atas untuk navigasi")
    
    # ===== TAB NAVIGATION (DEFINISI SEBELUM PENGGUNAAN) =====
    # ⚠️ PENTING: Baris ini HARUS dieksekusi sebelum 'with tab_xxx:'
    tab_mech, tab_hyd, tab_elec, tab_integrated = st.tabs([
        "🔧 Mechanical", "💧 Hydraulic", "⚡ Electrical", "🔗 Integrated"
    ])
    
    # ========================================================================
    # TAB 1: MECHANICAL
    # ========================================================================
    with tab_mech:
        st.header("🔧 Mechanical Vibration")
        
        # Simple input demo
        col1, col2 = st.columns(2)
        with col1:
            vel_de_h = st.number_input("Pump DE Horizontal (mm/s)", 0.0, 20.0, 1.5, 0.1)
            vel_de_v = st.number_input("Pump DE Vertical (mm/s)", 0.0, 20.0, 1.8, 0.1)
        with col2:
            vel_nde_h = st.number_input("Pump NDE Horizontal (mm/s)", 0.0, 20.0, 1.2, 0.1)
            vel_nde_v = st.number_input("Pump NDE Vertical (mm/s)", 0.0, 20.0, 1.4, 0.1)
        
        if st.button("Analyze Mechanical", type="primary"):
            max_vel = max(vel_de_h, vel_de_v, vel_nde_h, vel_nde_v)
            if max_vel > ISO_LIMITS_VELOCITY["Zone B (Acceptable)"]:
                st.warning(f"⚠️ Velocity {max_vel:.1f} mm/s > 4.5 mm/s (Zone B)")
                st.info(get_mechanical_recommendation("UNBALANCE", "Pump DE", "Medium"))
            else:
                st.success("✅ All points within Zone A/B - Normal operation")
            
            # Store result for integration
            st.session_state.mech_result = {
                "diagnosis": "UNBALANCE" if max_vel > 4.5 else "Normal",
                "confidence": 85 if max_vel > 4.5 else 99,
                "severity": "Medium" if max_vel > 4.5 else "Low",
                "fault_type": "low_freq" if max_vel > 4.5 else "normal",
                "domain": "mechanical"
            }
    
    # ========================================================================
    # TAB 2: HYDRAULIC
    # ========================================================================
    with tab_hyd:
        st.header("💧 Hydraulic Troubleshooting")
        
        # Steady-state gate
        with st.expander("✅ Steady-State Verification", expanded=True):
            c1, c2 = st.columns(2)
            s1 = c1.checkbox("Stabil ≥15 menit", True)
            s2 = c1.checkbox("Pressure fluct <±2%", True)
            s3 = c2.checkbox("Flow fluct <±3%", True)
            s4 = c2.checkbox("Valve position fixed", True)
            steady_ok = all([s1, s2, s3, s4])
            if not steady_ok:
                st.warning("⚠️ Lengkapi verifikasi untuk analisis")
        
        # Primary inputs
        col1, col2, col3 = st.columns(3)
        with col1:
            suction = st.number_input("Suction (bar)", 0.0, 10.0, 1.2, 0.1)
            discharge = st.number_input("Discharge (bar)", 0.0, 20.0, 8.5, 0.1)
        with col2:
            flow = st.number_input("Flow (m³/h)", 0.0, 500.0, 100.0, 1.0)
            power = st.number_input("Motor Power (kW)", 0.0, 500.0, 45.0, 0.5)
        with col3:
            props = FLUID_PROPERTIES[fluid_type]
            sg = st.number_input("SG", 0.5, 1.5, props["sg"], 0.01)
            if discharge > suction and sg > 0:
                head = (discharge - suction) * 10.2 / sg
                st.metric("Head", f"{head:.1f} m")
        
        # Design reference
        with st.expander("📋 Design Reference"):
            c1, c2 = st.columns(2)
            q_design = c1.number_input("Q_design (m³/h)", 0.0, 500.0, 120.0)
            h_design = c1.number_input("H_design (m)", 0.0, 200.0, 85.0)
            eff_bep = c2.number_input("BEP Efficiency (%)", 0, 100, 78)
            npshr = c2.number_input("NPSHr (m)", 0.0, 20.0, 4.2)
        
        # Observations
        noise = st.radio("Noise Type", ["Normal", "Whining", "Grinding", "Crackling"], index=0)
        
        if st.button("Analyze Hydraulic", type="primary", disabled=not steady_ok):
            calc = calculate_hydraulic_parameters(suction, discharge, flow, power, sg)
            
            # Simple diagnosis logic
            dev_head = ((calc["head_m"] - h_design) / h_design * 100) if h_design > 0 else 0
            dev_eff = ((calc["efficiency_percent"] - eff_bep) / eff_bep * 100) if eff_bep > 0 else 0
            
            if noise == "Crackling" and dev_head < -5:
                diag, conf, sev = "CAVITATION", 85, "High"
            elif dev_head < -10:
                diag, conf, sev = "IMPELLER_WEAR", 75, "Medium"
            elif dev_eff < -15:
                diag, conf, sev = "EFFICIENCY_DROP", 70, "Medium"
            else:
                diag, conf, sev = "NORMAL_OPERATION", 95, "Low"
            
            st.session_state.hyd_result = {
                "diagnosis": diag, "confidence": conf, "severity": sev,
                "fault_type": "cavitation" if diag=="CAVITATION" else "wear" if diag=="IMPELLER_WEAR" else "normal",
                "domain": "hydraulic", "details": {"head_dev": dev_head, "eff_dev": dev_eff}
            }
            
            if diag != "NORMAL_OPERATION":
                st.warning(f"⚠️ {diag} ({conf}%)")
                st.info(get_hydraulic_recommendation(diag, fluid_type, sev))
            else:
                st.success("✅ Normal operation")
    
    # ========================================================================
    # TAB 3: ELECTRICAL
    # ========================================================================
    with tab_elec:
        st.header("⚡ Electrical Condition")
        
        # Motor specs
        with st.expander("⚙️ Motor Nameplate", expanded=True):
            c1, c2 = st.columns(2)
            rated_v = c1.number_input("Rated Voltage (V)", 200, 690, 400, 10)
            rated_i = c1.number_input("Rated Current (A)", 10, 500, 85, 5)
            rated_kw = c2.number_input("Rated Power (kW)", 1, 500, 45, 1)
            pf = c2.number_input("Power Factor", 0.5, 1.0, 0.85, 0.01)
        
        # 3-phase measurements
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption("Voltage (V)")
            v1 = st.number_input("L1-L2", 0.0, 700.0, 400.0, 1.0)
            v2 = st.number_input("L2-L3", 0.0, 700.0, 402.0, 1.0)
            v3 = st.number_input("L3-L1", 0.0, 700.0, 398.0, 1.0)
        with col2:
            st.caption("Current (A)")
            i1 = st.number_input("L1", 0.0, 500.0, 82.0, 0.5)
            i2 = st.number_input("L2", 0.0, 500.0, 84.0, 0.5)
            i3 = st.number_input("L3", 0.0, 500.0, 83.0, 0.5)
        with col3:
            st.caption("Status")
            calc_e = calculate_electrical_parameters(v1,v2,v3,i1,i2,i3,pf,rated_v,rated_i)
            st.metric("V Unbalance", f"{calc_e['voltage_unbalance_percent']:.1f}%")
            st.metric("I Unbalance", f"{calc_e['current_unbalance_percent']:.1f}%")
            st.metric("Load Est.", f"{calc_e['load_estimate_percent']:.0f}%")
        
        if st.button("Analyze Electrical", type="primary"):
            # Simple diagnosis
            v_unbal = calc_e["voltage_unbalance_percent"]
            i_unbal = calc_e["current_unbalance_percent"]
            
            if v_unbal > ELECTRICAL_LIMITS["voltage_unbalance_critical"]:
                diag, conf, sev = "VOLTAGE_UNBALANCE", 75, "High"
            elif v_unbal > ELECTRICAL_LIMITS["voltage_unbalance_warning"]:
                diag, conf, sev = "VOLTAGE_UNBALANCE", 65, "Medium"
            elif i_unbal > ELECTRICAL_LIMITS["current_unbalance_critical"]:
                diag, conf, sev = "CURRENT_UNBALANCE", 70, "High"
            else:
                diag, conf, sev = "NORMAL_ELECTRICAL", 95, "Low"
            
            st.session_state.elec_result = {
                "diagnosis": diag, "confidence": conf, "severity": sev,
                "fault_type": "voltage" if "VOLTAGE" in diag else "current" if "CURRENT" in diag else "normal",
                "domain": "electrical", "details": calc_e
            }
            
            if diag != "NORMAL_ELECTRICAL":
                st.warning(f"⚠️ {diag} ({conf}%)")
                st.info(get_electrical_recommendation(diag, sev))
            else:
                st.success("✅ Normal electrical condition")
    
    # ========================================================================
    # TAB 4: INTEGRATED SUMMARY
    # ========================================================================
    with tab_integrated:
        st.header("🔗 Integrated Summary")
        
        # Check completion
        completed = all([
            "mech_result" in st.session_state and st.session_state.mech_result,
            "hyd_result" in st.session_state and st.session_state.hyd_result,
            "elec_result" in st.session_state and st.session_state.elec_result
        ])
        
        if not completed:
            st.info("💡 Jalankan analisis di ketiga tab terlebih dahulu untuk integrated summary.")
            # Status indicators
            c1, c2, c3 = st.columns(3)
            c1.metric("Mechanical", "✅" if "mech_result" in st.session_state else "⏳")
            c2.metric("Hydraulic", "✅" if "hyd_result" in st.session_state else "⏳")
            c3.metric("Electrical", "✅" if "elec_result" in st.session_state else "⏳")
        else:
            # Aggregate results
            mech = st.session_state.mech_result
            hyd = st.session_state.hyd_result
            elec = st.session_state.elec_result
            
            # Simple integration logic
            severities = [mech["severity"], hyd["severity"], elec["severity"]]
            overall_sev = "High" if "High" in severities else "Medium" if "Medium" in severities else "Low"
            
            # Confidence boost for correlation
            base_conf = np.mean([mech["confidence"], hyd["confidence"], elec["confidence"]])
            corr_bonus = 10 if mech["fault_type"] != "normal" and hyd["fault_type"] != "normal" else 0
            overall_conf = min(95, int(base_conf + corr_bonus))
            
            # Display
            col1, col2, col3 = st.columns(3)
            col1.metric("Overall Diagnosis", "Coupled Fault" if corr_bonus > 0 else "Normal", f"{overall_conf}%")
            col2.metric("Severity", {"Low":"🟢","Medium":"🟠","High":"🔴"}.get(overall_sev,"⚪"))
            col3.metric("Domains", "3/3 Analyzed")
            
            # QCDSM Recommendations
            with st.expander("✅ QCDSM Action Plan", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Quality**")
                    st.write("• Verifikasi condition vs OEM spec")
                    st.write("**Cost**")
                    st.write("• Estimasi energy loss jika efficiency drop")
                with c2:
                    st.write("**Safety**")
                    if fluid_type in ["Pertalite (RON 90)", "Pertamax (RON 92)"]:
                        st.write("• ⚠️ Monitor NPSH margin untuk fluid mudah menguap")
                    else:
                        st.write("• Tidak ada immediate safety risk")
                    st.write("**Spirit**")
                    st.write("• Dokumentasikan lesson learned di CMMS")
            
            # Export
            if st.button("📥 Export CSV Report", type="primary"):
                csv = f"PUMP_DIAGNOSTIC,{machine_id},{datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                csv += f"Mechanical,{mech['diagnosis']},{mech['confidence']}%,{mech['severity']}\n"
                csv += f"Hydraulic,{hyd['diagnosis']},{hyd['confidence']}%,{hyd['severity']}\n"
                csv += f"Electrical,{elec['diagnosis']},{elec['confidence']}%,{elec['severity']}\n"
                csv += f"Integrated,Overall,{overall_conf}%,{overall_sev}\n"
                
                st.download_button("Download", csv, f"DIAG_{machine_id}.csv", "text/csv")
                st.success("✅ Report generated")
    
    # Footer
    st.divider()
    st.caption("Standar: ISO 10816 | API 610 | NEMA MG-1 | ⚠️ Decision Support Tool")

if __name__ == "__main__":
    main()
