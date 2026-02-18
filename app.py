import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter

# ============================================================================
# KONFIGURASI GLOBAL - MULTI-DOMAIN EXPERT SYSTEM
# ============================================================================

# --- Mechanical Vibration Limits (ISO 10816-3/7) ---
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

# --- Hydraulic Fluid Properties (BBM Specific - Pertamina) ---
FLUID_PROPERTIES = {
    "Pertalite (RON 90)": {
        "sg": 0.73,
        "vapor_pressure_kpa_38C": 52,
        "viscosity_cst_40C": 0.6,
        "flash_point_C": -40,
        "risk_level": "High"
    },
    "Pertamax (RON 92)": {
        "sg": 0.74,
        "vapor_pressure_kpa_38C": 42,
        "viscosity_cst_40C": 0.6,
        "flash_point_C": -40,
        "risk_level": "High"
    },
    "Diesel / Solar": {
        "sg": 0.84,
        "vapor_pressure_kpa_38C": 0.5,
        "viscosity_cst_40C": 3.0,
        "flash_point_C": 52,
        "risk_level": "Moderate"
    }
}

# --- Electrical Thresholds (NEMA MG-1 & Practical Limits) ---
ELECTRICAL_LIMITS = {
    "voltage_unbalance_warning": 2.0,  # %
    "voltage_unbalance_critical": 3.5,  # %
    "current_unbalance_warning": 5.0,   # %
    "current_unbalance_critical": 10.0, # %
    "voltage_tolerance_low": 90,        # % of rated
    "voltage_tolerance_high": 110,      # % of rated
    "current_load_warning": 90,         # % of rated
    "current_load_critical": 100        # % of rated
}

# ============================================================================
# FUNGSI REKOMENDASI - MULTI-DOMAIN
# ============================================================================

def get_mechanical_recommendation(diagnosis: str, location: str, severity: str = "Medium") -> str:
    rec_map = {
        "UNBALANCE": (
            f"🔧 **{location} - Unbalance**\n"
            f"• Lakukan single/dual plane balancing pada rotor\n"
            f"• Periksa: material buildup pada impeller, korosi blade, keyway wear\n"
            f"• Target residual unbalance: < 4W/N (g·mm) per ISO 1940-1\n"
            f"• Severity: {severity} → {'Segera jadwalkan balancing' if severity != 'Low' else 'Monitor trend'}"
        ),
        "MISALIGNMENT": (
            f"🔧 **{location} - Misalignment**\n"
            f"• Lakukan laser alignment pump-motor coupling\n"
            f"• Toleransi target: < 0.05 mm offset, < 0.05 mm/m angular\n"
            f"• Periksa: pipe strain, soft foot, coupling wear\n"
            f"• Severity: {severity} → {'Stop & align segera' if severity == 'High' else 'Jadwalkan alignment'}"
        ),
        "LOOSENESS": (
            f"🔧 **{location} - Mechanical Looseness**\n"
            f"• Torque check semua baut: foundation, bearing housing, baseplate\n"
            f"• Periksa: crack pada struktur, worn dowel pins, grout deterioration\n"
            f"• Gunakan torque wrench sesuai spec manufacturer\n"
            f"• Severity: {severity} → {'Amankan sebelum operasi' if severity == 'High' else 'Jadwalkan tightening'}"
        ),
        "BEARING_EARLY": (
            f"🔧 **{location} - Early Bearing Fault / Lubrication**\n"
            f"• Cek lubrication: jenis grease, interval, quantity\n"
            f"• Ambil oil sample jika applicable (particle count, viscosity)\n"
            f"• Monitor trend Band 3 mingguan\n"
            f"• Severity: {severity} → {'Ganti grease & monitor ketat' if severity != 'Low' else 'Lanjutkan monitoring'}"
        ),
        "BEARING_DEVELOPED": (
            f"🔧 **{location} - Developed Bearing Fault**\n"
            f"• Jadwalkan bearing replacement dalam 1-3 bulan\n"
            f"• Siapkan spare bearing (pastikan clearance & fit sesuai spec)\n"
            f"• Monitor weekly: jika Band 1 naik drastis → percepat jadwal\n"
            f"• Severity: {severity} → {'Plan shutdown segera' if severity == 'High' else 'Siapkan work order'}"
        ),
        "BEARING_SEVERE": (
            f"🔴 **{location} - Severe Bearing Damage**\n"
            f"• RISK OF CATASTROPHIC FAILURE - Pertimbangkan immediate shutdown\n"
            f"• Jika continue operasi: monitor hourly, siapkan emergency replacement\n"
            f"• Investigasi root cause: lubrication, installation, loading?\n"
            f"• Severity: HIGH → Action required dalam 24 jam"
        ),
        "Tidak Terdiagnosa": (
            "⚠️ **Pola Tidak Konsisten**\n"
            "• Data tidak match dengan rule mekanikal standar\n"
            "• Kemungkinan: multi-fault interaction, measurement error, atau fault non-rutin\n"
            "• Rekomendasi: Analisis manual oleh Vibration Analyst Level II+ dengan full spectrum review"
        )
    }
    return rec_map.get(diagnosis, rec_map["Tidak Terdiagnosa"])


def get_hydraulic_recommendation(diagnosis: str, fluid_type: str, severity: str = "Medium") -> str:
    rec_map = {
        "CAVITATION": (
            f"💧 **{fluid_type} - Cavitation Risk**\n"
            f"• Tingkatkan suction pressure atau turunkan fluid temperature\n"
            f"• Cek: strainer clogged, valve posisi, NPSH margin\n"
            f"• Target NPSH margin: > 0.5 m untuk {fluid_type}\n"
            f"• Severity: {severity} → {'Evaluasi immediate shutdown jika NPSH margin <0.3m' if severity == 'High' else 'Monitor intensif'}"
        ),
        "IMPELLER_WEAR": (
            f"💧 **{fluid_type} - Impeller Wear / Internal Clearance**\n"
            f"• Jadwalkan inspection impeller & wear ring\n"
            f"• Ukur internal clearance vs spec OEM\n"
            f"• Pertimbangkan: fluid viscosity effect pada slip loss\n"
            f"• Severity: {severity} → {'Siapkan spare impeller' if severity != 'Low' else 'Monitor trend efisiensi'}"
        ),
        "SYSTEM_RESISTANCE_HIGH": (
            f"💧 **{fluid_type} - System Resistance Higher Than Design**\n"
            f"• Cek valve discharge position, clogged line, atau filter pressure drop\n"
            f"• Verifikasi P&ID vs as-built condition\n"
            f"• Evaluasi: apakah operating point masih dalam acceptable range?\n"
            f"• Severity: {severity} → {'Adjust valve / clean line segera' if severity == 'High' else 'Jadwalkan system review'}"
        ),
        "EFFICIENCY_DROP": (
            f"💧 **{fluid_type} - Efficiency Degradation**\n"
            f"• Hitung energy loss: {(100-75):.1f}% efficiency → Rp X/jam additional cost\n"
            f"• Investigasi: mechanical loss vs hydraulic loss vs fluid property mismatch\n"
            f"• Pertimbangkan: balancing repair cost vs continued operation cost\n"
            f"• Severity: {severity} → {'Plan overhaul dalam 1-3 bulan' if severity != 'Low' else 'Monitor monthly'}"
        ),
        "NORMAL_OPERATION": (
            f"✅ **{fluid_type} - Normal Operation**\n"
            f"• Semua parameter dalam batas acceptable (±5% dari design)\n"
            f"• Rekam data ini sebagai baseline untuk trend monitoring\n"
            f"• Interval pengukuran berikutnya: sesuai maintenance plan\n"
            f"• Severity: Low → Continue routine monitoring"
        ),
        "Tidak Terdiagnosa": (
            "⚠️ **Pola Tidak Konsisten**\n"
            "• Data hydraulic tidak match dengan rule standar\n"
            "• Kemungkinan: multi-fault interaction, measurement error, atau fluid property mismatch\n"
            "• Rekomendasi: Verifikasi data lapangan + cross-check dengan electrical/mechanical data"
        )
    }
    return rec_map.get(diagnosis, rec_map["Tidak Terdiagnosa"])


def get_electrical_recommendation(diagnosis: str, severity: str = "Medium") -> str:
    rec_map = {
        "UNDER_VOLTAGE": (
            f"⚡ **Under Voltage Condition**\n"
            f"• Cek supply voltage di MCC: possible transformer tap / cable voltage drop\n"
            f"• Verify: motor rated voltage vs actual operating voltage\n"
            f"• Impact: torque ↓ ~ (V/Vrated)² → hydraulic performance ↓\n"
            f"• Severity: {severity} → {'Coordinate dengan electrical team segera' if severity == 'High' else 'Monitor voltage trend'}"
        ),
        "VOLTAGE_UNBALANCE": (
            f"⚡ **Voltage Unbalance Detected**\n"
            f"• Cek 3-phase supply balance di source: possible single-phase loading\n"
            f"• Inspect: loose connection, corroded terminal, faulty breaker\n"
            f"• Impact: negative sequence current → rotor heating ↑ → bearing stress ↑\n"
            f"• Severity: {severity} → {'Balance supply sebelum mechanical damage' if severity != 'Low' else 'Monitor monthly'}"
        ),
        "CURRENT_UNBALANCE": (
            f"⚡ **Current Unbalance Detected**\n"
            f"• Investigasi: winding fault, rotor bar issue, atau supply problem\n"
            f"• Cek insulation resistance & winding resistance balance\n"
            f"• Monitor: motor temperature & vibration @ 2×line frequency\n"
            f"• Severity: {severity} → {'Schedule electrical inspection' if severity != 'Low' else 'Continue monitoring'}"
        ),
        "HIGH_CURRENT_NORMAL_OUTPUT": (
            f"⚡ **High Current + Normal Hydraulic Output**\n"
            f"• Indikasi: mechanical loss ↑ (bearing, misalignment) atau electrical inefficiency\n"
            f"• Cross-check: vibration level, bearing temperature, coupling condition\n"
            f"• Evaluate: motor efficiency vs nameplate rating\n"
            f"• Severity: {severity} → {'Investigate mechanical domain first' if severity != 'Low' else 'Monitor trend'}"
        ),
        "NORMAL_ELECTRICAL": (
            f"✅ **Normal Electrical Condition**\n"
            f"• Voltage balance <2%, current balance <5%, within rated limits\n"
            f"• Power factor dalam range acceptable untuk motor induksi\n"
            f"• Tidak ada indikasi electrical fault yang mempengaruhi mechanical/hydraulic\n"
            f"• Severity: Low → Continue routine electrical monitoring"
        ),
        "Tidak Terdiagnosa": (
            "⚠️ **Pola Tidak Konsisten**\n"
            "• Data electrical tidak match dengan rule standar\n"
            "• Kemungkinan: measurement error, transient condition, atau multi-domain interaction\n"
            "• Rekomendasi: Verifikasi dengan power quality analyzer + cross-check domain lain"
        )
    }
    return rec_map.get(diagnosis, rec_map["Tidak Terdiagnosa"])


# ============================================================================
# FUNGSI PERHITUNGAN - HYDRAULIC DOMAIN
# ============================================================================

def calculate_hydraulic_parameters(suction_pressure, discharge_pressure, flow_rate, 
                                  motor_power, sg, fluid_temp_c=40):
    """
    Calculate derived hydraulic parameters for single-point analysis
    Returns dict with head, hydraulic_power, efficiency, and deviations
    """
    # Differential pressure (bar)
    delta_p = discharge_pressure - suction_pressure
    
    # Head calculation (m): H = ΔP × 10.2 / SG
    head = delta_p * 10.2 / sg if sg > 0 else 0
    
    # Hydraulic power (kW): P_hyd = (Q × H × SG × 9.81) / 3600
    # Q in m³/h → convert to m³/s: /3600
    hydraulic_power = (flow_rate * head * sg * 9.81) / 3600 if flow_rate > 0 and head > 0 else 0
    
    # Efficiency estimation (%)
    efficiency = (hydraulic_power / motor_power * 100) if motor_power > 0 else 0
    
    return {
        "delta_p_bar": delta_p,
        "head_m": head,
        "hydraulic_power_kw": hydraulic_power,
        "efficiency_percent": efficiency
    }


def classify_hydraulic_performance(head_aktual, head_design, efficiency_aktual, 
                                   efficiency_bep, flow_aktual, flow_design):
    """
    Classify hydraulic performance deviation pattern
    Returns: pattern_name, deviation_details dict
    """
    dev_head = ((head_aktual - head_design) / head_design * 100) if head_design > 0 else 0
    dev_eff = ((efficiency_aktual - efficiency_bep) / efficiency_bep * 100) if efficiency_bep > 0 else 0
    dev_flow = ((flow_aktual - flow_design) / flow_design * 100) if flow_design > 0 else 0
    
    if dev_head < -5 and dev_eff < -5:
        return "UNDER_PERFORMANCE", {"head_dev": dev_head, "eff_dev": dev_eff}
    elif dev_head > 5 and dev_flow < -5:
        return "OVER_RESISTANCE", {"head_dev": dev_head, "flow_dev": dev_flow}
    elif dev_eff < -10 and abs(dev_head) <= 5:
        return "EFFICIENCY_DROP", {"eff_dev": dev_eff}
    elif abs(dev_head) <= 5 and abs(dev_eff) <= 5 and abs(dev_flow) <= 5:
        return "NORMAL", {"head_dev": dev_head, "eff_dev": dev_eff, "flow_dev": dev_flow}
    else:
        return "MIXED_DEVIATION", {"head_dev": dev_head, "eff_dev": dev_eff, "flow_dev": dev_flow}


# ============================================================================
# FUNGSI PERHITUNGAN - ELECTRICAL DOMAIN
# ============================================================================

def calculate_electrical_parameters(v_l1l2, v_l2l3, v_l3l1, i_l1, i_l2, i_l3, 
                                    power_factor, rated_voltage, rated_current):
    """
    Calculate derived electrical parameters for 3-phase system
    Returns dict with unbalance %, load estimate, and power calculation
    """
    # Average voltage and current
    v_avg = (v_l1l2 + v_l2l3 + v_l3l1) / 3
    i_avg = (i_l1 + i_l2 + i_l3) / 3
    
    # Voltage unbalance (%) - NEMA method: max deviation / average × 100
    v_deviations = [abs(v - v_avg) for v in [v_l1l2, v_l2l3, v_l3l1]]
    voltage_unbalance = (max(v_deviations) / v_avg * 100) if v_avg > 0 else 0
    
    # Current unbalance (%)
    i_deviations = [abs(i - i_avg) for i in [i_l1, i_l2, i_l3]]
    current_unbalance = (max(i_deviations) / i_avg * 100) if i_avg > 0 else 0
    
    # Estimated motor load (%) - simplified method
    load_estimate = (i_avg / rated_current * 100) if rated_current > 0 else 0
    
    # Electrical power input (kW) - if not directly measured
    electrical_power = (np.sqrt(3) * v_avg * i_avg * power_factor / 1000) if v_avg > 0 and i_avg > 0 else 0
    
    # Voltage tolerance check
    voltage_within_tolerance = (ELECTRICAL_LIMITS["voltage_tolerance_low"] <= 
                               (v_avg / rated_voltage * 100) <= 
                               ELECTRICAL_LIMITS["voltage_tolerance_high"])
    
    return {
        "v_avg": v_avg,
        "i_avg": i_avg,
        "voltage_unbalance_percent": voltage_unbalance,
        "current_unbalance_percent": current_unbalance,
        "load_estimate_percent": load_estimate,
        "electrical_power_kw": electrical_power,
        "voltage_within_tolerance": voltage_within_tolerance
    }


def classify_electrical_condition(voltage_unbalance, current_unbalance, 
                                  load_estimate, voltage_within_tolerance,
                                  rated_voltage, v_avg):
    """
    Classify electrical condition based on calculated parameters
    Returns: diagnosis, confidence, severity
    """
    # Check under voltage
    if not voltage_within_tolerance and v_avg < rated_voltage:
        severity = "High" if load_estimate > 80 else "Medium"
        return "UNDER_VOLTAGE", 70, severity
    
    # Check voltage unbalance
    if voltage_unbalance > ELECTRICAL_LIMITS["voltage_unbalance_critical"]:
        return "VOLTAGE_UNBALANCE", 75, "High"
    elif voltage_unbalance > ELECTRICAL_LIMITS["voltage_unbalance_warning"]:
        return "VOLTAGE_UNBALANCE", 65, "Medium"
    
    # Check current unbalance
    if current_unbalance > ELECTRICAL_LIMITS["current_unbalance_critical"]:
        return "CURRENT_UNBALANCE", 70, "High"
    elif current_unbalance > ELECTRICAL_LIMITS["current_unbalance_warning"]:
        return "CURRENT_UNBALANCE", 60, "Medium"
    
    # Check high current with normal output (needs hydraulic context - handled in cross-domain)
    if load_estimate > ELECTRICAL_LIMITS["current_load_critical"]:
        return "HIGH_CURRENT_NORMAL_OUTPUT", 55, "Medium"
    
    # Normal condition
    return "NORMAL_ELECTRICAL", 95, "Low"


# ============================================================================
# FUNGSI DIAGNOSA - MECHANICAL DOMAIN (Existing - Enhanced)
# ============================================================================

def diagnose_single_point_mechanical(peaks, bands, rpm_hz, point, overall_vel, has_fft: bool = True):
    result = {
        "diagnosis": "Normal",
        "confidence": 0,
        "severity": "Low",
        "fault_type": None,
        "domain": "mechanical"
    }
    
    # --- LOW FREQUENCY FAULTS (hanya jika ada FFT data) ---
    if has_fft:
        # UNBALANCE
        if "Axial" not in point:
            peak_1x = None
            for freq, amp in peaks:
                if abs(freq - rpm_hz) < 0.05 * rpm_hz:
                    peak_1x = amp
                    break
            if peak_1x and peak_1x > 0.7 * sum(p[1] for p in peaks):
                result["diagnosis"] = "UNBALANCE"
                result["confidence"] = min(95, 70 + int((peak_1x / 4.5) * 10))
                result["severity"] = "High" if overall_vel > ISO_LIMITS_VELOCITY["Zone C (Unacceptable)"] else "Medium" if overall_vel > ISO_LIMITS_VELOCITY["Zone B (Acceptable)"] else "Low"
                result["fault_type"] = "low_freq"
                return result
        
        # MISALIGNMENT
        if "Axial" in point:
            has_1x = any(abs(p[0] - rpm_hz) < 0.05 * rpm_hz for p in peaks)
            has_2x = any(abs(p[0] - 2*rpm_hz) < 0.05 * rpm_hz for p in peaks)
            if has_1x and has_2x:
                amp_1x = next((p[1] for p in peaks if abs(p[0]-rpm_hz)<0.05*rpm_hz), 0)
                amp_2x = next((p[1] for p in peaks if abs(p[0]-2*rpm_hz)<0.05*rpm_hz), 0)
                if amp_2x > 0.5 * amp_1x:
                    result["diagnosis"] = "MISALIGNMENT"
                    result["confidence"] = min(95, 65 + int((amp_2x/amp_1x) * 20) if amp_1x > 0 else 65)
                    result["severity"] = "High" if overall_vel > ISO_LIMITS_VELOCITY["Zone C (Unacceptable)"] else "Medium"
                    result["fault_type"] = "low_freq"
                    return result
        
        # LOOSENESS
        if "Vertical" in point:
            harmonics_ok = all(any(abs(p[0] - (i+1)*rpm_hz) < 0.05*rpm_hz for p in peaks) for i in range(3))
            if harmonics_ok:
                amps = [next((p[1] for p in peaks if abs(p[0]-(i+1)*rpm_hz)<0.05*rpm_hz), 0) for i in range(3)]
                if amps[0] > 0 and amps[1] > 0.5*amps[0] and amps[2] > 0.3*amps[0]:
                    result["diagnosis"] = "LOOSENESS"
                    result["confidence"] = min(90, 60 + int((amps[1]/amps[0] + amps[2]/amps[0]) * 15))
                    result["severity"] = "High" if overall_vel > ISO_LIMITS_VELOCITY["Zone C (Unacceptable)"] else "Medium"
                    result["fault_type"] = "low_freq"
                    return result
    
    # --- HIGH FREQUENCY FAULTS (bisa dari Bands saja) ---
    b1, b2, b3 = bands["Band1"], bands["Band2"], bands["Band3"]
    base1, base2, base3 = ACCEL_BASELINE["Band1 (0.5-1.5kHz)"], ACCEL_BASELINE["Band2 (1.5-5kHz)"], ACCEL_BASELINE["Band3 (5-16kHz)"]
    
    # Pattern: Incipient / Lubrication
    if b3 > 2.0 * base3 and b2 < 1.5 * base2 and b1 < 1.5 * base1:
        result["diagnosis"] = "BEARING_EARLY"
        conf_boost = 10 if has_fft else 0
        result["confidence"] = min(85, 60 + int((b3/base3 - 2) * 10) + conf_boost)
        result["severity"] = "Medium" if b3 > 3*base3 else "Low"
        result["fault_type"] = "high_freq"
        return result
    
    # Pattern: Developing Fault
    if b2 > 2.0 * base2 and b3 > 1.5 * base3 and b1 < 1.5 * base1:
        result["diagnosis"] = "BEARING_DEVELOPED"
        conf_boost = 10 if has_fft else 0
        result["confidence"] = min(90, 70 + int((b2/base2 - 2) * 8) + conf_boost)
        result["severity"] = "High" if b2 > 3*base2 else "Medium"
        result["fault_type"] = "high_freq"
        return result
    
    # Pattern: Severe Fault
    if b1 > 2.5 * base1 and b2 > 1.5 * base2:
        result["diagnosis"] = "BEARING_SEVERE"
        result["confidence"] = min(95, 80 + int((b1/base1 - 2.5) * 6))
        result["severity"] = "High"
        result["fault_type"] = "high_freq"
        return result
    
    # Default
    if overall_vel <= ISO_LIMITS_VELOCITY["Zone B (Acceptable)"]:
        result["diagnosis"] = "Normal"
        result["confidence"] = 99
        result["severity"] = "Low"
    else:
        result["diagnosis"] = "Tidak Terdiagnosa"
        result["confidence"] = 30 if not has_fft else 40
        result["severity"] = "Medium"
    
    return result


# ============================================================================
# FUNGSI DIAGNOSA - HYDRAULIC DOMAIN (Single-Point Steady-State)
# ============================================================================

def diagnose_hydraulic_single_point(hydraulic_calc, design_params, fluid_props, 
                                    observations, context):
    """
    Diagnose hydraulic condition based on single-point steady-state measurement
    Returns: diagnosis, confidence, severity, details
    """
    result = {
        "diagnosis": "NORMAL_OPERATION",
        "confidence": 0,
        "severity": "Low",
        "fault_type": None,
        "domain": "hydraulic",
        "details": {}
    }
    
    # Extract calculated and design parameters
    head_aktual = hydraulic_calc.get("head_m", 0)
    eff_aktual = hydraulic_calc.get("efficiency_percent", 0)
    
    head_design = design_params.get("rated_head_m", 0)
    eff_bep = design_params.get("bep_efficiency", 0)
    flow_design = design_params.get("rated_flow_m3h", 0)
    flow_aktual = context.get("flow_aktual", 0)
    
    # Classify performance pattern
    pattern, deviations = classify_hydraulic_performance(
        head_aktual, head_design, eff_aktual, eff_bep, flow_aktual, flow_design
    )
    result["details"]["deviations"] = deviations
    
    # Check cavitation risk (NPSH margin simplified)
    # NPSHa ≈ (P_suction_abs - P_vapor) / (ρ×g) - friction_loss (simplified)
    suction_pressure_bar = context.get("suction_pressure_bar", 0)
    vapor_pressure_kpa = fluid_props.get("vapor_pressure_kpa_38C", 0)
    sg = fluid_props.get("sg", 0.84)
    
    # Simplified NPSHa estimation (m)
    # P_suction_abs = (suction_pressure_gauge + 1.013) bar → convert to kPa
    p_suction_abs_kpa = (suction_pressure_bar + 1.013) * 100
    p_vapor_kpa = vapor_pressure_kpa
    npsha_estimated = (p_suction_abs_kpa - p_vapor_kpa) / (sg * 9.81) if sg > 0 else 0
    npshr = design_params.get("npsh_required_m", 0)
    npsh_margin = npsha_estimated - npshr
    
    result["details"]["npsh_margin_m"] = npsh_margin
    
    # Rule-based diagnosis matrix
    noise_type = observations.get("noise_type", "Normal")
    fluid_condition = observations.get("fluid_condition", "Jernih")
    leakage = observations.get("leakage", "Tidak ada")
    
    # Cavitation detection
    if noise_type == "Crackling" and npsh_margin < 0.5:
        result["diagnosis"] = "CAVITATION"
        result["confidence"] = min(90, 70 + int((0.5 - npsh_margin) * 20) if npsh_margin < 0.5 else 70)
        result["severity"] = "High" if npsh_margin < 0.3 else "Medium"
        result["fault_type"] = "cavitation"
        return result
    
    # Impeller wear / internal clearance
    if pattern == "UNDER_PERFORMANCE" and noise_type == "Normal" and fluid_condition in ["Jernih", "Agak keruh"]:
        result["diagnosis"] = "IMPELLER_WEAR"
        result["confidence"] = min(85, 60 + int(abs(deviations.get("head_dev", 0)) * 2))
        result["severity"] = "High" if deviations.get("head_dev", 0) < -15 else "Medium"
        result["fault_type"] = "wear"
        return result
    
    # System resistance higher than design
    if pattern == "OVER_RESISTANCE":
        result["diagnosis"] = "SYSTEM_RESISTANCE_HIGH"
        result["confidence"] = 75
        result["severity"] = "Medium"
        result["fault_type"] = "system"
        return result
    
    # Efficiency drop investigation
    if pattern == "EFFICIENCY_DROP":
        result["diagnosis"] = "EFFICIENCY_DROP"
        result["confidence"] = min(80, 65 + int(abs(deviations.get("eff_dev", 0))))
        result["severity"] = "High" if deviations.get("eff_dev", 0) < -20 else "Medium"
        result["fault_type"] = "efficiency"
        return result
    
    # Normal operation
    if pattern == "NORMAL":
        result["diagnosis"] = "NORMAL_OPERATION"
        result["confidence"] = 95
        result["severity"] = "Low"
        result["fault_type"] = "normal"
        return result
    
    # Default fallback
    result["diagnosis"] = "Tidak Terdiagnosa"
    result["confidence"] = 40
    result["severity"] = "Medium"
    result["fault_type"] = "unknown"
    
    return result


# ============================================================================
# FUNGSI DIAGNOSA - ELECTRICAL DOMAIN
# ============================================================================

def diagnose_electrical_condition(electrical_calc, motor_specs, observations):
    """
    Diagnose electrical condition based on 3-phase measurements
    Returns: diagnosis, confidence, severity, details
    """
    result = {
        "diagnosis": "NORMAL_ELECTRICAL",
        "confidence": 0,
        "severity": "Low",
        "fault_type": None,
        "domain": "electrical",
        "details": {}
    }
    
    # Extract calculated parameters
    voltage_unbalance = electrical_calc.get("voltage_unbalance_percent", 0)
    current_unbalance = electrical_calc.get("current_unbalance_percent", 0)
    load_estimate = electrical_calc.get("load_estimate_percent", 0)
    voltage_within_tolerance = electrical_calc.get("voltage_within_tolerance", True)
    v_avg = electrical_calc.get("v_avg", 0)
    
    # Motor specs
    rated_voltage = motor_specs.get("rated_voltage", 400)
    
    # Classify condition
    diagnosis, confidence, severity = classify_electrical_condition(
        voltage_unbalance, current_unbalance, load_estimate, 
        voltage_within_tolerance, rated_voltage, v_avg
    )
    
    result["diagnosis"] = diagnosis
    result["confidence"] = confidence
    result["severity"] = severity
    result["fault_type"] = "voltage" if "VOLTAGE" in diagnosis else "current" if "CURRENT" in diagnosis else "normal"
    result["details"] = {
        "voltage_unbalance": voltage_unbalance,
        "current_unbalance": current_unbalance,
        "load_estimate": load_estimate
    }
    
    # Adjust confidence based on observations
    motor_temp = observations.get("motor_temperature", "Normal")
    if motor_temp in ["Panas (>90°C)", "Hangat (70-90°C)"] and severity == "Low":
        result["severity"] = "Medium"
        result["confidence"] = min(95, result["confidence"] + 10)
    
    return result


# ============================================================================
# CROSS-DOMAIN INTEGRATION LOGIC
# ============================================================================

def aggregate_cross_domain_diagnosis(mech_result, hyd_result, elec_result, shared_context):
    """
    Integrate diagnoses from three domains with correlation logic
    Returns: integrated diagnosis with confidence boost/correction
    """
    system_result = {
        "diagnosis": "Normal - All Domains",
        "confidence": 0,
        "severity": "Low",
        "location": "N/A",
        "domain_breakdown": {},
        "correlation_notes": [],
        "qcsm_recommendations": {}
    }
    
    # Store individual results
    system_result["domain_breakdown"] = {
        "mechanical": mech_result,
        "hydraulic": hyd_result,
        "electrical": elec_result
    }
    
    # Extract key indicators
    mech_fault = mech_result.get("fault_type")
    hyd_fault = hyd_result.get("fault_type")
    elec_fault = elec_result.get("fault_type")
    
    mech_sev = mech_result.get("severity", "Low")
    hyd_sev = hyd_result.get("severity", "Low")
    elec_sev = elec_result.get("severity", "Low")
    
    # Correlation logic: boost confidence if domains agree on root cause
    correlation_bonus = 0
    correlated_faults = []
    
    # Pattern 1: Electrical unbalance → Mechanical vibration @ 2×line freq → Hydraulic pulsation
    if (elec_fault == "voltage" and mech_result.get("diagnosis") in ["MISALIGNMENT", "LOOSENESS"] 
        and hyd_result.get("details", {}).get("deviations", {}).get("head_dev", 0) < -5):
        correlation_bonus += 15
        correlated_faults.append("Voltage unbalance → torque pulsation → hydraulic instability")
        system_result["diagnosis"] = "Electrical-Mechanical-Hydraulic Coupled Fault"
    
    # Pattern 2: Cavitation → Impeller erosion → Unbalance → Electrical stress
    if (hyd_fault == "cavitation" and mech_fault == "wear" 
        and elec_result.get("details", {}).get("current_unbalance", 0) > 5):
        correlation_bonus += 20
        correlated_faults.append("Cavitation → impeller erosion → unbalance → current fluctuation")
        system_result["diagnosis"] = "Cascading Failure: Cavitation Origin"
    
    # Pattern 3: High electrical load + low hydraulic output → internal loss
    if (elec_result.get("diagnosis") == "HIGH_CURRENT_NORMAL_OUTPUT" 
        and hyd_fault == "efficiency"):
        correlation_bonus += 10
        correlated_faults.append("High electrical input + low hydraulic output → internal mechanical/hydraulic loss")
        system_result["diagnosis"] = "Internal Loss Investigation Required"
    
    # Determine overall severity (safety-first approach)
    severities = [mech_sev, hyd_sev, elec_sev]
    if "High" in severities:
        system_result["severity"] = "High"
    elif "Medium" in severities:
        system_result["severity"] = "Medium"
    else:
        system_result["severity"] = "Low"
    
    # Calculate integrated confidence
    confidences = [r.get("confidence", 0) for r in [mech_result, hyd_result, elec_result] if r.get("confidence", 0) > 0]
    base_confidence = np.mean(confidences) if confidences else 0
    system_result["confidence"] = min(95, int(base_confidence + correlation_bonus))
    
    # Store correlation notes
    system_result["correlation_notes"] = correlated_faults if correlated_faults else ["No strong cross-domain correlation detected"]
    
    # Generate QCDSM-based recommendations
    system_result["qcsm_recommendations"] = generate_qcdsm_recommendations(
        mech_result, hyd_result, elec_result, system_result["severity"], shared_context
    )
    
    return system_result


def generate_qcdsm_recommendations(mech_result, hyd_result, elec_result, overall_severity, context):
    """
    Generate actionable recommendations structured by QCDSM criteria
    """
    recommendations = {
        "Quality": [],
        "Cost": [],
        "Delivery": [],
        "Safety": [],
        "Spirit": []
    }
    
    fluid_type = context.get("fluid_type", "Diesel / Solar")
    machine_id = context.get("machine_id", "Unknown")
    
    # Quality recommendations
    if mech_result.get("fault_type") in ["low_freq", "high_freq"]:
        recommendations["Quality"].append(f"Verifikasi mechanical condition {machine_id} vs OEM specification")
    if hyd_result.get("fault_type") in ["wear", "cavitation"]:
        recommendations["Quality"].append(f"Jadwalkan inspection internal components untuk {fluid_type} service")
    if elec_result.get("fault_type") in ["voltage", "current"]:
        recommendations["Quality"].append("Verifikasi electrical supply quality di MCC sebelum operasi lanjutan")
    if not recommendations["Quality"]:
        recommendations["Quality"].append("Continue routine monitoring per maintenance plan")
    
    # Cost recommendations
    if overall_severity == "High":
        recommendations["Cost"].append("Estimasi cost of failure vs cost of repair untuk prioritas tindakan")
    if hyd_result.get("details", {}).get("deviations", {}).get("eff_dev", 0) < -10:
        eff_loss = abs(hyd_result["details"]["deviations"]["eff_dev"])
        recommendations["Cost"].append(f"Efficiency drop {eff_loss:.1f}% → estimasi energy loss: Rp X/jam")
    if not recommendations["Cost"]:
        recommendations["Cost"].append("Preventive action lebih ekonomis vs corrective maintenance")
    
    # Delivery recommendations
    if overall_severity == "High":
        recommendations["Delivery"].append("Evaluasi impact pada production schedule sebelum shutdown")
    elif overall_severity == "Medium":
        recommendations["Delivery"].append("Operasi dapat dilanjutkan dengan monitoring intensif")
    else:
        recommendations["Delivery"].append("Tidak ada impact pada delivery schedule")
    
    # Safety recommendations (BBM context)
    if fluid_type in ["Pertalite (RON 90)", "Pertamax (RON 92)"] and hyd_result.get("fault_type") == "cavitation":
        recommendations["Safety"].append("⚠️ Cavitation risk dengan fluid mudah menguap → monitor NPSH margin ketat")
    if elec_result.get("severity") == "High":
        recommendations["Safety"].append("Pastikan electrical work dilakukan oleh personnel bersertifikat")
    if mech_result.get("severity") == "High":
        recommendations["Safety"].append("Verifikasi mechanical integrity sebelum continue operation")
    if not recommendations["Safety"]:
        recommendations["Safety"].append("Tidak ada immediate safety risk teridentifikasi")
    
    # Spirit / Learning recommendations
    recommendations["Spirit"].append("Dokumentasikan lesson learned untuk update inspection checklist")
    if any(r.get("fault_type") == "unknown" for r in [mech_result, hyd_result, elec_result]):
        recommendations["Spirit"].append("Share temuan ke tim reliability untuk root cause analysis mendalam")
    recommendations["Spirit"].append("Update asset history di CMMS untuk predictive maintenance improvement")
    
    return recommendations


# ============================================================================
# REPORT GENERATION - UNIFIED FORMAT
# ============================================================================

def generate_unified_csv_report(machine_id, rpm, timestamp, mech_data, hyd_data, elec_data, integrated_result):
    """
    Generate unified CSV report with all three domains + integrated summary
    """
    lines = []
    lines.append(f"MULTI-DOMAIN PUMP DIAGNOSTIC REPORT - {machine_id.upper()}")
    lines.append(f"Generated: {timestamp}")
    lines.append(f"RPM: {rpm} | 1x RPM: {rpm/60:.2f} Hz")
    lines.append(f"Standards: ISO 10816-3/7 (Mech) | API 610 (Hyd) | NEMA MG-1 (Elec)")
    lines.append("")
    
    # Mechanical section
    lines.append("=== MECHANICAL VIBRATION ===")
    if mech_data.get("points"):
        lines.append("POINT,Overall_Vel(mm/s),Band1(g),Band2(g),Band3(g),Diagnosis,Confidence,Severity")
        for point, data in mech_data["points"].items():
            lines.append(f"{point},{data['velocity']:.2f},{data['bands']['Band1']:.3f},{data['bands']['Band2']:.3f},{data['bands']['Band3']:.3f},{data['diagnosis']},{data['confidence']},{data['severity']}")
    lines.append(f"System Diagnosis: {mech_data.get('system_diagnosis', 'N/A')}")
    lines.append("")
    
    # Hydraulic section
    lines.append("=== HYDRAULIC PERFORMANCE (Single-Point) ===")
    if hyd_data.get("measurements"):
        m = hyd_data["measurements"]
        lines.append(f"Fluid: {hyd_data.get('fluid_type', 'N/A')} | SG: {hyd_data.get('sg', 'N/A')}")
        lines.append(f"Suction: {m.get('suction_pressure', 0):.2f} bar | Discharge: {m.get('discharge_pressure', 0):.2f} bar")
        lines.append(f"Flow: {m.get('flow_rate', 0):.1f} m³/h | Power: {m.get('motor_power', 0):.1f} kW")
        lines.append(f"Calculated Head: {hyd_data.get('head_m', 0):.1f} m | Efficiency: {hyd_data.get('efficiency_percent', 0):.1f}%")
        lines.append(f"NPSH Margin: {hyd_data.get('npsh_margin_m', 0):.2f} m")
    lines.append(f"Diagnosis: {hyd_data.get('diagnosis', 'N/A')} | Confidence: {hyd_data.get('confidence', 0)}% | Severity: {hyd_data.get('severity', 'N/A')}")
    lines.append("")
    
    # Electrical section
    lines.append("=== ELECTRICAL CONDITION (3-Phase) ===")
    if elec_data.get("measurements"):
        e = elec_data["measurements"]
        lines.append(f"Voltage L1-L2: {e.get('v_l1l2', 0):.1f}V | L2-L3: {e.get('v_l2l3', 0):.1f}V | L3-L1: {e.get('v_l3l1', 0):.1f}V")
        lines.append(f"Current L1: {e.get('i_l1', 0):.1f}A | L2: {e.get('i_l2', 0):.1f}A | L3: {e.get('i_l3', 0):.1f}A")
        lines.append(f"Voltage Unbalance: {elec_data.get('voltage_unbalance', 0):.2f}% | Current Unbalance: {elec_data.get('current_unbalance', 0):.2f}%")
        lines.append(f"Load Estimate: {elec_data.get('load_estimate', 0):.1f}% | PF: {e.get('power_factor', 0):.2f}")
    lines.append(f"Diagnosis: {elec_data.get('diagnosis', 'N/A')} | Confidence: {elec_data.get('confidence', 0)}% | Severity: {elec_data.get('severity', 'N/A')}")
    lines.append("")
    
    # Integrated summary
    lines.append("=== INTEGRATED DIAGNOSIS ===")
    lines.append(f"Overall Diagnosis: {integrated_result.get('diagnosis', 'N/A')}")
    lines.append(f"Overall Confidence: {integrated_result.get('confidence', 0)}%")
    lines.append(f"Overall Severity: {integrated_result.get('severity', 'N/A')}")
    lines.append(f"Correlation Notes: {'; '.join(integrated_result.get('correlation_notes', []))}")
    lines.append("")
    
    # QCDSM Recommendations
    lines.append("=== QCDSM RECOMMENDATIONS ===")
    for category, recs in integrated_result.get("qcsm_recommendations", {}).items():
        lines.append(f"{category.upper()}:")
        for rec in recs:
            lines.append(f"  • {rec}")
        lines.append("")
    
    return "\n".join(lines)


# ============================================================================
# STREAMLIT UI - MAIN APPLICATION
# ============================================================================

def main():
    # Page configuration
    st.set_page_config(
        page_title="Pump Diagnostic Expert System",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state for cross-tab data sharing
    if "shared_context" not in st.session_state:
        st.session_state.shared_context = {
            "machine_id": "P-101",
            "rpm": 2950,
            "service_criticality": "Essential (Utility)",
            "fluid_type": "Diesel / Solar",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    # Header with branding
    st.markdown("""
    <div style="background-color:#1E3A5F; padding:15px; border-radius:8px; margin-bottom:20px">
        <h2 style="color:white; margin:0">🔧💧⚡ Pump Diagnostic Expert System</h2>
        <p style="color:#E0E0E0; margin:5px 0 0 0">
            Integrated Mechanical • Hydraulic • Electrical Analysis | Pertamina Patra Niaga
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar: Shared Context Panel (sticky)
    with st.sidebar:
        st.subheader("📍 Shared Context")
        
        # Machine identification
        machine_id = st.text_input("Machine ID / Tag", value=st.session_state.shared_context["machine_id"])
        rpm = st.number_input("Operating RPM", min_value=600, max_value=3600, 
                             value=st.session_state.shared_context["rpm"], step=10)
        
        # Service criticality
        service_type = st.selectbox("Service Criticality", 
                                   ["Critical (Process)", "Essential (Utility)", "Standby"],
                                   index=["Critical (Process)", "Essential (Utility)", "Standby"].index(
                                       st.session_state.shared_context["service_criticality"]))
        
        # Fluid type (BBM specific)
        fluid_type = st.selectbox("Fluid Type (BBM)", 
                                 list(FLUID_PROPERTIES.keys()),
                                 index=list(FLUID_PROPERTIES.keys()).index(
                                     st.session_state.shared_context["fluid_type"]))
        
        # Update session state
        st.session_state.shared_context.update({
            "machine_id": machine_id,
            "rpm": rpm,
            "service_criticality": service_type,
            "fluid_type": fluid_type
        })
        
        # Context info box
        fluid_props = FLUID_PROPERTIES[fluid_type]
        st.info(f"""
        **Fluid Properties ({fluid_type}):**
        • SG: {fluid_props['sg']}
        • Vapor Pressure @38°C: {fluid_props['vapor_pressure_kpa_38C']} kPa
        • Risk Level: {fluid_props['risk_level']}
        """)
        
        st.divider()
        
        # Quick navigation
        st.subheader("🧭 Quick Navigation")
        st.page_link("#mechanical-tab", label="🔧 Mechanical", icon="🔧")
        st.page_link("#hydraulic-tab", label="💧 Hydraulic", icon="💧")
        st.page_link("#electrical-tab", label="⚡ Electrical", icon="⚡")
        st.page_link("#integrated-tab", label="🔗 Integrated", icon="🔗")
    
    # Main tab navigation
    tab_mech, tab_hyd, tab_elec, tab_integrated = st.tabs([
        "🔧 Mechanical", "💧 Hydraulic", "⚡ Electrical", "🔗 Integrated Summary"
    ])
    
    # ========================================================================
    # TAB 1: MECHANICAL VIBRATION ANALYSIS
    # ========================================================================
    with tab_mech:
        st.header("🔧 Mechanical Vibration Analysis")
        st.caption("ISO 10816-3/7 | Centrifugal Pump + Electric Motor | Fixed Speed")
        
        # FFT mode selection
        col1, col2 = st.columns([2, 1])
        with col1:
            fft_mode = st.radio(
                "Mode Analisis FFT",
                options=["🚀 Efisien (Hanya Titik Bermasalah)", "🔍 Lengkap (Semua 12 Titik)"],
                index=0 if service_type != "Critical (Process)" else 1,
                help="Efisien: cepat untuk routine. Lengkap: wajib untuk critical equipment / troubleshooting"
            )
        
        with col2:
            rpm_hz = rpm / 60
            st.info(f"""
            **Konfigurasi:**
            • 1×RPM = {rpm_hz:.1f} Hz
            • Zone B Limit = 4.5 mm/s
            • Mode: {fft_mode}
            """)
        
        # 12-point measurement input
        st.subheader("📊 Input Data 12 Titik Pengukuran")
        points = [f"{machine} {end} {direction}" 
                 for machine in ["Pump", "Motor"] 
                 for end in ["DE", "NDE"] 
                 for direction in ["Horizontal", "Vertical", "Axial"]]
        
        input_data = {}
        bands_inputs = {}
        cols = st.columns(3)
        
        for idx, point in enumerate(points):
            with cols[idx % 3]:
                with st.expander(f"📍 {point}", expanded=False):
                    overall = st.number_input("Overall Velocity (mm/s)", min_value=0.0, max_value=20.0, 
                                             value=1.0, step=0.1, key=f"mech_vel_{point}")
                    input_data[point] = overall
                    
                    st.caption("🔹 Frequency Bands - Acceleration (g)")
                    b1 = st.number_input("Band 1: 0.5-1.5 kHz", min_value=0.0, value=0.2, step=0.05, key=f"mech_b1_{point}")
                    b2 = st.number_input("Band 2: 1.5-5 kHz", min_value=0.0, value=0.15, step=0.05, key=f"mech_b2_{point}")
                    b3 = st.number_input("Band 3: 5-16 kHz", min_value=0.0, value=0.1, step=0.05, key=f"mech_b3_{point}")
                    bands_inputs[point] = {"Band1": b1, "Band2": b2, "Band3": b3}
                    
                    # Real-time feedback
                    if overall > ISO_LIMITS_VELOCITY["Zone B (Acceptable)"]:
                        st.warning(f"⚠️ >4.5 mm/s")
                    if b3 > 2*ACCEL_BASELINE["Band3 (5-16kHz)"]:
                        st.error(f"🔴 Band 3 tinggi")
        
        # FFT input section (dynamic based on mode)
        flagged_points = [p for p in points if input_data[p] > ISO_LIMITS_VELOCITY["Zone B (Acceptable)"]]
        bearing_alert_points = [p for p in points if bands_inputs[p]["Band3"] > 2*ACCEL_BASELINE["Band3 (5-16kHz)"]]
        
        if fft_mode == "🔍 Lengkap (Semua 12 Titik)":
            targets = points
        else:
            targets = list(set(flagged_points + bearing_alert_points))
        
        fft_inputs = {}
        if targets:
            with st.expander("📈 Input FFT Spectrum (Top 3 Peaks)", expanded=len(targets)>0):
                tabs = st.tabs(targets) if targets else []
                for idx, point in enumerate(targets):
                    with tabs[idx]:
                        st.write(f"**{point}** | Vel: {input_data[point]:.2f} mm/s | B3: {bands_inputs[point]['Band3']:.3f} g")
                        peaks = []
                        for i in range(1, 4):
                            c1, c2 = st.columns(2)
                            with c1:
                                default_freq = rpm_hz * i if i <= 2 else rpm_hz
                                freq = st.number_input(f"Peak {i} Freq (Hz)", min_value=0.1, value=default_freq, key=f"mech_f_{point}_{i}")
                            with c2:
                                amp = st.number_input(f"Peak {i} Amp (mm/s)", min_value=0.01, value=1.0, step=0.1, key=f"mech_a_{point}_{i}")
                            peaks.append((freq, amp))
                        fft_inputs[point] = peaks
        
        # Analysis trigger
        if st.button("🔍 Jalankan Mechanical Analysis", type="primary", key="run_mech"):
            with st.spinner("Menganalisis data vibration..."):
                # Diagnose each point
                point_results = []
                for point in points:
                    peaks = fft_inputs.get(point, [(rpm_hz,0.1),(2*rpm_hz,0.05),(3*rpm_hz,0.02)])
                    bands = bands_inputs[point]
                    has_fft = point in fft_inputs
                    result = diagnose_single_point_mechanical(peaks, bands, rpm_hz, point, input_data[point], has_fft)
                    result["point"] = point
                    result["location_hint"] = f"{'Pump' if 'Pump' in point else 'Motor'} {point.split()[1]}"
                    point_results.append(result)
                
                # Aggregate system diagnosis
                mech_system = {
                    "diagnosis": "Normal",
                    "confidence": 99,
                    "severity": "Low",
                    "fault_type": "normal",
                    "domain": "mechanical"
                }
                
                # Simple aggregation: take worst case for demo
                high_sev_results = [r for r in point_results if r["severity"] == "High"]
                if high_sev_results:
                    worst = max(high_sev_results, key=lambda x: x["confidence"])
                    mech_system.update({
                        "diagnosis": worst["diagnosis"],
                        "confidence": worst["confidence"],
                        "severity": "High",
                        "fault_type": worst["fault_type"]
                    })
                elif any(r["severity"] == "Medium" for r in point_results):
                    med_results = [r for r in point_results if r["severity"] == "Medium"]
                    avg_conf = int(np.mean([r["confidence"] for r in med_results]))
                    mech_system.update({
                        "diagnosis": med_results[0]["diagnosis"],
                        "confidence": avg_conf,
                        "severity": "Medium",
                        "fault_type": med_results[0]["fault_type"]
                    })
                
                # Store for integrated analysis
                st.session_state.mech_result = mech_system
                st.session_state.mech_data = {
                    "points": {p: {"velocity": input_data[p], "bands": bands_inputs[p], 
                                  "diagnosis": next((r["diagnosis"] for r in point_results if r["point"]==p), "Normal"),
                                  "confidence": next((r["confidence"] for r in point_results if r["point"]==p), 0),
                                  "severity": next((r["severity"] for r in point_results if r["point"]==p), "Low")} 
                           for p in points},
                    "system_diagnosis": mech_system["diagnosis"]
                }
                
                st.success(f"✅ Mechanical Analysis Complete: {mech_system['diagnosis']} ({mech_system['confidence']}%)")
        
        # Display results if available
        if hasattr(st.session_state, "mech_result"):
            result = st.session_state.mech_result
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Diagnosis", result["diagnosis"], delta=f"{result['confidence']}%")
            with col_b:
                st.metric("Severity", {"Low":"🟢","Medium":"🟠","High":"🔴"}.get(result["severity"],"⚪"))
            with col_c:
                st.metric("Domain", "Mechanical")
            
            if result["diagnosis"] != "Normal":
                st.info(get_mechanical_recommendation(result["diagnosis"], "Pump-Motor System", result["severity"]))
    
    # ========================================================================
    # TAB 2: HYDRAULIC TROUBLESHOOTING (Single-Point Steady-State)
    # ========================================================================
    with tab_hyd:
        st.header("💧 Hydraulic Troubleshooting")
        st.caption("Single-Point Steady-State Measurement | BBM Fluid Types")
        
        # Steady-state verification (quality gate)
        with st.expander("✅ Verifikasi Steady-State (Wajib)", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                steady_15min = st.checkbox("Sistem stabil ≥15 menit setelah start-up/load change")
                steady_pressure = st.checkbox("Fluktuasi pressure < ±2% selama 2 menit")
                steady_flow = st.checkbox("Fluktuasi flow < ±3% selama 2 menit")
            with col2:
                steady_valve = st.checkbox("Tidak ada perubahan posisi valve selama pengukuran")
                steady_load = st.checkbox("Kondisi beban proses konfirm stabil")
                steady_time = st.text_input("Waktu Verifikasi Steady (HH:MM)", value="10:28")
            
            steady_verified = all([steady_15min, steady_pressure, steady_flow, steady_valve, steady_load])
            if not steady_verified:
                st.warning("⚠️ Lengkapi verifikasi steady-state untuk mengaktifkan analisis")
        
        # Primary hydraulic parameters (single column, high precision)
        st.subheader("📊 Data Primer Hidrolik")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            suction_pressure = st.number_input("Suction Pressure", min_value=0.0, value=1.2, step=0.1, format="%.1f")
            discharge_pressure = st.number_input("Discharge Pressure", min_value=0.0, value=8.5, step=0.1, format="%.1f")
            delta_p = discharge_pressure - suction_pressure
            st.metric("Differential Pressure", f"{delta_p:.1f} bar")
        
        with col2:
            flow_rate = st.number_input("Flow Rate", min_value=0.0, value=100.0, step=1.0, format="%.1f")
            motor_power = st.number_input("Motor Power Input", min_value=0.0, value=45.0, step=0.5, format="%.1f")
            fluid_temp = st.number_input("Fluid Temperature", min_value=0, max_value=100, value=40, step=1)
        
        with col3:
            # Auto-populate fluid properties based on selection
            fluid_props = FLUID_PROPERTIES[fluid_type]
            sg = st.number_input("Specific Gravity (SG)", min_value=0.5, max_value=1.5, value=fluid_props["sg"], step=0.01, disabled=False)
            st.caption(f"Default: {fluid_props['sg']} untuk {fluid_type}")
            
            # Real-time head calculation
            if delta_p > 0 and sg > 0:
                head_calc = delta_p * 10.2 / sg
                st.metric("Calculated Head", f"{head_calc:.1f} m")
        
        # Design reference parameters
        with st.expander("📋 Data Referensi Desain (OEM)"):
            col1, col2 = st.columns(2)
            with col1:
                rated_flow = st.number_input("Rated Flow (Q_design)", min_value=0.0, value=120.0, step=1.0)
                rated_head = st.number_input("Rated Head (H_design)", min_value=0.0, value=85.0, step=0.5)
            with col2:
                bep_efficiency = st.number_input("BEP Efficiency", min_value=0, max_value=100, value=78, step=1)
                npsh_required = st.number_input("NPSH Required @ Q_aktual", min_value=0.0, value=4.2, step=0.1)
        
        # Secondary observations (checklist)
        st.subheader("🔍 Observasi Sekunder")
        col1, col2 = st.columns(2)
        
        with col1:
            st.caption("Kondisi Visual & Fluida")
            leakage = st.radio("Kebocoran Seal", ["Tidak ada", "Minor (≤2 tetes/menit)", "Mayor"], index=0)
            fluid_condition = st.radio("Kondisi Fluida", ["Jernih", "Agak keruh", "Keruh/partikel", "Berbusa"], index=0)
        
        with col2:
            st.caption("Kebisingan & Getaran")
            noise_type = st.radio("Jenis Noise", ["Normal", "Whining", "Grinding", "Crackling"], index=0)
            vibration_qual = st.radio("Getaran (kualitatif)", ["Tidak terasa", "Halus", "Jelas", "Kuat"], index=0)
        
        # Operational context
        with st.expander("📋 Konteks Operasional"):
            valve_position = st.slider("Valve Discharge Position", 0, 100, 100, format="%d%%")
            recent_changes = st.text_area("Recent Changes (7 hari terakhir)", placeholder="Contoh: ganti fluida, maintenance, dll")
            operator_complaint = st.text_area("Keluhan Operator (verbatim)", placeholder="Tulis persis seperti disampaikan")
        
        # Analysis trigger
        analyze_hyd_disabled = not steady_verified or suction_pressure >= discharge_pressure
        if st.button("💧 Generate Hydraulic Diagnosis", type="primary", key="run_hyd", disabled=analyze_hyd_disabled):
            if not steady_verified:
                st.error("❌ Verifikasi steady-state belum lengkap")
                st.stop()
            
            with st.spinner("Menganalisis performa hidrolik..."):
                # Calculate hydraulic parameters
                hyd_calc = calculate_hydraulic_parameters(
                    suction_pressure, discharge_pressure, flow_rate, motor_power, sg, fluid_temp
                )
                
                # Prepare inputs for diagnosis
                design_params = {
                    "rated_flow_m3h": rated_flow,
                    "rated_head_m": rated_head,
                    "bep_efficiency": bep_efficiency,
                    "npsh_required_m": npsh_required
                }
                
                observations = {
                    "leakage": leakage,
                    "fluid_condition": fluid_condition,
                    "noise_type": noise_type
                }
                
                context = {
                    "flow_aktual": flow_rate,
                    "suction_pressure_bar": suction_pressure
                }
                
                # Run diagnosis
                hyd_result = diagnose_hydraulic_single_point(
                    hyd_calc, design_params, fluid_props, observations, context
                )
                
                # Store for integrated analysis
                st.session_state.hyd_result = hyd_result
                st.session_state.hyd_data = {
                    "measurements": {
                        "suction_pressure": suction_pressure,
                        "discharge_pressure": discharge_pressure,
                        "flow_rate": flow_rate,
                        "motor_power": motor_power,
                        "power_factor": 0.85  # default, can be added as input
                    },
                    "fluid_type": fluid_type,
                    "sg": sg,
                    "head_m": hyd_calc["head_m"],
                    "efficiency_percent": hyd_calc["efficiency_percent"],
                    "npsh_margin_m": hyd_result["details"].get("npsh_margin_m", 0),
                    "diagnosis": hyd_result["diagnosis"],
                    "confidence": hyd_result["confidence"],
                    "severity": hyd_result["severity"]
                }
                
                st.success(f"✅ Hydraulic Analysis Complete: {hyd_result['diagnosis']} ({hyd_result['confidence']}%)")
        
        # Display results if available
        if hasattr(st.session_state, "hyd_result"):
            result = st.session_state.hyd_result
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Diagnosis", result["diagnosis"], delta=f"{result['confidence']}%")
            with col_b:
                st.metric("Severity", {"Low":"🟢","Medium":"🟠","High":"🔴"}.get(result["severity"],"⚪"))
            with col_c:
                st.metric("Domain", "Hydraulic")
            
            if result["diagnosis"] != "NORMAL_OPERATION":
                st.info(get_hydraulic_recommendation(result["diagnosis"], fluid_type, result["severity"]))
            
            # Show NPSH margin warning if critical
            npsh_margin = result["details"].get("npsh_margin_m", 999)
            if npsh_margin < 0.5:
                st.warning(f"""
                ⚠️ **NPSH Margin Critical**: {npsh_margin:.2f} m
                • Risk of cavitation untuk {fluid_type} (vapor pressure tinggi)
                • Rekomendasi: tingkatkan suction pressure atau turunkan fluid temperature
                """)
    
    # ========================================================================
    # TAB 3: ELECTRICAL CONDITION ANALYSIS
    # ========================================================================
    with tab_elec:
        st.header("⚡ Electrical Condition Analysis")
        st.caption("3-Phase Voltage/Current | Unbalance Detection | Motor Load Estimation")
        
        # Motor specifications
        with st.expander("⚙️ Motor Specifications (Nameplate)", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                rated_voltage = st.number_input("Rated Voltage (V)", min_value=200, max_value=690, value=400, step=10)
                rated_current = st.number_input("Rated Current (A)", min_value=10, max_value=500, value=85, step=5)
                rated_power = st.number_input("Rated Power (kW)", min_value=1, max_value=500, value=45, step=1)
            with col2:
                motor_efficiency = st.number_input("Motor Efficiency (%)", min_value=70, max_value=98, value=92, step=1)
                connection_type = st.radio("Connection", ["Star", "Delta"], index=0)
                pf_default = st.number_input("Expected Power Factor", min_value=0.5, max_value=1.0, value=0.85, step=0.05)
        
        # 3-phase measurements
        st.subheader("📊 Pengukuran 3-Phase")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.caption("Voltage (Line-to-Line)")
            v_l1l2 = st.number_input("L1-L2 (V)", min_value=0.0, value=400.0, step=1.0)
            v_l2l3 = st.number_input("L2-L3 (V)", min_value=0.0, value=402.0, step=1.0)
            v_l3l1 = st.number_input("L3-L1 (V)", min_value=0.0, value=398.0, step=1.0)
        
        with col2:
            st.caption("Current (Per Phase)")
            i_l1 = st.number_input("L1 (A)", min_value=0.0, value=82.0, step=0.5)
            i_l2 = st.number_input("L2 (A)", min_value=0.0, value=84.0, step=0.5)
            i_l3 = st.number_input("L3 (A)", min_value=0.0, value=83.0, step=0.5)
        
        with col3:
            st.caption("Power & Frequency")
            power_factor = st.number_input("Power Factor", min_value=0.0, max_value=1.0, value=pf_default, step=0.01)
            frequency = st.number_input("Frequency (Hz)", min_value=45.0, max_value=65.0, value=50.0, step=0.1)
            # Optional: direct power measurement
            measured_power = st.number_input("Active Power (kW) - Optional", min_value=0.0, value=0.0, step=0.5)
        
        # Real-time calculations display
        with st.expander("📈 Perhitungan Real-Time", expanded=True):
            elec_calc = calculate_electrical_parameters(
                v_l1l2, v_l2l3, v_l3l1, i_l1, i_l2, i_l3, power_factor, rated_voltage, rated_current
            )
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Voltage Unbalance", f"{elec_calc['voltage_unbalance_percent']:.2f}%", 
                         delta="⚠️ Warning" if elec_calc['voltage_unbalance_percent'] > 2 else "✅ OK")
            with col2:
                st.metric("Current Unbalance", f"{elec_calc['current_unbalance_percent']:.2f}%",
                         delta="⚠️ Warning" if elec_calc['current_unbalance_percent'] > 5 else "✅ OK")
            with col3:
                st.metric("Load Estimate", f"{elec_calc['load_estimate_percent']:.1f}%")
            with col4:
                st.metric("Est. Power", f"{elec_calc['electrical_power_kw']:.1f} kW")
            
            # Voltage tolerance check
            if not elec_calc["voltage_within_tolerance"]:
                st.warning(f"⚠️ Voltage {elec_calc['v_avg']:.1f}V outside tolerance ({ELECTRICAL_LIMITS['voltage_tolerance_low']}-{ELECTRICAL_LIMITS['voltage_tolerance_high']}% of {rated_voltage}V)")
        
        # Observations
        st.subheader("🔍 Observasi Visual & Sensorik")
        col1, col2 = st.columns(2)
        with col1:
            motor_temp = st.radio("Motor Surface Temperature", 
                                 ["Normal (<70°C)", "Hangat (70-90°C)", "Panas (>90°C)"], index=0)
            panel_condition = st.radio("Panel/Kabel Condition", 
                                      ["Normal", "Hangat", "Panas berlebih"], index=0)
        with col2:
            noise_elec = st.radio("Electrical Noise", 
                                 ["Tidak ada", "Dengung normal", "Buzzing/arcing"], index=0)
            odor = st.radio("Bau/Asap", 
                           ["Tidak ada", "Bau isolasi", "Bau gosong"], index=0)
        
        # Analysis trigger
        if st.button("⚡ Generate Electrical Diagnosis", type="primary", key="run_elec"):
            with st.spinner("Menganalisis kondisi electrical..."):
                motor_specs = {
                    "rated_voltage": rated_voltage,
                    "rated_current": rated_current,
                    "rated_power": rated_power
                }
                
                observations = {
                    "motor_temperature": motor_temp,
                    "panel_condition": panel_condition,
                    "electrical_noise": noise_elec
                }
                
                # Run diagnosis
                elec_result = diagnose_electrical_condition(elec_calc, motor_specs, observations)
                
                # Store for integrated analysis
                st.session_state.elec_result = elec_result
                st.session_state.elec_data = {
                    "measurements": {
                        "v_l1l2": v_l1l2, "v_l2l3": v_l2l3, "v_l3l1": v_l3l1,
                        "i_l1": i_l1, "i_l2": i_l2, "i_l3": i_l3,
                        "power_factor": power_factor
                    },
                    "voltage_unbalance": elec_calc["voltage_unbalance_percent"],
                    "current_unbalance": elec_calc["current_unbalance_percent"],
                    "load_estimate": elec_calc["load_estimate_percent"],
                    "diagnosis": elec_result["diagnosis"],
                    "confidence": elec_result["confidence"],
                    "severity": elec_result["severity"]
                }
                
                st.success(f"✅ Electrical Analysis Complete: {elec_result['diagnosis']} ({elec_result['confidence']}%)")
        
        # Display results if available
        if hasattr(st.session_state, "elec_result"):
            result = st.session_state.elec_result
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Diagnosis", result["diagnosis"], delta=f"{result['confidence']}%")
            with col_b:
                st.metric("Severity", {"Low":"🟢","Medium":"🟠","High":"🔴"}.get(result["severity"],"⚪"))
            with col_c:
                st.metric("Domain", "Electrical")
            
            if result["diagnosis"] != "NORMAL_ELECTRICAL":
                st.info(get_electrical_recommendation(result["diagnosis"], result["severity"]))
    
    # ========================================================================
    # TAB 4: INTEGRATED SUMMARY & CROSS-DOMAIN CORRELATION
    # ========================================================================
    with tab_integrated:
        st.header("🔗 Integrated Diagnostic Summary")
        st.caption("Cross-Domain Correlation | QCDSM-Based Recommendations")
        
        # Check if all analyses have been run
        analyses_complete = all([
            hasattr(st.session_state, "mech_result"),
            hasattr(st.session_state, "hyd_result"),
            hasattr(st.session_state, "elec_result")
        ])
        
        if not analyses_complete:
            st.info("""
            💡 **Langkah Selanjutnya:**
            1. Jalankan analisis di tab **🔧 Mechanical**
            2. Jalankan analisis di tab **💧 Hydraulic** 
            3. Jalankan analisis di tab **⚡ Electrical**
            4. Kembali ke tab ini untuk integrated diagnosis
            
            *Data akan tersimpan otomatis di session selama browser tidak di-refresh*
            """)
            
            # Quick status overview
            col1, col2, col3 = st.columns(3)
            with col1:
                status_mech = "✅" if hasattr(st.session_state, "mech_result") else "⏳"
                st.metric("Mechanical", status_mech)
            with col2:
                status_hyd = "✅" if hasattr(st.session_state, "hyd_result") else "⏳"
                st.metric("Hydraulic", status_hyd)
            with col3:
                status_elec = "✅" if hasattr(st.session_state, "elec_result") else "⏳"
                st.metric("Electrical", status_elec)
        else:
            # Run integrated analysis
            with st.spinner("Mengintegrasikan hasil tiga domain..."):
                integrated_result = aggregate_cross_domain_diagnosis(
                    st.session_state.mech_result,
                    st.session_state.hyd_result,
                    st.session_state.elec_result,
                    st.session_state.shared_context
                )
                
                st.session_state.integrated_result = integrated_result
            
            # Executive dashboard
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Integrated Diagnosis", integrated_result["diagnosis"], 
                         delta=f"{integrated_result['confidence']}%")
            with col2:
                severity_icon = {"Low":"🟢","Medium":"🟠","High":"🔴"}.get(integrated_result["severity"],"⚪")
                st.metric("Overall Severity", f"{severity_icon} {integrated_result['severity']}")
            with col3:
                st.metric("Correlation Boost", f"+{integrated_result['confidence'] - 70}%" if integrated_result['confidence'] > 70 else "Standard")
            
            # Fault propagation map (text-based visualization)
            with st.expander("🗺️ Fault Propagation Map", expanded=True):
                st.write("**Cross-Domain Correlation Notes:**")
                for note in integrated_result["correlation_notes"]:
                    st.write(f"• {note}")
                
                # Simple text-based flow diagram
                st.write("\n**Propagation Path:**")
                st.text(f"""
    ⚡ Electrical: {st.session_state.elec_result['diagnosis']}
         │
         ▼ (correlation: {integrated_result['correlation_notes'][0][:50]}...)
    🔧 Mechanical: {st.session_state.mech_result['diagnosis']}
         │
         ▼
    💧 Hydraulic: {st.session_state.hyd_result['diagnosis']}
                """)
            
            # QCDSM Recommendations
            st.subheader("✅ QCDSM-Based Action Plan")
            qcsm = integrated_result["qcsm_recommendations"]
            
            tabs_qcdsm = st.tabs(["Quality", "Cost", "Delivery", "Safety", "Spirit"])
            with tabs_qcdsm[0]:
                for rec in qcsm["Quality"]:
                    st.write(f"• {rec}")
            with tabs_qcdsm[1]:
                for rec in qcsm["Cost"]:
                    st.write(f"• {rec}")
            with tabs_qcdsm[2]:
                for rec in qcsm["Delivery"]:
                    st.write(f"• {rec}")
            with tabs_qcdsm[3]:
                for rec in qcsm["Safety"]:
                    st.write(f"• {rec}")
            with tabs_qcdsm[4]:
                for rec in qcsm["Spirit"]:
                    st.write(f"• {rec}")
            
            # Export options
            st.divider()
            st.subheader("📥 Export Report")
            
            if st.button("📊 Generate Unified CSV Report", type="primary"):
                csv_report = generate_unified_csv_report(
                    machine_id, rpm, 
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    st.session_state.get("mech_data", {}),
                    st.session_state.get("hyd_data", {}),
                    st.session_state.get("elec_data", {}),
                    integrated_result
                )
                
                st.download_button(
                    label="📥 Download CSV Report",
                    data=csv_report,
                    file_name=f"PUMP_DIAG_{machine_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                st.success("✅ Report generated successfully!")
    
    # Footer
    st.divider()
    st.caption("""
    **Standar Acuan**: ISO 10816-3/7 | ISO 13373-1 | API 610 | NEMA MG-1  
    **Algoritma**: Hybrid rule-based dengan cross-domain correlation + confidence scoring  
    ⚠️ Decision Support System - Verifikasi oleh personnel kompeten untuk keputusan kritis  
    🏭 Pertamina Patra Niaga - Asset Integrity Management
    """)


if __name__ == "__main__":
    main()
