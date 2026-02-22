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
# ✅ Bearing Temperature Thresholds (IEC 60034-1, API 610, SKF)
BEARING_TEMP_LIMITS = {
    "normal_max": 70,        # Sama (conservative)
    "elevated_min": 70,      # Sama
    "elevated_max": 80,      # ❗ IEC lebih ketat (NEMA: 85°C)
    "warning_min": 80,       # ❗ IEC lebih ketat
    "warning_max": 90,       # ❗ IEC lebih ketat (NEMA: 95°C)
    "critical_min": 90,      # ❗ IEC lebih ketat
    "delta_threshold": 15,   # Sama
    "ambient_reference": 30  # Sama
}
# --- Hydraulic Fluid Properties (BBM Specific - Pertamina) ---
FLUID_PROPERTIES = {
    "Pertalite (RON 90)": {
        "sg": 0.73,
        "vapor_pressure_kpa_38C": 52,
        "viscosity_cst_40C": 0.6,
        "flash_point_C": -43,
        "risk_level": "High"
    },
    "Pertamax (RON 92)": {
        "sg": 0.74,
        "vapor_pressure_kpa_38C": 42,
        "viscosity_cst_40C": 0.6,
        "flash_point_C": -43,
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
# --- Electrical Thresholds (IEC 60034-1 & Practical Limits) ---
ELECTRICAL_LIMITS = {
    "voltage_unbalance_warning": 1.0,      # ❗ IEC lebih ketat (NEMA: 2.0%)
    "voltage_unbalance_critical": 2.0,     # ❗ IEC lebih ketat (NEMA: 3.5%)
    "current_unbalance_warning": 5.0,      # Sama
    "current_unbalance_critical": 8.0,     # ❗ IEC lebih ketat (NEMA: 10.0%)
    "voltage_tolerance_low": 90,           # Sama (±10%)
    "voltage_tolerance_high": 110,         # Sama (±10%)
    "current_load_warning": 90,            # Sama
    "current_load_critical": 100,          # ❗ IEC tidak ada Service Factor (NEMA: 115%)
    "service_factor": 1.0                  # ❗ IEC default SF = 1.0 (NEMA: 1.15)
}
# ============================================================================
# FUNGSI REKOMENDASI - MULTI-DOMAIN (TECHNICAL ONLY)
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
            f"• Investigasi: mechanical loss vs hydraulic loss vs fluid property mismatch\n"
            f"• Severity: {severity} → {'Plan overhaul dalam 1-3 bulan' if severity != 'Low' else 'Monitor monthly'}"
        ),
        "NORMAL_OPERATION": (
            f"✅ **{fluid_type} - Normal Operation**\n"
            f"• Semua parameter dalam batas acceptable (±5% dari design)\n"
            f"• Rekam data ini sebagai baseline untuk trend monitoring\n"
            f"• Severity: Low → Continue routine monitoring"
        ),
        "Tidak Terdiagnosa": (
            "⚠️ **Pola Tidak Konsisten**\n"
            "• Data hydraulic tidak match dengan rule standar\n"
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
            f"• Severity: {severity} → {'Coordinate dengan electrical team segera' if severity == 'High' else 'Monitor voltage trend'}"
        ),
        "OVER_VOLTAGE": (
            f"⚡ **Over Voltage Condition**\n"
            f"• Cek supply voltage di MCC: possible transformer tap issue\n"
            f"• Verify: motor rated voltage vs actual operating voltage\n"
            f"• Severity: {severity} → {'Coordinate dengan electrical team segera' if severity == 'High' else 'Monitor voltage trend'}"
        ),
        "VOLTAGE_UNBALANCE": (
            f"⚡ **Voltage Unbalance Detected**\n"
            f"• Cek 3-phase supply balance di source: possible single-phase loading\n"
            f"• Inspect: loose connection, corroded terminal, faulty breaker\n"
            f"• Severity: {severity} → {'Balance supply sebelum mechanical damage' if severity != 'Low' else 'Monitor monthly'}"
        ),
        "ELECTRICAL_ARCING": (
            f"🔴 **Electrical Arcing Detected**\n"
            f"• ⚠️ IMMEDIATE SAFETY RISK - Potential fire/explosion hazard\n"
            f"• Periksa: loose connection, corroded terminal, damaged cable insulation\n"
            f"• Severity: HIGH → Immediate shutdown & electrical inspection required"
        ),
        "INSULATION_OVERHEAT": (
            f"🔴 **Insulation Overheat / Breakdown**\n"
            f"• ⚠️ RISK OF MOTOR FAILURE - Insulation degradation detected\n"
            f"• Cek: megger test insulation resistance, winding temperature\n"
            f"• Severity: HIGH → Schedule inspection within 24-48 hours"
        ),
        "CONNECTION_OVERHEAT": (
            f"🔴 **Electrical Connection Overheat**\n"
            f"• Periksa: terminal tightness, contact resistance, cable sizing\n"
            f"• Inspect: contactor, breaker, fuse connections di MCC\n"
            f"• Severity: HIGH → Tighten/replace connections before continue operation"
        ),
        "MOTOR_OVERHEAT": (
            f"🟠 **Motor Overheat Condition**\n"
            f"• Cek: motor ventilation, cooling fan, ambient temperature\n"
            f"• Verify: load vs rated capacity, service factor\n"
            f"• Inspect: bearing condition (friction can cause heating)\n"
            f"• Severity: MEDIUM-HIGH → Reduce load if possible, investigate root cause"
        ),
        "CURRENT_UNBALANCE": (
            f"⚡ **Current Unbalance Detected**\n"
            f"• Investigasi: winding fault, rotor bar issue, atau supply problem\n"
            f"• Cek insulation resistance & winding resistance balance\n"
            f"• Severity: {severity} → {'Schedule electrical inspection' if severity != 'Low' else 'Continue monitoring'}"
        ),
        "OVER_LOAD": (
            f"⚡ **Over Load Condition**\n"
            f"• Motor operating above FLA rating\n"
            f"• Verify: process load, mechanical binding, or electrical issue\n"
            f"• Severity: {severity} → {'Reduce load immediately' if severity == 'High' else 'Monitor trend closely'}"
        ),
        "UNDER_LOAD": (
            f"⚡ **Under Load Condition**\n"
            f"• Motor operating below 50% FLA\n"
            f"• Verify: process demand, pump sizing, or system resistance\n"
            f"• Severity: Low → Review operating point vs BEP"
        ),
        "NORMAL_ELECTRICAL": (
            f"✅ **Normal Electrical Condition**\n"
            f"• Voltage balance <2%, current balance <5%, within rated limits\n"
            f"• Severity: Low → Continue routine electrical monitoring"
        ),
        "Tidak Terdiagnosa": (
            "⚠️ **Pola Tidak Konsisten**\n"
            "• Data electrical tidak match dengan rule standar\n"
            "• Rekomendasi: Verifikasi dengan power quality analyzer + cross-check domain lain"
        )
    }
    return rec_map.get(diagnosis, rec_map["Tidak Terdiagnosa"])

# ============================================================================
# FUNGSI TEMPERATURE ANALYSIS
# ============================================================================
def get_temperature_status(temp_celsius):
    if temp_celsius < BEARING_TEMP_LIMITS["normal_max"]:
        return "Normal", "🟢", 0
    elif temp_celsius < BEARING_TEMP_LIMITS["elevated_max"]:
        return "Elevated", "🟡", 0
    elif temp_celsius < BEARING_TEMP_LIMITS["warning_max"]:
        return "Warning", "🟠", 1
    else:
        return "Critical", "🔴", 2

def calculate_temperature_confidence_adjustment(temp_dict, diagnosis_consistent):
    adjustment = 0
    notes = []
    for location, temp in temp_dict.items():
        if temp is None or temp == 0:
            continue
        status, color, sev_level = get_temperature_status(temp)
        if status == "Critical":
            if diagnosis_consistent:
                adjustment += 20
                notes.append(f"⚠️ {location}: {temp}°C (Critical) - Strong thermal confirmation")
            else:
                adjustment -= 10
                notes.append(f"⚠️ {location}: {temp}°C (Critical) - Review required")
        elif status == "Warning":
            if diagnosis_consistent:
                adjustment += 15
                notes.append(f"⚠️ {location}: {temp}°C (Warning) - Thermal confirmation")
            else:
                adjustment -= 5
                notes.append(f"⚠️ {location}: {temp}°C (Warning) - Monitor closely")
        elif status == "Elevated":
            if diagnosis_consistent:
                adjustment += 10
                notes.append(f"📈 {location}: {temp}°C (Elevated) - Early thermal indication")
            else:
                notes.append(f"📈 {location}: {temp}°C (Elevated) - Monitor trend")
    if temp_dict.get("Pump_DE") and temp_dict.get("Pump_NDE"):
        delta_pump = abs(temp_dict["Pump_DE"] - temp_dict["Pump_NDE"])
        if delta_pump > BEARING_TEMP_LIMITS["delta_threshold"]:
            adjustment += 5
            notes.append(f"🔍 Pump DE-NDE ΔT: {delta_pump}°C (>15°C) - Localized fault indicated")
    if temp_dict.get("Motor_DE") and temp_dict.get("Motor_NDE"):
        delta_motor = abs(temp_dict["Motor_DE"] - temp_dict["Motor_NDE"])
        if delta_motor > BEARING_TEMP_LIMITS["delta_threshold"]:
            adjustment += 5
            notes.append(f"🔍 Motor DE-NDE ΔT: {delta_motor}°C (>15°C) - Localized fault indicated")
    if temp_dict.get("Motor_DE") and temp_dict.get("Pump_DE"):
        if temp_dict["Motor_DE"] > temp_dict["Pump_DE"] + 10:
            notes.append("⚡ Motor DE > Pump DE - Possible electrical origin")
    return min(20, max(-10, adjustment)), notes

# ============================================================================
# FUNGSI PERHITUNGAN - HYDRAULIC DOMAIN
# ============================================================================
def calculate_hydraulic_parameters(suction_pressure, discharge_pressure, flow_rate,
                                   motor_power, sg, fluid_temp_c=40):
    delta_p = discharge_pressure - suction_pressure
    head = delta_p * 10.2 / sg if sg > 0 else 0
    hydraulic_power = (flow_rate * head * sg * 9.81) / 3600 if flow_rate > 0 and head > 0 else 0
    efficiency = (hydraulic_power / motor_power * 100) if motor_power > 0 else 0
    return {
        "delta_p_bar": delta_p,
        "head_m": head,
        "hydraulic_power_kw": hydraulic_power,
        "efficiency_percent": efficiency
    }

def classify_hydraulic_performance(head_aktual, head_design, efficiency_aktual,
                                   efficiency_bep, flow_aktual, flow_design):
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
# FUNGSI PERHITUNGAN - ELECTRICAL DOMAIN (SIMPLIFIED)
# ============================================================================
def calculate_electrical_parameters(v_l1l2, v_l2l3, v_l3l1, i_l1, i_l2, i_l3,
                                    rated_voltage, fla):
    """
    Simplified electrical calculation - Voltage, Current, Load only
    No Power Factor, No Active Power calculation
    """
    v_avg = (v_l1l2 + v_l2l3 + v_l3l1) / 3
    i_avg = (i_l1 + i_l2 + i_l3) / 3
    v_deviations = [abs(v - v_avg) for v in [v_l1l2, v_l2l3, v_l3l1]]
    voltage_unbalance = (max(v_deviations) / v_avg * 100) if v_avg > 0 else 0
    i_deviations = [abs(i - i_avg) for i in [i_l1, i_l2, i_l3]]
    current_unbalance = (max(i_deviations) / i_avg * 100) if i_avg > 0 else 0
    load_estimate = (i_avg / fla * 100) if fla > 0 else 0
    voltage_within_tolerance = (ELECTRICAL_LIMITS["voltage_tolerance_low"] <=
                                (v_avg / rated_voltage * 100) <=
                                ELECTRICAL_LIMITS["voltage_tolerance_high"])
    return {
        "v_avg": v_avg,
        "i_avg": i_avg,
        "voltage_unbalance_percent": voltage_unbalance,
        "current_unbalance_percent": current_unbalance,
        "load_estimate_percent": load_estimate,
        "voltage_within_tolerance": voltage_within_tolerance
    }

def classify_electrical_condition(voltage_unbalance, current_unbalance,
                                  load_estimate, voltage_within_tolerance,
                                  rated_voltage, v_avg):
    """
    Simplified electrical classification - Voltage, Current, Load only
    """
    if not voltage_within_tolerance and v_avg < rated_voltage:
        severity = "High" if load_estimate > 80 else "Medium"
        return "UNDER_VOLTAGE", 70, severity
    elif not voltage_within_tolerance and v_avg > rated_voltage * 1.1:
        return "OVER_VOLTAGE", 70, "Medium"
    if voltage_unbalance > ELECTRICAL_LIMITS["voltage_unbalance_critical"]:
        return "VOLTAGE_UNBALANCE", 75, "High"
    elif voltage_unbalance > ELECTRICAL_LIMITS["voltage_unbalance_warning"]:
        return "VOLTAGE_UNBALANCE", 65, "Medium"
    if current_unbalance > ELECTRICAL_LIMITS["current_unbalance_critical"]:
        return "CURRENT_UNBALANCE", 70, "High"
    elif current_unbalance > ELECTRICAL_LIMITS["current_unbalance_warning"]:
        return "CURRENT_UNBALANCE", 60, "Medium"
    if load_estimate > ELECTRICAL_LIMITS["current_load_critical"]:
        return "OVER_LOAD", 55, "Medium"
    elif load_estimate < 50:
        return "UNDER_LOAD", 50, "Low"
    return "NORMAL_ELECTRICAL", 95, "Low"

# ============================================================================
# FUNGSI DIAGNOSA - MECHANICAL DOMAIN
# ============================================================================
def diagnose_single_point_mechanical(peaks, bands, rpm_hz, point, overall_vel,
                                     has_fft: bool = True, bearing_temp=None):
    result = {
        "diagnosis": "Normal",
        "confidence": 0,
        "severity": "Low",
        "fault_type": None,
        "domain": "mechanical",
        "temperature_status": None,
        "temperature_notes": []
    }
    if bearing_temp is not None and bearing_temp > 0:
        temp_status, temp_color, temp_sev = get_temperature_status(bearing_temp)
        result["temperature_status"] = temp_status
        result["temperature_notes"].append(f"Bearing Temp: {bearing_temp}°C ({temp_status})")
    if has_fft:
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
    b1, b2, b3 = bands["Band1"], bands["Band2"], bands["Band3"]
    base1, base2, base3 = ACCEL_BASELINE["Band1 (0.5-1.5kHz)"], ACCEL_BASELINE["Band2 (1.5-5kHz)"], ACCEL_BASELINE["Band3 (5-16kHz)"]
    if b3 > 2.0 * base3 and b2 < 1.5 * base2 and b1 < 1.5 * base1:
        result["diagnosis"] = "BEARING_EARLY"
        conf_boost = 10 if has_fft else 0
        if bearing_temp and bearing_temp > BEARING_TEMP_LIMITS["elevated_min"]:
            conf_boost += 10
        result["confidence"] = min(85, 60 + int((b3/base3 - 2) * 10) + conf_boost)
        result["severity"] = "Medium" if b3 > 3*base3 else "Low"
        result["fault_type"] = "high_freq"
        return result
    if b2 > 2.0 * base2 and b3 > 1.5 * base3 and b1 < 1.5 * base1:
        result["diagnosis"] = "BEARING_DEVELOPED"
        conf_boost = 10 if has_fft else 0
        if bearing_temp and bearing_temp > BEARING_TEMP_LIMITS["elevated_min"]:
            conf_boost += 10
        result["confidence"] = min(90, 70 + int((b2/base2 - 2) * 8) + conf_boost)
        result["severity"] = "High" if b2 > 3*base2 else "Medium"
        result["fault_type"] = "high_freq"
        return result
    if b1 > 2.5 * base1 and b2 > 1.5 * base2:
        result["diagnosis"] = "BEARING_SEVERE"
        result["confidence"] = min(95, 80 + int((b1/base1 - 2.5) * 6))
        result["severity"] = "High"
        result["fault_type"] = "high_freq"
        return result
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
# FUNGSI DIAGNOSA - HYDRAULIC DOMAIN
# ============================================================================
def diagnose_hydraulic_single_point(hydraulic_calc, design_params, fluid_props,
                                    observations, context):
    result = {
        "diagnosis": "NORMAL_OPERATION",
        "confidence": 0,
        "severity": "Low",
        "fault_type": None,
        "domain": "hydraulic",
        "details": {}
    }
    head_aktual = hydraulic_calc.get("head_m", 0)
    eff_aktual = hydraulic_calc.get("efficiency_percent", 0)
    head_design = design_params.get("rated_head_m", 0)
    eff_bep = design_params.get("bep_efficiency", 0)
    flow_design = design_params.get("rated_flow_m3h", 0)
    flow_aktual = context.get("flow_aktual", 0)
    pattern, deviations = classify_hydraulic_performance(
        head_aktual, head_design, eff_aktual, eff_bep, flow_aktual, flow_design
    )
    result["details"]["deviations"] = deviations
    suction_pressure_bar = context.get("suction_pressure_bar", 0)
    vapor_pressure_kpa = fluid_props.get("vapor_pressure_kpa_38C", 0)
    sg = fluid_props.get("sg", 0.84)
    p_suction_abs_kpa = (suction_pressure_bar + 1.013) * 100
    p_vapor_kpa = vapor_pressure_kpa
    npsha_estimated = (p_suction_abs_kpa - p_vapor_kpa) / (sg * 9.81) if sg > 0 else 0
    npshr = design_params.get("npsh_required_m", 0)
    npsh_margin = npsha_estimated - npshr
    result["details"]["npsh_margin_m"] = npsh_margin
    noise_type = observations.get("noise_type", "Normal")
    fluid_condition = observations.get("fluid_condition", "Jernih")
    if noise_type == "Crackling" and npsh_margin < 0.5:
        result["diagnosis"] = "CAVITATION"
        result["confidence"] = min(90, 70 + int((0.5 - npsh_margin) * 20) if npsh_margin < 0.5 else 70)
        result["severity"] = "High" if npsh_margin < 0.3 else "Medium"
        result["fault_type"] = "cavitation"
        return result
    if pattern == "UNDER_PERFORMANCE" and noise_type == "Normal" and fluid_condition in ["Jernih", "Agak keruh"]:
        result["diagnosis"] = "IMPELLER_WEAR"
        result["confidence"] = min(85, 60 + int(abs(deviations.get("head_dev", 0)) * 2))
        result["severity"] = "High" if deviations.get("head_dev", 0) < -15 else "Medium"
        result["fault_type"] = "wear"
        return result
    if pattern == "OVER_RESISTANCE":
        result["diagnosis"] = "SYSTEM_RESISTANCE_HIGH"
        result["confidence"] = 75
        result["severity"] = "Medium"
        result["fault_type"] = "system"
        return result
    if pattern == "EFFICIENCY_DROP":
        result["diagnosis"] = "EFFICIENCY_DROP"
        result["confidence"] = min(80, 65 + int(abs(deviations.get("eff_dev", 0))))
        result["severity"] = "High" if deviations.get("eff_dev", 0) < -20 else "Medium"
        result["fault_type"] = "efficiency"
        return result
    if pattern == "NORMAL":
        result["diagnosis"] = "NORMAL_OPERATION"
        result["confidence"] = 95
        result["severity"] = "Low"
        result["fault_type"] = "normal"
        return result
    result["diagnosis"] = "Tidak Terdiagnosa"
    result["confidence"] = 40
    result["severity"] = "Medium"
    result["fault_type"] = "unknown"
    return result

# ============================================================================
# FUNGSI DIAGNOSA - ELECTRICAL DOMAIN (SIMPLIFIED)
# ============================================================================
def diagnose_electrical_condition(electrical_calc, motor_specs):
    """
    Simplified electrical diagnosis - Voltage, Current, Load only
    No temperature/odor/noise observations required
    """
    result = {
        "diagnosis": "NORMAL_ELECTRICAL",
        "confidence": 0,
        "severity": "Low",
        "fault_type": None,
        "domain": "electrical",
        "details": {}
    }
    voltage_unbalance = electrical_calc.get("voltage_unbalance_percent", 0)
    current_unbalance = electrical_calc.get("current_unbalance_percent", 0)
    load_estimate = electrical_calc.get("load_estimate_percent", 0)
    voltage_within_tolerance = electrical_calc.get("voltage_within_tolerance", True)
    v_avg = electrical_calc.get("v_avg", 0)
    rated_voltage = motor_specs.get("rated_voltage", 400)
    
    # 1. Under/Over Voltage
    if not voltage_within_tolerance:
        if v_avg < rated_voltage * 0.9:
            result["diagnosis"] = "UNDER_VOLTAGE"
            result["confidence"] = 70
            result["severity"] = "High" if load_estimate > 80 else "Medium"
            result["fault_type"] = "voltage"
        elif v_avg > rated_voltage * 1.1:
            result["diagnosis"] = "OVER_VOLTAGE"
            result["confidence"] = 70
            result["severity"] = "Medium"
            result["fault_type"] = "voltage"
        result["details"] = {
            "voltage_unbalance": voltage_unbalance,
            "current_unbalance": current_unbalance,
            "load_estimate": load_estimate
        }
        return result
    
    # 2. Voltage Unbalance
    if voltage_unbalance > ELECTRICAL_LIMITS["voltage_unbalance_critical"]:
        result["diagnosis"] = "VOLTAGE_UNBALANCE"
        result["confidence"] = 75
        result["severity"] = "High"
        result["fault_type"] = "voltage"
    elif voltage_unbalance > ELECTRICAL_LIMITS["voltage_unbalance_warning"]:
        result["diagnosis"] = "VOLTAGE_UNBALANCE"
        result["confidence"] = 65
        result["severity"] = "Medium"
        result["fault_type"] = "voltage"
    else:
        # 3. Current Unbalance
        if current_unbalance > ELECTRICAL_LIMITS["current_unbalance_critical"]:
            result["diagnosis"] = "CURRENT_UNBALANCE"
            result["confidence"] = 70
            result["severity"] = "High"
            result["fault_type"] = "current"
        elif current_unbalance > ELECTRICAL_LIMITS["current_unbalance_warning"]:
            result["diagnosis"] = "CURRENT_UNBALANCE"
            result["confidence"] = 60
            result["severity"] = "Medium"
            result["fault_type"] = "current"
        else:
            # 4. Load Capacity
            if load_estimate > ELECTRICAL_LIMITS["current_load_critical"]:
                result["diagnosis"] = "OVER_LOAD"
                result["confidence"] = 55
                result["severity"] = "Medium"
                result["fault_type"] = "load"
            elif load_estimate < 50:
                result["diagnosis"] = "UNDER_LOAD"
                result["confidence"] = 50
                result["severity"] = "Low"
                result["fault_type"] = "load"
            else:
                # 5. Normal
                result["diagnosis"] = "NORMAL_ELECTRICAL"
                result["confidence"] = 95
                result["severity"] = "Low"
                result["fault_type"] = "normal"
    
    result["details"] = {
        "voltage_unbalance": voltage_unbalance,
        "current_unbalance": current_unbalance,
        "load_estimate": load_estimate
    }
    return result

# ============================================================================
# CROSS-DOMAIN INTEGRATION LOGIC
# ============================================================================
def aggregate_cross_domain_diagnosis(mech_result, hyd_result, elec_result,
                                     shared_context, temp_data=None):
    system_result = {
        "diagnosis": "Tidak Ada Korelasi Antar Domain Terdeteksi",
        "confidence": 0,
        "severity": "Low",
        "location": "N/A",
        "domain_breakdown": {},
        "correlation_notes": [],
        "temperature_notes": []
    }
    system_result["domain_breakdown"] = {
        "mechanical": mech_result,
        "hydraulic": hyd_result,
        "electrical": elec_result
    }
    mech_fault = mech_result.get("fault_type")
    hyd_fault = hyd_result.get("fault_type")
    elec_fault = elec_result.get("fault_type")
    mech_diag = mech_result.get("diagnosis", "Normal")
    hyd_diag = hyd_result.get("diagnosis", "NORMAL_OPERATION")
    elec_diag = elec_result.get("diagnosis", "NORMAL_ELECTRICAL")
    mech_sev = mech_result.get("severity", "Low")
    hyd_sev = hyd_result.get("severity", "Low")
    elec_sev = elec_result.get("severity", "Low")
    correlation_bonus = 0
    correlated_faults = []
    # Pattern 1: Voltage → Mechanical → Hydraulic
    if (elec_fault == "voltage" and
        mech_result.get("diagnosis") in ["MISALIGNMENT", "LOOSENESS"] and
        hyd_result.get("details", {}).get("deviations", {}).get("head_dev", 0) < -5):
        correlation_bonus += 15
        correlated_faults.append("Voltage unbalance → torque pulsation → hydraulic instability")
        system_result["diagnosis"] = "Electrical-Mechanical-Hydraulic Coupled Fault"
    # Pattern 2: Cavitation → Wear → Current
    if (hyd_fault == "cavitation" and mech_fault == "wear" and
        elec_result.get("details", {}).get("current_unbalance", 0) > 5):
        correlation_bonus += 20
        correlated_faults.append("Cavitation → impeller erosion → unbalance → current fluctuation")
        system_result["diagnosis"] = "Cascading Failure: Cavitation Origin"
    # Pattern 3: High Current + Low Efficiency
    if (elec_result.get("diagnosis") == "OVER_LOAD" and
        hyd_fault == "efficiency"):
        correlation_bonus += 10
        correlated_faults.append("High electrical input + low hydraulic output → internal mechanical/hydraulic loss")
        system_result["diagnosis"] = "Internal Loss Investigation Required"
    # Temperature-based correlation
    if temp_data:
        temp_adjustment, temp_notes = calculate_temperature_confidence_adjustment(
            temp_data,
            diagnosis_consistent=(mech_fault is not None and mech_fault != "normal")
        )
        correlation_bonus += temp_adjustment
        system_result["temperature_notes"] = temp_notes
        if temp_data.get("Pump_DE") and temp_data.get("Pump_NDE"):
            if abs(temp_data["Pump_DE"] - temp_data["Pump_NDE"]) > BEARING_TEMP_LIMITS["delta_threshold"]:
                correlated_faults.append(f"Pump DE-NDE ΔT >15°C → Localized fault on DE bearing")
        if temp_data.get("Motor_DE") and temp_data.get("Pump_DE"):
            if temp_data["Motor_DE"] > temp_data["Pump_DE"] + 10:
                correlated_faults.append("Motor DE > Pump DE → Possible electrical origin")
    # Severity: Ambil yang tertinggi dari 3 domain
    severities = [mech_sev, hyd_sev, elec_sev]
    if "High" in severities:
        system_result["severity"] = "High"
    elif "Medium" in severities:
        system_result["severity"] = "Medium"
    else:
        system_result["severity"] = "Low"
    # Upgrade severity if critical temperature
    if temp_data:
        for temp in temp_data.values():
            if temp and temp > BEARING_TEMP_LIMITS["critical_min"]:
                system_result["severity"] = "High"
                correlated_faults.append("⚠️ Critical bearing temperature detected")
                break
    # Confidence: Base dari rata-rata domain + correlation bonus
    confidences = [r.get("confidence", 0) for r in [mech_result, hyd_result, elec_result]
                   if r.get("confidence", 0) > 0]
    base_confidence = np.mean(confidences) if confidences else 0
    system_result["confidence"] = min(95, int(base_confidence + correlation_bonus))
    # Correlation Notes
    system_result["correlation_notes"] = correlated_faults if correlated_faults else ["Tidak ada korelasi kuat antar domain terdeteksi"]
    return system_result

# ============================================================================
# REPORT GENERATION
# ============================================================================
def generate_unified_csv_report(machine_id, rpm, timestamp, mech_data, hyd_data,
                                elec_data, integrated_result, temp_data=None):
    lines = []
    lines.append(f"MULTI-DOMAIN PUMP DIAGNOSTIC REPORT - {machine_id.upper()}")
    lines.append(f"Generated: {timestamp}")
    lines.append(f"RPM: {rpm} | 1x RPM: {rpm/60:.2f} Hz")
    lines.append(f"Standards: ISO 10816-3/7 (Mech) | API 610 (Hyd) | IEC 60034 (Elec)")
    lines.append("")
    if temp_data:
        lines.append("=== BEARING TEMPERATURE ===")
        lines.append(f"Pump_DE: {temp_data.get('Pump_DE', 'N/A')}°C | Pump_NDE: {temp_data.get('Pump_NDE', 'N/A')}°C")
        lines.append(f"Motor_DE: {temp_data.get('Motor_DE', 'N/A')}°C | Motor_NDE: {temp_data.get('Motor_NDE', 'N/A')}°C")
        if temp_data.get('Pump_DE') and temp_data.get('Pump_NDE'):
            lines.append(f"Pump ΔT (DE-NDE): {abs(temp_data['Pump_DE'] - temp_data['Pump_NDE']):.1f}°C")
        if temp_data.get('Motor_DE') and temp_data.get('Motor_NDE'):
            lines.append(f"Motor ΔT (DE-NDE): {abs(temp_data['Motor_DE'] - temp_data['Motor_NDE']):.1f}°C")
        lines.append("")
    lines.append("=== MECHANICAL VIBRATION ===")
    if mech_data.get("points"):
        lines.append("POINT,Overall_Vel(mm/s),Band1(g),Band2(g),Band3(g),Diagnosis,Confidence,Severity")
        for point, data in mech_data["points"].items():
            lines.append(f"{point},{data['velocity']:.2f},{data['bands']['Band1']:.3f},{data['bands']['Band2']:.3f},{data['bands']['Band3']:.3f},{data['diagnosis']},{data['confidence']},{data['severity']}")
        lines.append(f"System Diagnosis: {mech_data.get('system_diagnosis', 'N/A')}")
    lines.append("")
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
    lines.append("=== ELECTRICAL CONDITION (3-Phase) ===")
    if elec_data.get("measurements"):
        e = elec_data["measurements"]
        lines.append(f"Voltage L1-L2: {e.get('v_l1l2', 0):.1f}V | L2-L3: {e.get('v_l2l3', 0):.1f}V | L3-L1: {e.get('v_l3l1', 0):.1f}V")
        lines.append(f"Current L1: {e.get('i_l1', 0):.1f}A | L2: {e.get('i_l2', 0):.1f}A | L3: {e.get('i_l3', 0):.1f}A")
        lines.append(f"Voltage Unbalance: {elec_data.get('voltage_unbalance', 0):.2f}% | Current Unbalance: {elec_data.get('current_unbalance', 0):.2f}%")
        lines.append(f"Load Estimate: {elec_data.get('load_estimate', 0):.1f}%")
        lines.append(f"Diagnosis: {elec_data.get('diagnosis', 'N/A')} | Confidence: {elec_data.get('confidence', 0)}% | Severity: {elec_data.get('severity', 'N/A')}")
    lines.append("")
    lines.append("=== INTEGRATED DIAGNOSIS ===")
    lines.append(f"Overall Diagnosis: {integrated_result.get('diagnosis', 'N/A')}")
    lines.append(f"Overall Confidence: {integrated_result.get('confidence', 0)}%")
    lines.append(f"Overall Severity: {integrated_result.get('severity', 'N/A')}")
    lines.append(f"Correlation Notes: {'; '.join(integrated_result.get('correlation_notes', []))}")
    if integrated_result.get("temperature_notes"):
        lines.append(f"Temperature Notes: {'; '.join(integrated_result['temperature_notes'])}")
    lines.append("")
    return "\n".join(lines)

# ============================================================================
# STREAMLIT UI - MAIN APPLICATION
# ============================================================================
def main():
    st.set_page_config(
        page_title="Pump Diagnostic Expert System",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    if "shared_context" not in st.session_state:
        st.session_state.shared_context = {
            "machine_id": "P-101",
            "rpm": 2950,
            "service_criticality": "Essential (Utility)",
            "fluid_type": "Diesel / Solar",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    st.markdown("""
    <div style="background-color:#1E3A5F; padding:15px; border-radius:8px; margin-bottom:20px">
    <h2 style="color:white; margin:0">🔧💧⚡ Pump Diagnostic Expert System</h2>
    <p style="color:#E0E0E0; margin:5px 0 0 0">
    Integrated Mechanical • Hydraulic • Electrical Analysis | Pertamina Patra Niaga
    </p>
    </div>
    """, unsafe_allow_html=True)
    with st.sidebar:
        st.subheader("📍 Shared Context")
        machine_id = st.text_input("Machine ID / Tag", value=st.session_state.shared_context["machine_id"])
        rpm = st.number_input("Operating RPM", min_value=600, max_value=3600,
                              value=st.session_state.shared_context["rpm"], step=10)
        service_type = st.selectbox("Service Criticality",
                                    ["Critical (Process)", "Essential (Utility)", "Standby"],
                                    index=["Critical (Process)", "Essential (Utility)", "Standby"].index(
                                        st.session_state.shared_context["service_criticality"]))
        fluid_type = st.selectbox("Fluid Type (BBM)",
                                  list(FLUID_PROPERTIES.keys()),
                                  index=list(FLUID_PROPERTIES.keys()).index(
                                      st.session_state.shared_context["fluid_type"]))
        st.session_state.shared_context.update({
            "machine_id": machine_id,
            "rpm": rpm,
            "service_criticality": service_type,
            "fluid_type": fluid_type
        })
        fluid_props = FLUID_PROPERTIES[fluid_type]
        st.info(f"""
        **Fluid Properties ({fluid_type}):**
        • SG: {fluid_props['sg']}
        • Vapor Pressure @38°C: {fluid_props['vapor_pressure_kpa_38C']} kPa
        • Risk Level: {fluid_props['risk_level']}
        """)
        st.divider()
        st.subheader("🧭 Navigasi Cepat")
        st.markdown("""
        <div style="background-color:#f0f2f6; padding:10px; border-radius:5px; font-size:0.9em">
        <strong>💡 Gunakan tab di atas untuk:</strong><br><br>
        🔧 <strong>Mechanical</strong>: Vibration analysis (12 points)<br>
        💧 <strong>Hydraulic</strong>: Performance troubleshooting (single-point)<br>
        ⚡ <strong>Electrical</strong>: 3-phase condition monitoring<br>
        🔗 <strong>Integrated</strong>: Cross-domain correlation + Temperature
        </div>
        """, unsafe_allow_html=True)
        st.divider()
        st.caption("📊 Status Analisis:")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            mech_done = "✅" if "mech_result" in st.session_state else "⏳"
            st.write(f"{mech_done} Mechanical")
        with col_s2:
            hyd_done = "✅" if "hyd_result" in st.session_state else "⏳"
            st.write(f"{hyd_done} Hydraulic")
        col_s3, col_s4 = st.columns(2)
        with col_s3:
            elec_done = "✅" if "elec_result" in st.session_state else "⏳"
            st.write(f"{elec_done} Electrical")
        with col_s4:
            int_done = "✅" if "integrated_result" in st.session_state else "⏳"
            st.write(f"{int_done} Integrated")
    tab_mech, tab_hyd, tab_elec, tab_integrated = st.tabs([
        "🔧 Mechanical", "💧 Hydraulic", "⚡ Electrical", "🔗 Integrated Summary"
    ])
    # ========================================================================
    # TAB 1: MECHANICAL VIBRATION ANALYSIS
    # ========================================================================
    with tab_mech:
        st.header("🔧 Mechanical Vibration Analysis")
        st.caption("ISO 10816-3/7 | Centrifugal Pump + Electric Motor | Fixed Speed")
        col1, col2 = st.columns([2, 1])
        with col1:
            fft_mode = st.radio(
                "Mode Analisis FFT",
                options=["🚀 Efisien (Hanya Titik Bermasalah)", "🔍 Lengkap (Semua 12 Titik)"],
                index=0 if service_type != "Critical (Process)" else 1,
                key="mech_fft_mode"
            )
        with col2:
            rpm_hz = rpm / 60
            st.info(f"""
            **Konfigurasi:**
            • 1×RPM = {rpm_hz:.1f} Hz
            • Zone B Limit = 4.5 mm/s
            • Mode: {fft_mode}
            """)
        st.subheader("🌡️ Bearing Temperature (4 Points)")
        st.caption("API 610/API 670: Normal <70°C | Elevated 70-85°C | Warning 85-95°C | Critical >95°C")
        temp_cols = st.columns(4)
        temp_data = {}
        with temp_cols[0]:
            pump_de_temp = st.number_input("Pump DE (°C)", min_value=0, max_value=150,
                                           value=65, step=1, key="temp_pump_de")
            temp_data["Pump_DE"] = pump_de_temp
            if pump_de_temp > BEARING_TEMP_LIMITS["warning_min"]:
                st.error(f"🔴 {pump_de_temp}°C - Warning")
            elif pump_de_temp > BEARING_TEMP_LIMITS["elevated_min"]:
                st.warning(f"🟡 {pump_de_temp}°C - Elevated")
            else:
                st.success(f"🟢 {pump_de_temp}°C - Normal")
        with temp_cols[1]:
            pump_nde_temp = st.number_input("Pump NDE (°C)", min_value=0, max_value=150,
                                            value=63, step=1, key="temp_pump_nde")
            temp_data["Pump_NDE"] = pump_nde_temp
            if pump_nde_temp > BEARING_TEMP_LIMITS["warning_min"]:
                st.error(f"🔴 {pump_nde_temp}°C - Warning")
            elif pump_nde_temp > BEARING_TEMP_LIMITS["elevated_min"]:
                st.warning(f"🟡 {pump_nde_temp}°C - Elevated")
            else:
                st.success(f"🟢 {pump_nde_temp}°C - Normal")
        with temp_cols[2]:
            motor_de_temp = st.number_input("Motor DE (°C)", min_value=0, max_value=150,
                                            value=68, step=1, key="temp_motor_de")
            temp_data["Motor_DE"] = motor_de_temp
            if motor_de_temp > BEARING_TEMP_LIMITS["warning_min"]:
                st.error(f"🔴 {motor_de_temp}°C - Warning")
            elif motor_de_temp > BEARING_TEMP_LIMITS["elevated_min"]:
                st.warning(f"🟡 {motor_de_temp}°C - Elevated")
            else:
                st.success(f"🟢 {motor_de_temp}°C - Normal")
        with temp_cols[3]:
            motor_nde_temp = st.number_input("Motor NDE (°C)", min_value=0, max_value=150,
                                             value=66, step=1, key="temp_motor_nde")
            temp_data["Motor_NDE"] = motor_nde_temp
            if motor_nde_temp > BEARING_TEMP_LIMITS["warning_min"]:
                st.error(f"🔴 {motor_nde_temp}°C - Warning")
            elif motor_nde_temp > BEARING_TEMP_LIMITS["elevated_min"]:
                st.warning(f"🟡 {motor_nde_temp}°C - Elevated")
            else:
                st.success(f"🟢 {motor_nde_temp}°C - Normal")
        if pump_de_temp and pump_nde_temp:
            delta_pump = abs(pump_de_temp - pump_nde_temp)
            if delta_pump > BEARING_TEMP_LIMITS["delta_threshold"]:
                st.warning(f"⚠️ Pump ΔT (DE-NDE): {delta_pump}°C > 15°C threshold")
        if motor_de_temp and motor_nde_temp:
            delta_motor = abs(motor_de_temp - motor_nde_temp)
            if delta_motor > BEARING_TEMP_LIMITS["delta_threshold"]:
                st.warning(f"⚠️ Motor ΔT (DE-NDE): {delta_motor}°C > 15°C threshold")
        st.divider()
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
                    if overall > ISO_LIMITS_VELOCITY["Zone B (Acceptable)"]:
                        st.warning(f"⚠️ >4.5 mm/s")
                    if b3 > 2*ACCEL_BASELINE["Band3 (5-16kHz)"]:
                        st.error(f"🔴 Band 3 tinggi")
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
        if st.button("🔍 Jalankan Mechanical Analysis", type="primary", key="run_mech"):
            with st.spinner("Menganalisis data vibration..."):
                point_results = []
                for point in points:
                    peaks = fft_inputs.get(point, [(rpm_hz,0.1),(2*rpm_hz,0.05),(3*rpm_hz,0.02)])
                    bands = bands_inputs[point]
                    has_fft = point in fft_inputs
                    bearing_temp = None
                    if "Pump" in point and "DE" in point:
                        bearing_temp = temp_data.get("Pump_DE")
                    elif "Pump" in point and "NDE" in point:
                        bearing_temp = temp_data.get("Pump_NDE")
                    elif "Motor" in point and "DE" in point:
                        bearing_temp = temp_data.get("Motor_DE")
                    elif "Motor" in point and "NDE" in point:
                        bearing_temp = temp_data.get("Motor_NDE")
                    result = diagnose_single_point_mechanical(peaks, bands, rpm_hz, point,
                                                              input_data[point], has_fft, bearing_temp)
                    result["point"] = point
                    result["location_hint"] = f"{'Pump' if 'Pump' in point else 'Motor'} {point.split()[1]}"
                    point_results.append(result)
                mech_system = {
                    "diagnosis": "Normal",
                    "confidence": 99,
                    "severity": "Low",
                    "fault_type": "normal",
                    "domain": "mechanical"
                }
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
                st.session_state.mech_result = mech_system
                st.session_state.mech_data = {
                    "points": {p: {"velocity": input_data[p], "bands": bands_inputs[p],
                                   "diagnosis": next((r["diagnosis"] for r in point_results if r["point"]==p), "Normal"),
                                   "confidence": next((r["confidence"] for r in point_results if r["point"]==p), 0),
                                   "severity": next((r["severity"] for r in point_results if r["point"]==p), "Low")}
                           for p in points},
                    "system_diagnosis": mech_system["diagnosis"]
                }
                st.session_state.temp_data = temp_data
                st.success(f"✅ Mechanical Analysis Complete: {mech_system['diagnosis']} ({mech_system['confidence']}%)")
        if "mech_result" in st.session_state:
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
    # TAB 2: HYDRAULIC TROUBLESHOOTING
    # ========================================================================
    with tab_hyd:
        st.header("💧 Hydraulic Troubleshooting")
        st.caption("Single-Point Steady-State Measurement | BBM Fluid Types")
        steady_verified = True
        st.subheader("📊 Data Primer Hidrolik")
        col1, col2, col3 = st.columns(3)
        with col1:
            suction_pressure = st.number_input("Suction Pressure", min_value=0.0, value=1.2, step=0.1, format="%.1f", key="suction_p")
            discharge_pressure = st.number_input("Discharge Pressure", min_value=0.0, value=8.5, step=0.1, format="%.1f", key="discharge_p")
            delta_p = discharge_pressure - suction_pressure
            st.metric("Differential Pressure", f"{delta_p:.1f} bar")
        with col2:
            flow_rate = st.number_input("Flow Rate", min_value=0.0, value=100.0, step=1.0, format="%.1f", key="flow_rate")
            motor_power = st.number_input("Motor Power Input", min_value=0.0, value=45.0, step=0.5, format="%.1f", key="motor_power")
            fluid_temp = st.number_input("Fluid Temperature", min_value=0, max_value=100, value=40, step=1, key="fluid_temp")
        with col3:
            fluid_props = FLUID_PROPERTIES[fluid_type]
            sg = st.number_input("Specific Gravity (SG)", min_value=0.5, max_value=1.5, value=fluid_props["sg"], step=0.01, key="sg_input")
            st.caption(f"Default: {fluid_props['sg']} untuk {fluid_type}")
            if delta_p > 0 and sg > 0:
                head_calc = delta_p * 10.2 / sg
                st.metric("Calculated Head", f"{head_calc:.1f} m")
        with st.expander("📋 Data Referensi Desain (OEM)"):
            col1, col2 = st.columns(2)
            with col1:
                rated_flow = st.number_input("Rated Flow (Q_design)", min_value=0.0, value=120.0, step=1.0, key="rated_flow")
                rated_head = st.number_input("Rated Head (H_design)", min_value=0.0, value=85.0, step=0.5, key="rated_head")
            with col2:
                bep_efficiency = st.number_input("BEP Efficiency", min_value=0, max_value=100, value=78, step=1, key="bep_eff")
                npsh_required = st.number_input("NPSH Required @ Q_aktual", min_value=0.0, value=4.2, step=0.1, key="npshr")
        st.subheader("🔍 Observasi Sekunder")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Kondisi Visual & Fluida")
            leakage = st.radio("Kebocoran Seal", ["Tidak ada", "Minor (≤2 tetes/menit)", "Mayor"], index=0, key="leakage")
            fluid_condition = st.radio("Kondisi Fluida", ["Jernih", "Agak keruh", "Keruh/partikel", "Berbusa"], index=0, key="fluid_cond")
        with col2:
            st.caption("Kebisingan & Getaran")
            noise_type = st.radio("Jenis Noise", ["Normal", "Whining", "Grinding", "Crackling"], index=0, key="noise_type")
            vibration_qual = st.radio("Getaran (kualitatif)", ["Tidak terasa", "Halus", "Jelas", "Kuat"], index=0, key="vib_qual")
        with st.expander("📋 Konteks Operasional"):
            valve_position = st.slider("Valve Discharge Position", 0, 100, 100, format="%d%%", key="valve_pos")
            recent_changes = st.text_area("Recent Changes (7 hari terakhir)", placeholder="Contoh: ganti fluida, maintenance, dll", key="recent_changes")
            operator_complaint = st.text_area("Keluhan Operator (verbatim)", placeholder="Tulis persis seperti disampaikan", key="operator_complaint")
        analyze_hyd_disabled = not steady_verified or suction_pressure >= discharge_pressure
        if st.button("💧 Generate Hydraulic Diagnosis", type="primary", key="run_hyd", disabled=analyze_hyd_disabled):
            with st.spinner("Menganalisis performa hidrolik..."):
                hyd_calc = calculate_hydraulic_parameters(
                    suction_pressure, discharge_pressure, flow_rate, motor_power, sg, fluid_temp
                )
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
                hyd_result = diagnose_hydraulic_single_point(
                    hyd_calc, design_params, fluid_props, observations, context
                )
                st.session_state.hyd_result = hyd_result
                st.session_state.hyd_data = {
                    "measurements": {
                        "suction_pressure": suction_pressure,
                        "discharge_pressure": discharge_pressure,
                        "flow_rate": flow_rate,
                        "motor_power": motor_power,
                        "power_factor": 0.85
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
        if "hyd_result" in st.session_state:
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
            npsh_margin = result["details"].get("npsh_margin_m", 999)
            if npsh_margin < 0.5:
                st.warning(f"""
                ⚠️ **NPSH Margin Critical**: {npsh_margin:.2f} m
                • Risk of cavitation untuk {fluid_type} (vapor pressure tinggi)
                • Rekomendasi: tingkatkan suction pressure atau turunkan fluid temperature
                """)
    # ========================================================================
    # TAB 3: ELECTRICAL CONDITION ANALYSIS (SIMPLIFIED)
    # ========================================================================
    with tab_elec:
        st.header("⚡ Electrical Condition Analysis")
        st.caption("3-Phase Voltage/Current | Unbalance Detection | Motor Load Estimation")
        # ✅ SIMPLIFIED: Only 2 nameplate parameters needed
        with st.expander("⚙️ Motor Nameplate (Minimal)", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                rated_voltage = st.number_input("Rated Voltage (V)", min_value=200, max_value=690, value=400, step=10, key="rated_v",
                                                help="Dari nameplate motor - acuan toleransi ±10%")
            with col2:
                fla = st.number_input("Full Load Amps - FLA (A)", min_value=10, max_value=500, value=85, step=5, key="rated_i",
                                      help="Dari nameplate motor - acuan load capacity")
            st.caption("💡 Hanya 2 parameter nameplate yang diperlukan untuk voltage/current analysis")
        st.subheader("📊 Pengukuran 3-Phase")
        # ✅ SIMPLIFIED: 2 columns instead of 3 (removed Power & Frequency column)
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Voltage (Line-to-Line)")
            v_l1l2 = st.number_input("L1-L2 (V)", min_value=0.0, value=400.0, step=1.0, key="v_l1l2")
            v_l2l3 = st.number_input("L2-L3 (V)", min_value=0.0, value=402.0, step=1.0, key="v_l2l3")
            v_l3l1 = st.number_input("L3-L1 (V)", min_value=0.0, value=398.0, step=1.0, key="v_l3l1")
        with col2:
            st.caption("Current (Per Phase)")
            i_l1 = st.number_input("L1 (A)", min_value=0.0, value=82.0, step=0.5, key="i_l1")
            i_l2 = st.number_input("L2 (A)", min_value=0.0, value=84.0, step=0.5, key="i_l2")
            i_l3 = st.number_input("L3 (A)", min_value=0.0, value=83.0, step=0.5, key="i_l3")
        # ✅ SIMPLIFIED: Real-time calculation display (3 columns instead of 4)
        with st.expander("📈 Perhitungan Real-Time", expanded=True):
            elec_calc = calculate_electrical_parameters(
                v_l1l2, v_l2l3, v_l3l1, i_l1, i_l2, i_l3,
                rated_voltage, fla
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Voltage Unbalance", f"{elec_calc['voltage_unbalance_percent']:.2f}%",
                          delta="⚠️ Warning" if elec_calc['voltage_unbalance_percent'] > ELECTRICAL_LIMITS["voltage_unbalance_warning"] else "✅ OK")
            with col2:
                st.metric("Current Unbalance", f"{elec_calc['current_unbalance_percent']:.2f}%",
                          delta="⚠️ Warning" if elec_calc['current_unbalance_percent'] > ELECTRICAL_LIMITS["current_unbalance_warning"] else "✅ OK")
            with col3:
                st.metric("Load Estimate", f"{elec_calc['load_estimate_percent']:.1f}%")
            if not elec_calc["voltage_within_tolerance"]:
                st.warning(f"⚠️ Voltage {elec_calc['v_avg']:.1f}V outside tolerance ({ELECTRICAL_LIMITS['voltage_tolerance_low']}-{ELECTRICAL_LIMITS['voltage_tolerance_high']}% of {rated_voltage}V)")
        # ✅ SIMPLIFIED: Removed visual/sensory observations section
        if st.button("⚡ Generate Electrical Diagnosis", type="primary", key="run_elec"):
            with st.spinner("Menganalisis kondisi electrical..."):
                # ✅ SIMPLIFIED: Only rated_voltage and fla needed
                motor_specs = {
                    "rated_voltage": rated_voltage,
                    "fla": fla
                }
                # ✅ SIMPLIFIED: No observations parameter
                elec_result = diagnose_electrical_condition(elec_calc, motor_specs)
                st.session_state.elec_result = elec_result
                # ✅ SIMPLIFIED: No power_factor in measurements
                st.session_state.elec_data = {
                    "measurements": {
                        "v_l1l2": v_l1l2, "v_l2l3": v_l2l3, "v_l3l1": v_l3l1,
                        "i_l1": i_l1, "i_l2": i_l2, "i_l3": i_l3
                    },
                    "voltage_unbalance": elec_calc["voltage_unbalance_percent"],
                    "current_unbalance": elec_calc["current_unbalance_percent"],
                    "load_estimate": elec_calc["load_estimate_percent"],
                    "diagnosis": elec_result["diagnosis"],
                    "confidence": elec_result["confidence"],
                    "severity": elec_result["severity"]
                }
                st.success(f"✅ Electrical Analysis Complete: {elec_result['diagnosis']} ({elec_result['confidence']}%)")
        if "elec_result" in st.session_state:
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
    # TAB 4: INTEGRATED SUMMARY
    # ========================================================================
    with tab_integrated:
        st.header("🔗 Integrated Diagnostic Summary")
        st.caption("Cross-Domain Correlation | Temperature Analysis")
        analyses_complete = all([
            "mech_result" in st.session_state,
            "hyd_result" in st.session_state,
            "elec_result" in st.session_state
        ])
        if not analyses_complete:
            st.info("""
            💡 **Langkah Selanjutnya:**
            1. Jalankan analisis di tab **🔧 Mechanical**
            2. Jalankan analisis di tab **💧 Hydraulic**
            3. Jalankan analisis di tab **⚡ Electrical**
            4. Kembali ke tab ini untuk integrated diagnosis
            """)
            col1, col2, col3 = st.columns(3)
            with col1:
                status_mech = "✅" if "mech_result" in st.session_state else "⏳"
                st.metric("Mechanical", status_mech)
            with col2:
                status_hyd = "✅" if "hyd_result" in st.session_state else "⏳"
                st.metric("Hydraulic", status_hyd)
            with col3:
                status_elec = "✅" if "elec_result" in st.session_state else "⏳"
                st.metric("Electrical", status_elec)
        else:
            with st.spinner("Mengintegrasikan hasil tiga domain..."):
                temp_data = st.session_state.get("temp_data", None)
                integrated_result = aggregate_cross_domain_diagnosis(
                    st.session_state.mech_result,
                    st.session_state.hyd_result,
                    st.session_state.elec_result,
                    st.session_state.shared_context,
                    temp_data
                )
                st.session_state.integrated_result = integrated_result
                st.subheader("📊 Overall Assessment")
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"""
                    <div style="background-color:#f0f2f6; padding:15px; border-radius:8px; border-left:5px solid #1E3A5F">
                    <h4 style="margin:0 0 10px 0; color:#1E3A5F">🔗 Integrated Diagnosis</h4>
                    <p style="margin:0; font-size:1.1em; font-weight:600; color:#2c3e50; word-wrap:break-word; white-space:normal;">
                    {integrated_result["diagnosis"]}
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    severity_config = {
                        "Low": ("🟢", "#27ae60"),
                        "Medium": ("🟠", "#f39c12"),
                        "High": ("🔴", "#c0392b")
                    }
                    sev_icon, sev_color = severity_config.get(integrated_result["severity"], ("⚪", "#95a5a6"))
                    st.markdown(f"""
                    <div style="background-color:#f0f2f6; padding:15px; border-radius:8px; border-left:5px solid {sev_color}">
                    <h4 style="margin:0 0 10px 0; color:#1E3A5F">⚠️ Overall Severity</h4>
                    <p style="margin:0; font-size:1.5em; font-weight:700; color:{sev_color};">
                    {sev_icon} {integrated_result["severity"]}
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                col3, col4, col5 = st.columns(3)
                with col3:
                    st.metric("Confidence", f"{integrated_result['confidence']}%")
                with col4:
                    correlation_text = "Detected" if integrated_result['correlation_notes'] and integrated_result['correlation_notes'][0] != "Tidak ada korelasi kuat antar domain terdeteksi" else "None"
                    st.metric("Cross-Domain Correlation", correlation_text)
                with col5:
                    has_valid_temp = (
                        temp_data and
                        any(v and v > 0 for v in temp_data.values())
                    )
                    temp_status = "Available" if has_valid_temp else "N/A"
                    st.metric("Temperature Data", temp_status)
                if temp_data:
                    with st.expander("🌡️ Temperature Analysis Summary", expanded=True):
                        temp_cols = st.columns(4)
                        with temp_cols[0]:
                            temp_status, temp_color, _ = get_temperature_status(temp_data.get("Pump_DE", 0))
                            st.metric("Pump DE", f"{temp_data.get('Pump_DE', 0)}°C", temp_status)
                        with temp_cols[1]:
                            temp_status, temp_color, _ = get_temperature_status(temp_data.get("Pump_NDE", 0))
                            st.metric("Pump NDE", f"{temp_data.get('Pump_NDE', 0)}°C", temp_status)
                        with temp_cols[2]:
                            temp_status, temp_color, _ = get_temperature_status(temp_data.get("Motor_DE", 0))
                            st.metric("Motor DE", f"{temp_data.get('Motor_DE', 0)}°C", temp_status)
                        with temp_cols[3]:
                            temp_status, temp_color, _ = get_temperature_status(temp_data.get("Motor_NDE", 0))
                            st.metric("Motor NDE", f"{temp_data.get('Motor_NDE', 0)}°C", temp_status)
                        if integrated_result.get("temperature_notes"):
                            st.info("**Temperature Insights:**\n" + "\n".join(integrated_result["temperature_notes"]))
                with st.expander("🗺️ Fault Propagation Map", expanded=True):
                    st.markdown("**📌 Cross-Domain Correlation Notes:**")
                    for note in integrated_result["correlation_notes"]:
                        st.markdown(f"""
                        <div style="background-color:#fff3cd; padding:10px; border-radius:5px; margin:5px 0; border-left:4px solid #ffc107;">
                        {note}
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("**🔗 Propagation Path:**")
                    st.markdown(f"""
                    <div style="background-color:#f8f9fa; padding:15px; border-radius:8px; font-family:monospace; font-size:0.95em;">
                    <div style="color:#e74c3c; font-weight:bold;">⚡ Electrical: {st.session_state.elec_result['diagnosis']}</div>
                    <div style="color:#7f8c8d; margin:5px 0;">│</div>
                    <div style="color:#7f8c8d; margin:5px 0;">▼</div>
                    <div style="color:#7f8c8d; margin:5px 0;">│</div>
                    <div style="color:#3498db; font-weight:bold;">🔧 Mechanical: {st.session_state.mech_result['diagnosis']}</div>
                    <div style="color:#7f8c8d; margin:5px 0;">│</div>
                    <div style="color:#7f8c8d; margin:5px 0;">▼</div>
                    <div style="color:#7f8c8d; margin:5px 0;">│</div>
                    <div style="color:#27ae60; font-weight:bold;">💧 Hydraulic: {st.session_state.hyd_result['diagnosis']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if integrated_result.get("temperature_notes"):
                        st.markdown("**🌡️ Temperature Insights:**")
                        for temp_note in integrated_result["temperature_notes"]:
                            st.markdown(f"""
                            <div style="background-color:#d5f5e3; padding:10px; border-radius:5px; margin:5px 0; border-left:4px solid #27ae60;">
                            {temp_note}
                            </div>
                            """, unsafe_allow_html=True)
                st.divider()
                st.subheader("📥 Export Report")
                if st.button("📊 Generate Unified CSV Report", type="primary"):
                    csv_report = generate_unified_csv_report(
                        machine_id, rpm,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        st.session_state.get("mech_data", {}),
                        st.session_state.get("hyd_data", {}),
                        st.session_state.get("elec_data", {}),
                        integrated_result,
                        temp_data
                    )
                    st.download_button(
                        label="📥 Download CSV Report",
                        data=csv_report,
                        file_name=f"PUMP_DIAG_{machine_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    st.success("✅ Report generated successfully!")
                st.divider()
                st.caption("""
                **Standar Acuan**: ISO 10816-3/7 | ISO 13373-1 | API 610 | **IEC 60034** | API 670
                **Algoritma**: Hybrid rule-based dengan cross-domain correlation + confidence scoring + temperature analysis
                ⚠️ Decision Support System - Verifikasi oleh personnel kompeten untuk keputusan kritis
                🏭 Pertamina Patra Niaga - Asset Integrity Management
                """)

if __name__ == "__main__":
    main()
