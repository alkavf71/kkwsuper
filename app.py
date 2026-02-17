
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter

# ===== KONFIGURASI AWAL - MEKANIKAL FOKUS =====
ISO_LIMITS_VELOCITY = {
    "Zone A (Good)": 2.8,
    "Zone B (Acceptable)": 4.5,
    "Zone C (Unacceptable)": 7.1,
    "Zone D (Danger)": 11.0
}

# Threshold acceleration untuk bearing monitoring (high-frequency)
ACCEL_BASELINE = {
    "Band1 (0.5-1.5kHz)": 0.3,  # g - developed fault
    "Band2 (1.5-5kHz)": 0.2,    # g - early fault  
    "Band3 (5-16kHz)": 0.15     # g - incipient fault / lubrication
}

# ===== FUNGSI REKOMENDASI MEKANIKAL =====
def get_mechanical_recommendation(diagnosis: str, location: str, severity: str = "Medium") -> str:
    """Rekomendasi spesifik untuk fault mekanikal pump-motor"""
    
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


# ===== GENERATE CSV REPORT (12 TITIK + BANDS) =====
def generate_csv_report_mechanical(machine_id, rpm, all_points, input_data, fft_inputs, bands_inputs):
    """Generate CSV report sesuai ISO 13374 dengan frequency bands"""
    lines = []
    lines.append(f"MECHANICAL VIBRATION REPORT - {machine_id.upper()}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"RPM: {rpm} | 1x RPM: {rpm/60:.2f} Hz | Type: Centrifugal Pump + Electric Motor (Fixed Speed)")
    lines.append(f"Standard: ISO 10816-3/7 (Velocity) | Bearing Monitoring: Acceleration Bands")
    lines.append("")
    lines.append("POINT,Overall_Vel(mm/s),Band1(g),Band2(g),Band3(g),Peak1_Hz,Peak1_Amp,Peak2_Hz,Peak2_Amp,Peak3_Hz,Peak3_Amp,Diagnosis,Confidence,Severity")
    
    rpm_hz = rpm / 60
    for point in all_points:
        peaks = fft_inputs.get(point, [(0,0),(0,0),(0,0)])
        bands = bands_inputs.get(point, {"Band1":0, "Band2":0, "Band3":0})
        
        # Run diagnosis untuk titik ini
        result = diagnose_single_point(peaks, bands, rpm_hz, point, input_data[point])
        
        lines.append(
            f"{point},"
            f"{input_data[point]:.2f},"
            f"{bands['Band1']:.3f},{bands['Band2']:.3f},{bands['Band3']:.3f},"
            f"{peaks[0][0]:.1f},{peaks[0][1]:.2f},"
            f"{peaks[1][0]:.1f},{peaks[1][1]:.2f},"
            f"{peaks[2][0]:.1f},{peaks[2][1]:.2f},"
            f"{result['diagnosis']},{result['confidence']},{result['severity']}"
        )
    
    # Summary section
    lines.append("")
    lines.append("SYSTEM SUMMARY:")
    flagged = [p for p in all_points if input_data[p] > ISO_LIMITS_VELOCITY["Zone B (Acceptable)"]]
    lines.append(f"Points above Zone B: {len(flagged)}/12")
    bearing_flags = [p for p in all_points if bands_inputs.get(p,{}).get("Band3",0) > ACCEL_BASELINE["Band3 (5-16kHz)"]*2]
    lines.append(f"Points with high Band3 (bearing alert): {len(bearing_flags)}/12")
    
    return "\n".join(lines)


# ===== DIAGNOSA SINGLE POINT (MEKANIKAL RULES) =====
def diagnose_single_point(peaks, bands, rpm_hz, point, overall_vel):
    """Diagnosa per titik berdasarkan mechanical fault rules"""
    
    result = {
        "diagnosis": "Normal",
        "confidence": 0,
        "severity": "Low",
        "fault_type": None  # 'low_freq' atau 'high_freq'
    }
    
    # --- LOW FREQUENCY FAULTS (Velocity-based, FFT peaks) ---
    # 1. UNBALANCE: 1x RPM dominant, radial direction, H/V > A
    if "Axial" not in point:  # Hanya radial untuk unbalance
        peak_1x = None
        for freq, amp in peaks:
            if abs(freq - rpm_hz) < 0.05 * rpm_hz:  # Tolerance 5%
                peak_1x = amp
                break
        
        if peak_1x and peak_1x > 0.7 * sum(p[1] for p in peaks):  # 1x > 70% total energy
            # Cek ratio H/V vs Axial jika ada data axial dari bearing sama
            result["diagnosis"] = "UNBALANCE"
            result["confidence"] = min(95, 70 + int((peak_1x / 4.5) * 10))  # Normalize terhadap alarm limit
            result["severity"] = "High" if overall_vel > ISO_LIMITS_VELOCITY["Zone C (Unacceptable)"] else "Medium" if overall_vel > ISO_LIMITS_VELOCITY["Zone B (Acceptable)"] else "Low"
            result["fault_type"] = "low_freq"
            return result
    
    # 2. MISALIGNMENT: 1x + 2x RPM kuat, axial direction prominent
    if "Axial" in point:
        has_1x = any(abs(p[0] - rpm_hz) < 0.05 * rpm_hz for p in peaks)
        has_2x = any(abs(p[0] - 2*rpm_hz) < 0.05 * rpm_hz for p in peaks)
        
        if has_1x and has_2x:
            # Cek kekuatan 2x relatif terhadap 1x
            amp_1x = next((p[1] for p in peaks if abs(p[0]-rpm_hz)<0.05*rpm_hz), 0)
            amp_2x = next((p[1] for p in peaks if abs(p[0]-2*rpm_hz)<0.05*rpm_hz), 0)
            
            if amp_2x > 0.5 * amp_1x:  # 2x signifikan
                result["diagnosis"] = "MISALIGNMENT"
                result["confidence"] = min(95, 65 + int((amp_2x/amp_1x) * 20) if amp_1x > 0 else 65)
                result["severity"] = "High" if overall_vel > ISO_LIMITS_VELOCITY["Zone C (Unacceptable)"] else "Medium"
                result["fault_type"] = "low_freq"
                return result
    
    # 3. LOOSENESS: Harmonik 1x-3x dengan decay lambat, sering di Vertical
    if "Vertical" in point:
        harmonics_ok = True
        for i in range(3):
            target_freq = (i+1) * rpm_hz
            found = any(abs(p[0] - target_freq) < 0.05 * rpm_hz for p in peaks)
            if not found:
                harmonics_ok = False
                break
        
        if harmonics_ok:
            # Cek decay pattern: amp_2x > 0.5*amp_1x DAN amp_3x > 0.3*amp_1x
            amps = []
            for i in range(3):
                target_freq = (i+1) * rpm_hz
                amp = next((p[1] for p in peaks if abs(p[0]-target_freq)<0.05*rpm_hz), 0)
                amps.append(amp)
            
            if amps[0] > 0 and amps[1] > 0.5*amps[0] and amps[2] > 0.3*amps[0]:
                result["diagnosis"] = "LOOSENESS"
                result["confidence"] = min(90, 60 + int((amps[1]/amps[0] + amps[2]/amps[0]) * 15))
                result["severity"] = "High" if overall_vel > ISO_LIMITS_VELOCITY["Zone C (Unacceptable)"] else "Medium"
                result["fault_type"] = "low_freq"
                return result
    
    # --- HIGH FREQUENCY FAULTS (Acceleration-based, Frequency Bands) ---
    # Bearing fault detection berdasarkan pola bands
    b1, b2, b3 = bands["Band1"], bands["Band2"], bands["Band3"]
    base1, base2, base3 = ACCEL_BASELINE["Band1 (0.5-1.5kHz)"], ACCEL_BASELINE["Band2 (1.5-5kHz)"], ACCEL_BASELINE["Band3 (5-16kHz)"]
    
    # Pattern 1: Incipient / Lubrication Issue (Band3 tinggi saja)
    if b3 > 2.0 * base3 and b2 < 1.5 * base2 and b1 < 1.5 * base1:
        result["diagnosis"] = "BEARING_EARLY"
        result["confidence"] = min(85, 60 + int((b3/base3 - 2) * 10))
        result["severity"] = "Medium" if b3 > 3*base3 else "Low"
        result["fault_type"] = "high_freq"
        return result
    
    # Pattern 2: Developing Fault (Band2 dominan, Band3 masih tinggi)
    if b2 > 2.0 * base2 and b3 > 1.5 * base3 and b1 < 1.5 * base1:
        result["diagnosis"] = "BEARING_DEVELOPED"
        result["confidence"] = min(90, 70 + int((b2/base2 - 2) * 8))
        result["severity"] = "High" if b2 > 3*base2 else "Medium"
        result["fault_type"] = "high_freq"
        return result
    
    # Pattern 3: Severe Fault (Band1 dominan - energi sudah turun ke low freq)
    if b1 > 2.5 * base1 and b2 > 1.5 * base2:
        result["diagnosis"] = "BEARING_SEVERE"
        result["confidence"] = min(95, 80 + int((b1/base1 - 2.5) * 6))
        result["severity"] = "High"
        result["fault_type"] = "high_freq"
        return result
    
    # Default: Normal atau tidak cukup data
    if overall_vel <= ISO_LIMITS_VELOCITY["Zone B (Acceptable)"]:
        result["diagnosis"] = "Normal"
        result["confidence"] = 99
        result["severity"] = "Low"
    else:
        result["diagnosis"] = "Tidak Terdiagnosa"
        result["confidence"] = 30
        result["severity"] = "Medium"
    
    return result


# ===== SISTEM AGGREGATION (MULTI-POINT VOTING) =====
def aggregate_system_diagnosis(point_results, input_data, points):
    """Agregasi hasil 12 titik menjadi kesimpulan sistem dengan weighted voting"""
    
    # Pisahkan low_freq dan high_freq faults
    low_freq_faults = [r for r in point_results if r["fault_type"] == "low_freq"]
    high_freq_faults = [r for r in point_results if r["fault_type"] == "high_freq"]
    
    system_result = {
        "diagnosis": "Normal",
        "confidence": 0,
        "severity": "Low",
        "location": "N/A",
        "details": []
    }
    
    # --- LOW FREQUENCY AGGREGATION (Unbalance/Misalignment/Looseness) ---
    if low_freq_faults:
        fault_counter = Counter([r["diagnosis"] for r in low_freq_faults])
        dominant = fault_counter.most_common(1)[0]
        
        if dominant[1] >= 3:  # Minimal 3 titik support untuk keputusan
            # Ambil entry dengan confidence tertinggi sebagai representasi
            dominant_entries = [r for r in low_freq_faults if r["diagnosis"] == dominant[0]]
            best = max(dominant_entries, key=lambda x: x["confidence"])
            
            system_result["diagnosis"] = dominant[0]
            system_result["confidence"] = int(np.mean([r["confidence"] for r in dominant_entries]))
            system_result["severity"] = max([r["severity"] for r in dominant_entries], key=lambda x: ["Low","Medium","High"].index(x))
            system_result["location"] = best["location_hint"] if "location_hint" in best else "Pump-Motor System"
            system_result["details"] = dominant_entries
    
    # --- HIGH FREQUENCY AGGREGATION (Bearing) ---
    # Bearing fault bisa lokal di 1 bearing saja, threshold lebih rendah
    if high_freq_faults:
        # Prioritaskan severity tertinggi
        worst = max(high_freq_faults, key=lambda x: ["Low","Medium","High"].index(x["severity"]))
        
        # Jika severity High, override apapun yang ada di low_freq
        if worst["severity"] == "High":
            system_result["diagnosis"] = worst["diagnosis"]
            system_result["confidence"] = worst["confidence"]
            system_result["severity"] = "High"
            system_result["location"] = worst["point"]
            system_result["details"] = [worst]
        # Jika belum ada diagnosis low_freq, ambil bearing fault
        elif system_result["diagnosis"] == "Normal" or system_result["diagnosis"] == "Tidak Terdiagnosa":
            system_result["diagnosis"] = worst["diagnosis"]
            system_result["confidence"] = worst["confidence"]
            system_result["severity"] = worst["severity"]
            system_result["location"] = worst["point"]
            system_result["details"] = [worst]
    
    return system_result


# ===== ANTARMUKA STREAMLIT =====
st.set_page_config(page_title="Mechanical Vibration Expert System", layout="wide")

# Header dengan disclaimer
st.markdown("""
<div style="background-color:#E3F2FD; padding:12px; border-left:5px solid #2196F3; border-radius:4px; margin-bottom:15px">
<strong>🔧 Mechanical Vibration Expert System</strong><br>
Fokus: Centrifugal Pump + Electric Motor (Fixed Speed) | Fault: Unbalance, Misalignment, Looseness, Bearing<br>
⚠️ Decision Support Tool - Verifikasi akhir oleh Vibration Analyst Level II+ sesuai ISO 18436-2
</div>
""", unsafe_allow_html=True)

st.title("🎯 Expert System: Diagnosa Mekanikal Pump & Motor")

# ===== STEP 1: INPUT DATA MESIN =====
col1, col2 = st.columns(2)
with col1:
    machine_id = st.text_input("Machine ID / Tag Number", "P-101")
    rpm = st.number_input("Operating RPM (Fixed Speed)", min_value=600, max_value=3600, value=1780, step=10)
    service_type = st.selectbox("Service Criticality", ["Critical (Process)", "Essential (Utility)", "Standby"])
    
with col2:
    rpm_hz = rpm / 60
    st.info(f"""
    📋 **Konfigurasi Analisis**
    - 1×RPM = **{rpm_hz:.1f} Hz** | 2×RPM = **{2*rpm_hz:.1f} Hz**
    - Velocity Alarm (ISO 10816 Zone B): **4.5 mm/s**
    - Bearing Monitoring: Acceleration Bands (g)
    - Titik Ukur: **12 titik wajib** (Pump/Motor × DE/NDE × H/V/A)
    """)

# ===== STEP 2: INPUT 12 TITIK (Overall + Bands) =====
st.subheader("📊 Input Data 12 Titik Pengukuran")

points = [
    f"{machine} {end} {direction}" 
    for machine in ["Pump", "Motor"] 
    for end in ["DE", "NDE"] 
    for direction in ["Horizontal", "Vertical", "Axial"]
]

# Container untuk semua input
input_data = {}
bands_inputs = {}

# Tampilkan dalam grid 3 kolom × 4 baris
cols = st.columns(3)
for idx, point in enumerate(points):
    with cols[idx % 3]:
        with st.expander(f"📍 {point}", expanded=False):
            # Overall Velocity
            overall = st.number_input(
                "Overall Velocity (mm/s)", 
                min_value=0.0, max_value=20.0, value=1.0, step=0.1,
                key=f"vel_{point}"
            )
            input_data[point] = overall
            
            # Frequency Bands (Acceleration)
            st.caption("🔹 Frequency Bands - Acceleration (g)")
            b1 = st.number_input("Band 1: 0.5-1.5 kHz", min_value=0.0, value=0.2, step=0.05, key=f"b1_{point}")
            b2 = st.number_input("Band 2: 1.5-5 kHz", min_value=0.0, value=0.15, step=0.05, key=f"b2_{point}")
            b3 = st.number_input("Band 3: 5-16 kHz", min_value=0.0, value=0.1, step=0.05, key=f"b3_{point}")
            bands_inputs[point] = {"Band1": b1, "Band2": b2, "Band3": b3}
            
            # Status indicator
            if overall > ISO_LIMITS_VELOCITY["Zone B (Acceptable)"]:
                st.warning(f"⚠️ Velocity > 4.5 mm/s")
            if b3 > 2*ACCEL_BASELINE["Band3 (5-16kHz)"]:
                st.error(f"🔴 Band 3 tinggi - Cek bearing/lubrication")

# ===== STEP 3: INPUT FFT (HANYA JIKA ADA ANOMALI) =====
flagged_points = [p for p in points if input_data[p] > ISO_LIMITS_VELOCITY["Zone B (Acceptable)"]]
bearing_alert_points = [p for p in points if bands_inputs[p]["Band3"] > 2*ACCEL_BASELINE["Band3 (5-16kHz)"]]

if flagged_points or bearing_alert_points:
    st.divider()
    st.subheader("📈 Input FFT Spectrum (Top 3 Peaks)")
    
    # Hanya minta FFT untuk titik yang flagged atau bearing alert
    targets = list(set(flagged_points + bearing_alert_points))
    st.info(f"💡 Input FFT untuk {len(targets)} titik dengan anomali. Titik normal tidak perlu FFT.")
    
    fft_inputs = {}
    tabs = st.tabs(targets)
    
    for idx, point in enumerate(targets):
        with tabs[idx]:
            st.write(f"**{point}** | Velocity: {input_data[point]:.2f} mm/s | Band3: {bands_inputs[point]['Band3']:.3f} g")
            peaks = []
            for i in range(1,4):
                c1, c2 = st.columns(2)
                with c1:
                    default_freq = rpm_hz * i if i <= 2 else rpm_hz  # Prioritize 1x, 2x
                    freq = st.number_input(f"Peak {i} Freq (Hz)", min_value=0.1, value=default_freq, key=f"{point}_f{i}")
                with c2:
                    amp = st.number_input(f"Peak {i} Amp (mm/s)", min_value=0.01, value=1.0, step=0.1, key=f"{point}_a{i}")
                peaks.append((freq, amp))
            
            # Mini chart
            df = pd.DataFrame(peaks, columns=["Frequency (Hz)", "Amplitude (mm/s)"])
            st.bar_chart(df.set_index("Frequency (Hz)"), height=150)
            fft_inputs[point] = peaks
    
    # ===== TOMBOL ANALISIS =====
    if st.button("🔍 JALANKAN EXPERT SYSTEM", type="primary", use_container_width=True):
        st.divider()
        
        # 1. Diagnosa per titik
        point_results = []
        for point in points:
            peaks = fft_inputs.get(point, [(0,0),(0,0),(0,0)])  # Default jika tidak di-input
            bands = bands_inputs[point]
            
            result = diagnose_single_point(peaks, bands, rpm_hz, point, input_data[point])
            result["point"] = point
            # Tambahkan location hint
            if "Pump" in point:
                result["location_hint"] = f"Pump {point.split()[1]} {point.split()[2]}"
            else:
                result["location_hint"] = f"Motor {point.split()[1]} {point.split()[2]}"
            
            point_results.append(result)
        
        # 2. Agregasi sistem
        system = aggregate_system_diagnosis(point_results, input_data, points)
        
        # 3. Tampilkan hasil sistem
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Diagnosis Sistem", system["diagnosis"], delta=f"{system['confidence']}%" if system["confidence"]>0 else None)
        with col_b:
            st.metric("Lokasi Terindikasi", system["location"])
        with col_c:
            severity_color = {"Low":"🟢","Medium":"🟠","High":"🔴"}.get(system["severity"],"⚪")
            st.metric("Severity", f"{severity_color} {system['severity']}")
        
        # 4. Rekomendasi
        if system["diagnosis"] != "Normal":
            st.success(f"### 🎯 {system['diagnosis']} (Confidence: {system['confidence']}%)")
            st.info(get_mechanical_recommendation(system["diagnosis"], system["location"], system["severity"]))
        else:
            st.success("### ✅ Semua parameter dalam batas normal - Lanjutkan monitoring rutin")
        
        # 5. Detail per titik (expander)
        with st.expander("🔍 Detail Diagnosa Per Titik"):
            for r in point_results:
                if r["diagnosis"] != "Normal":
                    with st.container():
                        icon = "🔴" if r["severity"]=="High" else "🟠" if r["severity"]=="Medium" else "🟡"
                        st.write(f"{icon} **{r['point']}**: {r['diagnosis']} ({r['confidence']}%) | Severity: {r['severity']}")
                        if r["fault_type"] == "low_freq":
                            st.caption(f"Pattern: FFT peaks @ {rpm_hz:.1f}/{2*rpm_hz:.1f} Hz")
                        else:
                            b = bands_inputs[r["point"]]
                            st.caption(f"Pattern: Bands → B1:{b['Band1']:.2f}g | B2:{b['Band2']:.2f}g | B3:{b['Band3']:.2f}g")
                        st.divider()
        
        # 6. Ekspor
        st.divider()
        csv = generate_csv_report_mechanical(machine_id, rpm, points, input_data, fft_inputs, bands_inputs)
        st.download_button(
            "📥 Unduh Laporan CSV",
            data=csv,
            file_name=f"VIB_MECH_{machine_id}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

else:
    # Semua normal
    st.success("✅ **SEMUA 12 TITIK NORMAL** - Tidak diperlukan analisis FFT")
    st.balloons()
    st.info("""
    📌 **Status**: Continue routine monitoring
    - Rekam data ini sebagai baseline
    - Interval pengukuran berikutnya: sesuai maintenance plan
    - Jika ada perubahan operating condition, ulangi pengukuran
    """)

# Footer
st.divider()
st.caption("""
**Standar Acuan**: ISO 10816-3/7 (Vibration Severity) | ISO 13373-1 (Fault Diagnosis) | API 610 (Pumps)  
**Algoritma**: Rule-based expert system dengan weighted voting | Confidence scoring berbasis pattern matching  
⚠️ Tool ini adalah Decision Support System. Keputusan maintenance kritis memerlukan verifikasi oleh personnel kompeten.
""")
