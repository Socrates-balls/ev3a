# Filename: ev3a_debug_images.py
import streamlit as st
import cvxpy as cp
import numpy as np
from datetime import datetime, timedelta
import os # For checking image paths

# Plotly imports
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart EV Charging Pro",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
TARIFF_FULL_24H = np.array(
    [2.75]*16 + [3.50]*8  + [6.50]*4  + [7.00]*12 + [3.00]*6  + [5.00]*10 +
    [3.25]*4  + [4.75]*8  + [6.25]*4  + [7.50]*8  + [7.25]*4  + [5.50]*4  + [4.00]*8
)
DEFAULT_CAPACITY_KWH = 60.0
DEFAULT_MAX_POWER_KW = 7.0
DEFAULT_ALPHA = 0.01
DT_HR = 0.25

# --- EV Model Data (Incorporating your provided list) ---
EV_MODELS = [
    {
        "name": "Custom / Manual Input",
        "capacity_kwh": DEFAULT_CAPACITY_KWH,
        "max_power_kw": DEFAULT_MAX_POWER_KW,
        "image_path": None
    },
    { "name": "BMW i4", "capacity_kwh": 83.9, "max_power_kw": 11.0, "image_path": "assets/bmwi4.png" },
    { "name": "Volvo XC40 Recharge", "capacity_kwh": 78.0, "max_power_kw": 11.0, "image_path": "assets/volvoxc40.png" },
    { "name": "Kia EV6", "capacity_kwh": 77.4, "max_power_kw": 11.0, "image_path": "assets/kiaev6.png" },
    { "name": "MG ZS EV", "capacity_kwh": 50.3, "max_power_kw": 7.4, "image_path": "assets/mgzsev.png" },
    { "name": "Mahindra XUV400 EV", "capacity_kwh": 39.4, "max_power_kw": 7.2, "image_path": "assets/mhxuv400.png" },
    { "name": "Mercedes-Benz EQE SUV", "capacity_kwh": 90.6, "max_power_kw": 11.0, "image_path": "assets/benxeqe.png" },
    { "name": "Tata Nexon EV", "capacity_kwh": 40.5, "max_power_kw": 7.2, "image_path": "assets/nexonev.png" },
    { "name": "BMW i7", "capacity_kwh": 101.7, "max_power_kw": 11.0, "image_path": "assets/bmwic7.png" },
    { "name": "Volvo C40 Recharge", "capacity_kwh": 78.0, "max_power_kw": 11.0, "image_path": "assets/volvoc40.png" },
    { "name": "Kia EV9", "capacity_kwh": 99.8, "max_power_kw": 11.0, "image_path": "assets/kiaev9.png" }
]

# --- Helper Functions (calculate_max_achievable_soc, calculate_naive_charge_immediately_cost, run_optimization) ---
# ... (These functions should be the same as in the previous version, ensure they are copied correctly) ...
def calculate_max_achievable_soc(
    plug_time_str, return_time_str,
    initial_soc_pct, capacity_kWh, max_power_kW
):
    try:
        initial_soc = initial_soc_pct / 100.0
        if not (plug_time_str and return_time_str and isinstance(capacity_kWh, (int, float)) and capacity_kWh > 0 and isinstance(max_power_kW, (int, float)) and max_power_kW > 0):
            return initial_soc_pct
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        plug_time_dt_obj = datetime.strptime(plug_time_str, "%H:%M").replace(year=today.year, month=today.month, day=today.day)
        return_time_dt_obj = datetime.strptime(return_time_str, "%H:%M").replace(year=today.year, month=today.month, day=today.day)
        if return_time_dt_obj <= plug_time_dt_obj: return_time_dt_obj += timedelta(days=1)
        time_difference_seconds = (return_time_dt_obj - plug_time_dt_obj).total_seconds()
        if time_difference_seconds <= 0: return initial_soc_pct
        charging_window_hours = time_difference_seconds / 3600.0
        max_energy_deliverable_kWh = max_power_kW * charging_window_hours
        max_soc_gain = max_energy_deliverable_kWh / capacity_kWh if capacity_kWh > 0 else 0
        max_final_soc = min(initial_soc + max_soc_gain, 1.0)
        return max_final_soc * 100
    except ValueError: return None 
    except TypeError: return None 
    except Exception: return None 

def calculate_naive_charge_immediately_cost(
    initial_soc, target_soc, capacity_kWh, max_power_kW,
    plug_time_dt_obj, return_time_dt_obj,
    tariff_full_48h_np, times_full_48h_list
):
    soc_needed = target_soc - initial_soc
    if soc_needed <= 1e-4: return 0.0, np.array([]), [], []
    energy_required_kWh = soc_needed * capacity_kWh
    slots_needed_ideal = energy_required_kWh / (max_power_kW * DT_HR) if (max_power_kW * DT_HR) > 0 else float('inf')
    try: plug_time_idx_full = next(i for i, t in enumerate(times_full_48h_list) if t >= plug_time_dt_obj)
    except StopIteration: return -1, None, None, None 
    naive_cost, charged_energy_kWh = 0.0, 0.0
    naive_power_schedule_full = np.zeros(len(times_full_48h_list))
    active_charge_indices = []
    for slot_offset in range(min(int(np.ceil(slots_needed_ideal)) + 2, len(times_full_48h_list) - plug_time_idx_full)):
        current_slot_idx = plug_time_idx_full + slot_offset
        if current_slot_idx >= len(tariff_full_48h_np) or times_full_48h_list[current_slot_idx] >= return_time_dt_obj: break
        remaining_energy_needed = energy_required_kWh - charged_energy_kWh
        if remaining_energy_needed <= 1e-4: break
        power_this_slot = min(max_power_kW, remaining_energy_needed / DT_HR if DT_HR > 0 else float('inf'))
        power_this_slot = max(0, power_this_slot) 
        naive_power_schedule_full[current_slot_idx] = power_this_slot
        naive_cost += tariff_full_48h_np[current_slot_idx] * power_this_slot * DT_HR
        charged_energy_kWh += power_this_slot * DT_HR
        active_charge_indices.append(current_slot_idx)
    if not active_charge_indices: return naive_cost, np.array([]), [], []
    try: return_time_idx_full = next(i for i, t in enumerate(times_full_48h_list) if t >= return_time_dt_obj)
    except StopIteration: return_time_idx_full = len(times_full_48h_list)
    display_start_idx = plug_time_idx_full
    display_end_idx = min(return_time_idx_full, active_charge_indices[-1] + 1 if active_charge_indices else plug_time_idx_full)
    display_end_idx = max(display_end_idx, display_start_idx) 
    actual_naive_power = naive_power_schedule_full[display_start_idx : display_end_idx]
    actual_naive_times = times_full_48h_list[display_start_idx : display_end_idx]
    actual_naive_tariff = tariff_full_48h_np[display_start_idx : display_end_idx]
    return naive_cost, actual_naive_power, actual_naive_times, actual_naive_tariff

def run_optimization(plug_time_str, return_time_str, initial_soc_pct, target_soc_pct,
                     tariff_data_24h, capacity_kWh, max_power_kW, alpha,
                     enable_solver_verbose_output=False):
    try:
        initial_soc_val, target_soc_val = initial_soc_pct/100.0, target_soc_pct/100.0
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        plug_time_dt_obj = datetime.strptime(plug_time_str, "%H:%M").replace(year=today.year, month=today.month, day=today.day)
        return_time_dt_obj = datetime.strptime(return_time_str, "%H:%M").replace(year=today.year, month=today.month, day=today.day)
        if return_time_dt_obj <= plug_time_dt_obj: return_time_dt_obj += timedelta(days=1)
        times_full_48h_list = [today + timedelta(minutes=15 * i) for i in range(96 * 2)]
        tariff_full_48h_np = np.tile(tariff_data_24h, 2)
        try:
            start_idx_full = next(i for i, t in enumerate(times_full_48h_list) if t >= plug_time_dt_obj)
            end_idx_full = next(i for i, t in enumerate(times_full_48h_list) if t >= return_time_dt_obj)
        except StopIteration: st.error("Plug-in/return time outside 48h window."); return None
        if start_idx_full >= end_idx_full:
            st.warning("No valid charging window.")
            return (None, np.array([initial_soc_val]), [], [], 0.0, DT_HR, 0.0, initial_soc_val, target_soc_val, capacity_kWh, max_power_kW, plug_time_dt_obj, return_time_dt_obj, tariff_full_48h_np, times_full_48h_list)
        times_window_smart_list, tariff_window_smart_np = times_full_48h_list[start_idx_full:end_idx_full], tariff_full_48h_np[start_idx_full:end_idx_full]
        T_slots = len(tariff_window_smart_np)
        if T_slots == 0:
            st.info("No full 15-min slots for smart charging.")
            if initial_soc_val < target_soc_val: st.warning(f"Cannot reach target SoC.")
            return (np.array([]), np.array([initial_soc_val]), [], [], 0.0, DT_HR, 0.0, initial_soc_val, target_soc_val, capacity_kWh, max_power_kW, plug_time_dt_obj, return_time_dt_obj, tariff_full_48h_np, times_full_48h_list)
        p_var, soc_var = cp.Variable(T_slots, name="power"), cp.Variable(T_slots + 1, name="soc")
        constraints = [soc_var[0]==initial_soc_val, soc_var[T_slots]>=target_soc_val, soc_var>=0, soc_var<=1.0001, p_var>=0, p_var<=max_power_kW]
        for t_idx in range(T_slots): constraints.append(soc_var[t_idx+1] == soc_var[t_idx] + (p_var[t_idx]*DT_HR/capacity_kWh if capacity_kWh > 0 else 0) )
        cost_expr, health_penalty_expr = cp.sum(cp.multiply(tariff_window_smart_np, p_var)*DT_HR), cp.sum_squares(p_var)
        objective_expr = cp.Minimize(cost_expr + alpha * health_penalty_expr)
        prob = cp.Problem(objective_expr, constraints)
        if enable_solver_verbose_output: st.info("Solver verbose output in console.")
        prob.solve(solver=cp.OSQP, verbose=enable_solver_verbose_output)
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            st.error(f"Solver failed. Status: {prob.status}")
            if prob.status == cp.INFEASIBLE: st.warning("Problem INFEASIBLE.")
            return None
        p_opt_res, soc_opt_res = p_var.value, soc_var.value
        if p_opt_res is None or soc_opt_res is None: st.error("Solver returned no solution."); return None
        total_cost_val_smart = np.sum(tariff_window_smart_np * p_opt_res * DT_HR) if p_opt_res is not None else 0.0
        smart_charging_window_hours = T_slots * DT_HR
        return (p_opt_res, soc_opt_res, times_window_smart_list, tariff_window_smart_np, total_cost_val_smart, DT_HR, smart_charging_window_hours, initial_soc_val, target_soc_val, capacity_kWh, max_power_kW, plug_time_dt_obj, return_time_dt_obj, tariff_full_48h_np, times_full_48h_list)
    except ValueError as ve: st.error(f"Input error: {ve}."); return None
    except cp.error.SolverError as cpse: st.error(f"CVXPY SolverError: {cpse}"); return None
    except Exception as e: st.error(f"Optimization error: {e}"); import traceback; st.text(traceback.format_exc()); return None

# --- App UI ---
st.title("⚡ Smart EV Charging Optimizer Pro 🚗")
st.markdown("Optimize EV charging: save money, respect battery health!")
st.markdown("---")

with st.sidebar: # Sidebar Inputs
    st.header("🚗 EV Selection")
    ev_model_names = [ev["name"] for ev in EV_MODELS]
    
    if 'ev_selectbox_value' not in st.session_state:
        st.session_state.ev_selectbox_value = ev_model_names[0]
    if 'current_selected_ev_name' not in st.session_state: #This will hold the name for display
        st.session_state.current_selected_ev_name = st.session_state.ev_selectbox_value
    if 'current_capacity_kwh' not in st.session_state:
        first_ev_data_init = next((ev for ev in EV_MODELS if ev["name"] == st.session_state.ev_selectbox_value), EV_MODELS[0])
        st.session_state.current_capacity_kwh = first_ev_data_init["capacity_kwh"]
    if 'current_max_power_kw' not in st.session_state:
        first_ev_data_init = next((ev for ev in EV_MODELS if ev["name"] == st.session_state.ev_selectbox_value), EV_MODELS[0])
        st.session_state.current_max_power_kw = first_ev_data_init["max_power_kw"]

    def update_ev_params_from_selectbox():
        selected_ev_data = next((ev for ev in EV_MODELS if ev["name"] == st.session_state.ev_selectbox_value), EV_MODELS[0])
        st.session_state.current_selected_ev_name = st.session_state.ev_selectbox_value
        st.session_state.current_capacity_kwh = selected_ev_data["capacity_kwh"]
        st.session_state.current_max_power_kw = selected_ev_data["max_power_kw"]

    selected_ev_name_sb = st.selectbox(
        "Select EV Model:", ev_model_names,
        key="ev_selectbox_value",
        on_change=update_ev_params_from_selectbox
    )
    
    current_ev_display_data = next((ev for ev in EV_MODELS if ev["name"] == st.session_state.current_selected_ev_name), EV_MODELS[0])
    
    if current_ev_display_data["image_path"]:
        if os.path.exists(current_ev_display_data["image_path"]):
            try:
                st.image(current_ev_display_data["image_path"], width=150, caption=current_ev_display_data["name"])
            except Exception as img_e:
                st.sidebar.error(f"Error displaying image {current_ev_display_data['image_path']}: {img_e}")
                st.sidebar.caption(f"(Image for {current_ev_display_data['name']} could not be displayed)")
        else:
            st.sidebar.caption(f"(Image for {current_ev_display_data['name']} not found at specified path: {current_ev_display_data['image_path']})")


    st.markdown("---"); st.header("🗓️ Schedule & SoC")
    plug_time_str_input = st.text_input("Plug-in Time (HH:MM)", "21:30", key="plug_time_key")
    return_time_str_input = st.text_input("Ready by Time (HH:MM)", "07:30", key="return_time_key")
    initial_soc_pct_input = st.slider("Initial SoC (%)", 0, 100, 30, key="init_soc_key")
    
    st.markdown("---"); st.header("🔌 EV & Charger Settings")
    capacity_kWh_val_input = st.number_input("Battery Capacity (kWh)", 10.0, 200.0, value=float(st.session_state.current_capacity_kwh), step=1.0, format="%.1f", key="cap_key")
    max_power_kW_val_input = st.number_input("Max Charger Power (kW)", 1.0, 50.0, value=float(st.session_state.current_max_power_kw), step=0.1, format="%.1f", key="power_key")
    alpha_val_input = st.number_input("Battery Health Weight (α)", 0.0, 1.0, DEFAULT_ALPHA, 0.001, format="%.3f", help="Higher α = smoother charging", key="alpha_key")
    
    max_achievable_soc_pct = calculate_max_achievable_soc(plug_time_str_input, return_time_str_input, initial_soc_pct_input, capacity_kWh_val_input, max_power_kW_val_input)
    max_achievable_soc_display = initial_soc_pct_input
    if max_achievable_soc_pct is not None:
        max_achievable_soc_display = max(max_achievable_soc_pct, initial_soc_pct_input)
        st.info(f"💡 Max achievable SoC: **{max_achievable_soc_display:.1f}%**")
    else: st.caption("Enter valid times/params for max achievable SoC.")
    
    slider_max_val = int(min(100, max_achievable_soc_display if max_achievable_soc_pct is not None else 100))
    slider_max_val = max(slider_max_val, int(initial_soc_pct_input))
    slider_default_val = max(int(initial_soc_pct_input), min(80, slider_max_val))
    target_soc_pct_input = st.slider("Target SoC (%)", min_value=int(initial_soc_pct_input), max_value=slider_max_val, value=slider_default_val, key="target_soc_key")
    
    with st.form(key='charging_inputs_form'):
        st.caption("Verify settings."); optimize_button_form = st.form_submit_button(label="Optimize Charging Plan ✨", use_container_width=True)

# --- Main Content Area (Plotting, Metrics etc.) ---
# ... (This part should be identical to the v1.7 code you already have that works, with Plotly) ...
if optimize_button_form:
    if not plug_time_str_input or not return_time_str_input: st.warning("📢 Enter all time inputs.")
    elif target_soc_pct_input < initial_soc_pct_input: st.error("Target SoC < Initial SoC.")
    else:
        verbose_solve_cb = st.checkbox("Enable solver verbose output (console)", value=False, key="verbose_key")
        with st.spinner("🧠 Optimizing schedule..."):
            opt_results_tuple = run_optimization(plug_time_str_input, return_time_str_input, initial_soc_pct_input, target_soc_pct_input, TARIFF_FULL_24H, capacity_kWh_val_input, max_power_kW_val_input, alpha_val_input, enable_solver_verbose_output=verbose_solve_cb)
        if opt_results_tuple:
            (p_opt_s, soc_opt_s, times_plot_s, tariff_plot_s, cost_s, _, _, init_soc_opt, target_soc_opt, cap_opt, max_p_opt, plug_dt_opt, return_dt_opt, tariff_48_opt, times_48_opt) = opt_results_tuple
            naive_cost_val, savings, savings_pct = -1, 0.0, 0.0
            if init_soc_opt < target_soc_opt and p_opt_s is not None and len(p_opt_s)>0:
                naive_cost_val, _, _, _ = calculate_naive_charge_immediately_cost(init_soc_opt, target_soc_opt, cap_opt, max_p_opt, plug_dt_opt, return_dt_opt, tariff_48_opt, times_48_opt)
                if naive_cost_val >= 0 and cost_s >=0:
                    savings = naive_cost_val - cost_s
                    if naive_cost_val > 1e-6: savings_pct = (savings / naive_cost_val) * 100
                    elif cost_s < 0 and naive_cost_val <=1e-6 : savings_pct = float('inf')
                    elif cost_s == 0 and naive_cost_val == 0: savings_pct = 0.0
            
            st.success("✅ Optimization Complete!" if p_opt_s is not None and len(p_opt_s)>0 else "⚠️ Partial Results/No Action")
            st.subheader("📊 Key Results Comparison")
            col1,col2,col3=st.columns(3);
            with col1:st.metric(label="💡 Smart Cost",value=f"₹{cost_s:.2f}" if p_opt_s is not None and len(p_opt_s)>0 else "N/A")
            with col2:st.metric(label="💨 Naive Cost",value=f"₹{naive_cost_val:.2f}" if naive_cost_val >=0 else "N/A")
            with col3:
                if savings_pct == float('inf'): delta_text = "Inf. Savings!"
                else: delta_text = f"{savings_pct:.1f}% vs Naive" if naive_cost_val > 1e-6 or cost_s < 0 else "N/A"
                st.metric(label="🎉 Savings",value=f"₹{savings:.2f}",delta=delta_text if naive_cost_val>=0 and (cost_s>=0 if p_opt_s is not None else True) else"N/A",delta_color="normal" if savings >= -1e-6 else "inverse")
            st.markdown("---")
            final_soc_achieved = soc_opt_s[-1]*100 if soc_opt_s is not None and len(soc_opt_s)>0 else initial_soc_pct_input
            delta_soc = final_soc_achieved - target_soc_pct_input
            active_charge_hrs_s = np.sum(p_opt_s > 1e-3)*DT_HR if p_opt_s is not None and len(p_opt_s)>0 else 0
            col_soc,col_dur=st.columns(2)
            with col_soc:st.metric(label="🔋 Final SoC (Smart)",value=f"{final_soc_achieved:.1f}%",delta=f"{delta_soc:.1f}% vs Target",delta_color="normal" if delta_soc >= -0.5 else "inverse")
            with col_dur:st.metric(label="🔌 Active Smart Charge Time",value=f"{active_charge_hrs_s:.2f} hrs")
            st.markdown("---")

            if p_opt_s is not None and len(p_opt_s) > 0 and soc_opt_s is not None and len(soc_opt_s)>0:
                st.subheader("📈 Smart Charging Profile (Interactive)")
                fig_plotly = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=('Optimized Charging Power', 'State of Charge (Smart)', 'Electricity Tariff'))
                fig_plotly.add_trace(go.Scatter(x=times_plot_s, y=p_opt_s, mode='lines', name='Power', line=dict(color='deepskyblue', width=2.5, shape='hv'), fill='tozeroy', fillcolor='rgba(30,144,255,0.2)'), row=1, col=1)
                x_data_soc_plotly = [plug_dt_opt + timedelta(hours=DT_HR*i) for i in range(len(soc_opt_s))] if soc_opt_s is not None else []
                if len(soc_opt_s) > 0 and len(x_data_soc_plotly) == len(soc_opt_s):
                    fig_plotly.add_trace(go.Scatter(x=x_data_soc_plotly, y=soc_opt_s * 100, mode='lines+markers', name='SoC', line=dict(color='lime', width=2.5, shape='spline'), marker=dict(size=5, symbol='circle-open'), fill='tozeroy', fillcolor='rgba(40, 40, 40, 0.6)'), row=2, col=1) # Dark fill
                    fig_plotly.add_hline(y=target_soc_pct_input, line_dash="dot", annotation_text=f"Target ({target_soc_pct_input}%)", annotation_position="bottom right", line_color='red', row=2, col=1)
                fig_plotly.add_trace(go.Scatter(x=times_plot_s, y=tariff_plot_s, mode='lines', name='Tariff', line=dict(color='magenta', width=2.5, shape='hv'), fill='tozeroy', fillcolor='rgba(255,0,255,0.15)'), row=3, col=1)
                fig_plotly.update_layout(template='plotly_dark', height=700, showlegend=False, margin=dict(l=50, r=30, t=60, b=50), hovermode="x unified")
                fig_plotly.update_yaxes(title_text="Power (kW)", row=1, col=1); fig_plotly.update_yaxes(title_text="SoC (%)", range=[-5, 105], row=2, col=1); fig_plotly.update_yaxes(title_text="Tariff (₹/kWh)", row=3, col=1)
                fig_plotly.update_xaxes(title_text="Time of Day", row=3, col=1)
                st.plotly_chart(fig_plotly, use_container_width=True)
                with st.expander("📂 Raw Optimized Data (Smart)"):
                    st.dataframe({"Time": [t.strftime("%H:%M") for t in times_plot_s], "Power (kW)": np.round(p_opt_s, 2), "Tariff (₹/kWh)": np.round(tariff_plot_s,2)})
            else: st.info("No smart charging plot (e.g., target met or no valid window).")
        elif opt_results_tuple is None and optimize_button_form: st.error("🚨 Optimization issue. Check inputs or errors.")
else: st.info("👈 Adjust parameters & click 'Optimize Charging Plan ✨'")
st.markdown("---");st.markdown("<hr style='margin-top: 1em; margin-bottom: 0.5em;'><p style='text-align:center; color:grey; font-size:small;'>Smart EV Charging App v1.7 (Image Debug)</p>", unsafe_allow_html=True)