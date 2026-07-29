import streamlit as st
import numpy as np
import pandas as pd
import io
import openpyxl
import plotly.graph_objects as go

def round_quarter(x):
    return np.round(np.asarray(x, dtype=float) * 2) / 2

def calc_mass(t, A_stiff, b, D):
    num = np.floor(D * 1000 / b)
    mass = (A_stiff * 7850 * 1e-6 * num) + (D * 1000 * t * 7850 * 1e-6)
    return mass

def get_deck_reqs(l, b, row, par_base, state_var, g_val, az):
    Q, n0_val, a10, b10 = row
    P = (Q/n0_val/a10/b10 * (g_val + 3/np.sqrt(Q)) * 1e6) if state_var == 1 else (Q/n0_val/a10/b10 * (g_val + az) * 1e6)
    
    def calc_req(a1_val, b1_val):
        kZ = 1.0
        if 0.6 < b1_val/b <= 1: kZ = (1.15 - 0.25 * b1_val/b)
        elif 1 < b1_val/b < 3.4: kZ = (1.15 - 0.25 * b1_val/b) * b1_val/b

        c_val = b if b1_val > b else b1_val
        d_val = a1_val if a1_val < l else l
        
        ratio = a1_val/l
        if ratio <= 1: m = 38 / (ratio**2 - 4.7*ratio + 6.5)
        elif 1 < ratio <= 1.2:
            m1 = 38 / (1**2 - 4.7*1 + 6.5)
            m12 = 87 / (1.2**2 * (1.2**2 - 6.3*1.2 + 10.9))
            k1 = (m12 - m1) / 0.2
            k2 = m1 - k1 * 1
            m = k1 * ratio + k2
        elif 1.2 < ratio <= 2.5: m = 87 / (ratio**2 * (ratio**2 - 6.3*ratio + 10.9))
        elif 2.5 < ratio <= 3.5:
            m1 = 87 / (2.5**2 * (2.5**2 - 6.3*2.5 + 10.9))
            k1 = (12 - m1) / 1
            k2 = 12 - k1 * 3.5
            m = k1 * ratio + k2
        else: m = 12

        Cs = par_base['betas'] - par_base['alphas'] * (abs(par_base['sigmahg']) / par_base['ReH'])
        Cs = np.clip(Cs, 1e-5, par_base['Cs_max'])

        Z_val = P * kZ * c_val * d_val * l / m / Cs / par_base['ReH'] * 1e-6
        
        alphap = min(1.0, 1.2 - b / (2.1 * l))
        kw = 1.0 if a1_val >= 1.94*b else (1.3 - 4.2 / (a1_val/b + 1.8)**2)
        m_t = 13.57 if b1_val > b else 38 / ((b1_val/b)**2 - 4.7*(b1_val/b) + 6.5)
        
        Ca = par_base['betaa'] - par_base['alphaa'] * abs(par_base['sigmahg']) / par_base['ReH']
        Ca = np.clip(Ca, 1e-5, par_base['Ca_max'])
        
        t_min_val = round_quarter(77.4 * alphap * np.sqrt(kw * c_val * b * P) / np.sqrt(m_t * Ca * par_base['ReH']) * 1e-3)
        return Z_val, t_min_val

    Z_pe, t_pe = calc_req(a10, b10)
    Z_pa, t_pa = calc_req(b10, a10)

    return (Z_pe, t_pe), (Z_pa, t_pa), (max(Z_pe, Z_pa), max(t_pe, t_pa))

def calc_actual_section_modulus(t_deck, bhw, btw, b):
    # 根据输入的板厚、筋高、筋厚以及间距，计算实际的剖面模数 (Z)
    hw = bhw - bhw/9.2 + 2
    alpha = 1.1 + (120 - bhw)**2 / 3000 if bhw <= 120 else 1.0
    bf = alpha * (btw + bhw/6.7 - 2)
    tf = bhw/9.2 - 2
    tw = btw
    
    A_stiff = (hw-tf)*tw + bf*tf
    A = A_stiff + b*t_deck
    
    z = (t_deck**2/2*b + tw*hw*(hw/2+t_deck) + bf*tf*(tf/2+hw+t_deck)) / A
    Iy = (b*t_deck**3/12 + b*t_deck*(z-t_deck/2)**2) + (tw/12*hw**3 + hw*tw*(hw/2+t_deck-z)**2) + (bf*tf**3/12 + bf*tf*(tf/2+hw+t_deck-z)**2)
    Wy = Iy / (hw + t_deck + tf - z) * 1e-3
    
    return Wy

def stiffener_search(Z, t_min, l, b, par_base, stn):
    best_local_mass = np.inf
    opt_t = t_min
    opt_height, opt_thickness = 0, 0
    opt_inertia = 0
    D = par_base['D']

    for t_test in np.arange(t_min, t_min + 10.5, 0.5):
        for i in range(stn.shape[0]):
            bhw, btw = stn[i, 0], stn[i, 1]
            hw = bhw - bhw/9.2 + 2
            alpha = 1.1 + (120 - bhw)**2 / 3000 if bhw <= 120 else 1.0
            bf = alpha * (btw + bhw/6.7 - 2)
            tf = bhw/9.2 - 2
            tw = btw
            A_stiff = hw*tw + bf*tf
            A = A_stiff + b*t_test
            z = (t_test**2/2*b + tw*hw*(hw/2+t_test) + bf*tf*(tf/2+hw+t_test)) / A
            Iy = (b*t_test**3/12 + b*t_test*(z-t_test/2)**2) + (tw/12*hw**3 + hw*tw*(hw/2+t_test-z)**2) + (bf*tf**3/12 + bf*tf*(tf/2+hw+t_test-z)**2)
            Wy = Iy / (hw + t_test + tf - z) * 1e-3
            
            if Wy >= Z:
                current_mass = calc_mass(t_test, A_stiff, b, D)
                if current_mass < best_local_mass:
                    best_local_mass = current_mass
                    opt_t = t_test
                    opt_height, opt_thickness = bhw, btw
                    opt_inertia = 1
                break 
    
    if opt_inertia == 0:
        return np.inf, t_min, 0, 0, 0
        
    return best_local_mass, opt_t, opt_height, opt_thickness, opt_inertia

def fitness_multi_deck(sol, table_data, par_base, state_var, g_val, az, stn):
    l, b = sol[0], sol[1]
    total_mass = 0
    valid = True
    
    for row in table_data:
        _, _, req_u = get_deck_reqs(l, b, row, par_base, state_var, g_val, az)
        mass, _, _, _, inertia = stiffener_search(req_u[0], req_u[1], l, b, par_base, stn)
        
        # 只要有任意一层甲板在当前(l, b)下无解，整个方案判定失效
        if inertia == 0 or mass == np.inf:
            valid = False
            break
        total_mass += mass
        
    fit_val = 1.0 / total_mass if valid and total_mass > 0 else 0
    total_mass_out = total_mass if valid else np.inf
    return total_mass_out, fit_val

def snap_to_step(val, step_val, min_val, max_val):
    val = np.round(val / step_val) * step_val
    return np.clip(val, min_val, max_val)

def update_hof(hof, valid_pop):
    combined = np.vstack((hof, valid_pop)) if len(hof) > 0 else valid_pop
    if len(combined) == 0: return combined
    # 按适应度降序
    combined = combined[np.argsort(combined[:, 3])[::-1]]
    
    unique_cands = []
    seen_keys = set()
    for row in combined:
        l, b = row[0], row[1]
        key = b if l >= 2.1 * b else (l, b)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_cands.append(row)
        if len(unique_cands) == 10:
            break
    return np.array(unique_cands)

def run_ga_unified(bounds, table_data, par_base, state_var, g_val, az, initpop, gen_max, stn, step):
    pop_size = initpop.shape[0]
    startPop = initpop.copy()
    
    hof = np.empty((0, 4)) 
    history = {'gen': [], 'best_mass': [], 'avg_mass': []}

    for gen in range(1, gen_max + 1):
        valid_pop = startPop[startPop[:, 3] > 0]
        valid_pop = valid_pop[np.argsort(valid_pop[:, 3])[::-1]]
        
        hof = update_hof(hof, valid_pop)
        
        if len(valid_pop) > 0:
            history['gen'].append(gen)
            history['best_mass'].append(valid_pop[0, 2])
            history['avg_mass'].append(np.mean(valid_pop[:, 2]))
        else:
            history['gen'].append(gen)
            history['best_mass'].append(0)
            history['avg_mass'].append(0)
        
        elite_num = max(1, int(pop_size * 0.05))
        endPop = np.zeros_like(startPop)
        endPop[:elite_num] = startPop[np.argsort(startPop[:, 3])[::-1]][:elite_num].copy()
        
        def tournament_selection():
            candidates = np.random.choice(pop_size, size=3, replace=False)
            best_cand = candidates[np.argmax(startPop[candidates, 3])]
            return startPop[best_cand].copy()
        
        for i in range(elite_num, pop_size):
            if np.random.rand() < 0.8: 
                p1, p2 = tournament_selection(), tournament_selection()
                a = np.random.rand()
                child = p1.copy()
                child[0:2] = p1[0:2]*a + p2[0:2]*(1-a)
            else:
                child = tournament_selection()
            
            if np.random.rand() < 0.15: 
                mut_scale = 1.0 - (gen / gen_max)**1.5
                child[0] += np.random.uniform(- (bounds[0,1] - bounds[0,0]) * 0.15 * mut_scale, (bounds[0,1] - bounds[0,0]) * 0.15 * mut_scale)
                child[1] += np.random.uniform(- (bounds[1,1] - bounds[1,0]) * 0.15 * mut_scale, (bounds[1,1] - bounds[1,0]) * 0.15 * mut_scale)
                
            child[0] = snap_to_step(child[0], step[0], bounds[0,0], bounds[0,1])
            child[1] = snap_to_step(child[1], step[1], bounds[1,0], bounds[1,1])
            
            sec_mass, fit_val = fitness_multi_deck(child[0:2], table_data, par_base, state_var, g_val, az, stn)
            child[2], child[3] = sec_mass, fit_val
            endPop[i] = child
            
        startPop = endPop.copy()
    
    valid_pop = startPop[startPop[:, 3] > 0]
    hof = update_hof(hof, valid_pop)
    
    return hof, history 

def evaluate_candidate_details(l, b, table_data, par_base, state_var, g_val, az, stn):
    deck_data = []
    rule_mins = []
    tot_u, tot_pe, tot_pa = 0, 0, 0
    
    stiff_n= int(np.floor(par_base['D'] * 1000 / b))
    
    for idx, row in enumerate(table_data):
        req_pe, req_pa, req_u = get_deck_reqs(l, b, row, par_base, state_var, g_val, az)
        
        rule_mins.append({
            'Deck': f"Deck {idx + 1}",
            't_min_pe': req_pe[1],
            't_min_pa': req_pa[1]
        })

        m_u, t_u, h_u, ts_u, _ = stiffener_search(req_u[0], req_u[1], l, b, par_base, stn)
        m_pe, t_pe, h_pe, ts_pe, _ = stiffener_search(req_pe[0], req_pe[1], l, b, par_base, stn)
        m_pa, t_pa, h_pa, ts_pa, _ = stiffener_search(req_pa[0], req_pa[1], l, b, par_base, stn)
        
        tot_u += m_u if m_u != np.inf else 0
        tot_pe += m_pe if m_pe != np.inf else 0
        tot_pa += m_pa if m_pa != np.inf else 0
        
        deck_data.append({
            'Deck level': f"Deck {idx + 1}",
            'Q (t)': row[0],
            'n': stiff_n,
            
            'Safe Thickness (mm)': f"{t_u:.1f}" if m_u != np.inf else "N/A",
            'Safe Stiffener': f"{int(h_u)}x{int(ts_u)}" if m_u != np.inf else "N/A",
            'Safe Weight (kg/m)': round(m_u, 2) if m_u != np.inf else "N/A",
            
            'Perpendicular Thickness (mm)': f"{t_pe:.1f}" if m_pe != np.inf else "N/A",
            'Perpendicular Stiffener': f"{int(h_pe)}x{int(ts_pe)}" if m_pe != np.inf else "N/A",
            'Perpendicular Mass (kg/m)': round(m_pe, 2) if m_pe != np.inf else "N/A",
            
            'Parallel Thickness (mm)': f"{t_pa:.1f}" if m_pa != np.inf else "N/A",
            'Parallel Stiffener': f"{int(h_pa)}x{int(ts_pa)}" if m_pa != np.inf else "N/A",
            'Parallel Mass (kg/m)': round(m_pa, 2) if m_pa != np.inf else "N/A"
        })
        
    return deck_data, tot_u, tot_pe, tot_pa, rule_mins


def main():
    st.set_page_config(page_title="Wheel Deck(s) Optimization", layout="wide")
    st.title("Wheel Deck(s) Optimization")
    st.markdown("""
    **Overview:** This program uses Genetic Algorithm (GA) to iterate over variable Wheel Loads on each deck and minimize the structural weight of Ro-Ro wheel decks.
    The calculation theory and constraints strictly comply with **DNV-RU-SHIP-Pt3-Ch10-Sec5**.
    """)

    stn = np.array([
        [100, 6], [100, 7], [100, 8], [120, 6], [120, 7], [120, 8], [140, 7], [140, 8], [140, 9], 
        [160, 7], [160, 8], [160, 9], [180, 8], [180, 9], [180, 10], [200, 9], [200, 10], [200, 11.5],
        [220, 10], [220, 11.5], [240, 10], [240, 11], [240, 12], [260, 10], [260, 11], [260, 12], 
        [280, 11], [280, 12], [300, 11], [300, 12], [300, 13], [320, 12], [320, 13]
    ])

    st.sidebar.header("Algorithm Settings")
    individuals = st.sidebar.number_input("Population Size =", value=100, min_value=10)
    gen = st.sidebar.number_input("Max iterations =", value=30, min_value=1)
    D_val = st.sidebar.number_input("**D** (Width for linear calc) =", value=30.0, step=1.0)
    l_min = st.sidebar.number_input("**l_min** =", value=2000.0, step=10.0)
    l_max = st.sidebar.number_input("**l_max** =", value=3500.0, step=10.0)
    b_min = st.sidebar.number_input("**b_min** =", value=400.0, step=10.0)
    b_max = st.sidebar.number_input("**b_max** =", value=800.0, step=10.0)
    l_step = st.sidebar.number_input("**l_step** =", value=10.0, step=1.0)
    b_step = st.sidebar.number_input("**b_step** =", value=5.0, step=1.0)

    st.sidebar.markdown("---")
    st.sidebar.header("Calculation Coefficients")
    alphas = st.sidebar.selectbox("**αs** =", [0.00, 1.00], index=1)
    betas = st.sidebar.selectbox("**βs** =", [0.85, 0.95, 1.00, 1.10, 1.15, 1.20], index=1)
    Cs_max = st.sidebar.selectbox("**Cs_max** =", [0.85, 0.95, 1.00, 1.15], index=0)
    Ca_max = st.sidebar.selectbox("**Ca_max** =", [1.80, 2.00], index=0)
    alphaa = st.sidebar.selectbox("**αa** =", [0.00, 0.50, 1.00], index=1)
    betaa = st.sidebar.selectbox("**βa** =", [1.80, 1.90, 2.00, 2.10], index=1)
    sigmahg = st.sidebar.number_input("**σhg** =", value=80.0)
    g_val = st.sidebar.number_input("**g** =", value=9.81)
    ReH = st.sidebar.selectbox("**ReH** =", [235.0, 315.0, 355.0, 390.0, 460.0], index=0)
    az = st.sidebar.number_input("**az** =", value=4.5)

    if 'input_df' not in st.session_state:
        # 默认演示数据, 展示两层不同的载荷
        st.session_state.input_df = pd.DataFrame([
            [30.0, 2, 300.0, 150.0],
            [15.0, 2, 200.0, 150.0]
        ], columns=["Q", "n", "a1", "b1"])

    uploaded_file = st.file_uploader("Import Excel/CSV File for All Decks", type=["xlsx", "xls", "csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            df = df.dropna(how='all')
            for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna()
            if df.shape[1] >= 4:
                df = df.iloc[:, :4]
                df.columns = ["Q", "n", "a1", "b1"]
                st.session_state.input_df = df
        except Exception: pass

    st.markdown("### Deck Load Data Input (Each Row = One Deck)")
    edited_df = st.data_editor(st.session_state.input_df, num_rows="dynamic", use_container_width=True)
    state_var = st.radio("Load Case:", options=[1, 2], format_func=lambda x: "At Harbour" if x==1 else "At Seas", horizontal=True)

    if st.button("Run Optimization", type="primary"):
        for col in edited_df.columns: edited_df[col] = pd.to_numeric(edited_df[col], errors='coerce')
        edited_df = edited_df.dropna()
        
        if len(edited_df) > 0:
            with st.spinner("Calculating, please wait..."):
                bounds = np.array([[l_min, l_max], [b_min, b_max]])
                step = [l_step, b_step]
                table_data = edited_df.values
                
                par_base = {'alphaa': alphaa, 'betaa': betaa, 'Ca_max': Ca_max, 'sigmahg': sigmahg, 'ReH': ReH, 'D': D_val, 'alphas': alphas, 'betas': betas, 'Cs_max': Cs_max}

                pop = np.zeros((individuals, 4))
                pop[:, 0] = np.random.choice(np.arange(bounds[0, 0], bounds[0, 1] + step[0], step[0]), individuals)
                pop[:, 1] = np.random.choice(np.arange(bounds[1, 0], bounds[1, 1] + step[1], step[1]), individuals)
                for i in range(individuals):
                    sec_mass, fit_val = fitness_multi_deck(pop[i, 0:2], table_data, par_base, state_var, g_val, az, stn)
                    pop[i, 2:4] = [sec_mass, fit_val]
                
                hof, hist_unified = run_ga_unified(bounds, table_data, par_base, state_var, g_val, az, pop, gen, stn, step) 
                
                if len(hof) == 0: st.error("All combinations failed limits. Expand bounds or reduce loads.")
                else:
                    st.session_state.hof = hof
                    st.session_state.hist_unified = hist_unified
                    st.session_state.tb_data = table_data
                    st.session_state.par_base = par_base
                    st.success("Optimization Complete!")

    # 绘图和输出结果
    if 'hist_unified' in st.session_state:
        hist = st.session_state.hist_unified
        hof = st.session_state.hof
        table_data = st.session_state.tb_data
        par_base = st.session_state.par_base

        st.divider()
        st.subheader("Iteration curve")
        
        fig = go.Figure()
        valid_indices = [i for i, m in enumerate(hist['best_mass']) if m > 0]
        fig.add_trace(go.Scatter(
            x=[hist['gen'][i] for i in valid_indices], 
            y=[hist['best_mass'][i] for i in valid_indices], 
            mode='lines+markers', name='Best Gross Weight (kg/m)', line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=[hist['gen'][i] for i in valid_indices], 
            y=[hist['avg_mass'][i] for i in valid_indices], 
            mode='lines', name='Avg Gross Weight (kg/m)', line=dict(color='orange', dash='dot')
        ))
        fig.update_layout(height=400, xaxis_title="Generation", yaxis_title="Sum of Weights across all decks (kg/m)", hovermode="x")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Details of Top 10 Optimal Solutions")
        cand_opts = [f"Rank {i+1}: Span l={r[0]}mm, Spacing b={r[1]}mm (Gross Weight={r[2]:.2f} kg/m)" for i, r in enumerate(hof)]
        sel_cand = st.selectbox("Select Candidate Configuration:", cand_opts)
        sel_idx = cand_opts.index(sel_cand)
        sel_l, sel_b = hof[sel_idx][0], hof[sel_idx][1]
        
        deck_data, tot_u, tot_pe, tot_pa, rule_mins = evaluate_candidate_details(sel_l, sel_b, table_data, par_base, state_var, g_val, az, stn)
        
        st.markdown(f"**Detailed Layout for Span `l = {sel_l}` mm and Spacing `b = {sel_b}` mm**")
        min_thickness_text = " | ".join([
            f"**{rm['Deck']}**: Perp $t_{{min}}$ = {rm['t_min_pe']} mm, Para $t_{{min}}$ = {rm['t_min_pa']} mm" 
            for rm in rule_mins
        ])
        st.info(f"💡 **Rule Minimum Required Thickness:** {min_thickness_text}")

        df_deck_details = pd.DataFrame(deck_data)
        st.dataframe(df_deck_details, use_container_width=True)
        
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        col_sum1.metric("Sum of Gross Weight", f"{tot_u:.2f} kg/m")
        col_sum2.metric("Sum of Perpendicular Mass", f"{tot_pe:.2f} kg/m")
        col_sum3.metric("Sum of Parallel Mass", f"{tot_pa:.2f} kg/m")

        all_export_data = []
        for i, row in enumerate(hof):
            cand_l, cand_b = row[0], row[1]
            cand_deck_data, cand_tot_u, cand_tot_pe, cand_tot_pa, cand_rule_mins = evaluate_candidate_details(
                cand_l, cand_b, table_data, par_base, state_var, g_val, az, stn
            )
            for deck_row in cand_deck_data:
                export_row = {
                    'Rank': i + 1,
                    'Span l (mm)': cand_l,
                    'Spacing b (mm)': cand_b,
                }
                export_row.update(deck_row) 
                export_row['Sum (Gross) (kg/m)'] = round(cand_tot_u, 2)
                export_row['Sum (Perpendicular) (kg/m)'] = round(cand_tot_pe, 2)
                export_row['Sum (Parallel) (kg/m)'] = round(cand_tot_pa, 2)
                
                all_export_data.append(export_row)

        df_all_export = pd.DataFrame(all_export_data)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_all_export.to_excel(writer, sheet_name='Solutions', index=False)
        output.seek(0) 
        
        st.download_button(
            "Export ALL Results to Excel", 
            data=output, 
            file_name="Wheel_Deck_Optimization_Results.xlsx", 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    st.divider()
    with st.expander("Manual Calculation & Rule Validation"):
        calc_mode = st.selectbox(
            "Select Feature:", 
            ["1. Weight Distribution (All Decks)", "2. Custom Section Rule Check (Single Deck)"]
        )
        
        col_l, col_b = st.columns(2)
        with col_l: man_l = st.number_input("Span l (mm) =", value=2000.0, step=10.0, key='man_l')
        with col_b: man_b = st.number_input("Spacing b (mm) =", value=500.0, step=10.0, key='man_b')
        
        # 预先清理数据
        for col in edited_df.columns: edited_df[col] = pd.to_numeric(edited_df[col], errors='coerce')
        valid_df = edited_df.dropna()
        
        if calc_mode == "1. Weight Distribution (All Decks)":
            st.write("Input a specific Span and Spacing to calculate the weight distribution across all defined decks in your table.")
            if st.button("Calculate System Weight"):
                if len(valid_df) == 0: 
                    st.error("No valid load data.")
                else:
                    tb_data = valid_df.values
                    p_base = {'alphaa': alphaa, 'betaa': betaa, 'Ca_max': Ca_max, 'sigmahg': sigmahg, 'ReH': ReH, 'D': D_val, 'alphas': alphas, 'betas': betas, 'Cs_max': Cs_max}
                    
                    deck_data_man, t_u_man, t_pe_man, t_pa_man, rule_mins_man = evaluate_candidate_details(man_l, man_b, tb_data, p_base, state_var, g_val, az, stn)
                    
                    min_text_man = " | ".join([
                        f"**{rm['Deck']}**: Perp $t_{{min}}$ = {rm['t_min_pe']} mm, Para $t_{{min}}$ = {rm['t_min_pa']} mm" 
                        for rm in rule_mins_man
                    ])
                    st.info(f"💡 **Rule Minimum Required Thickness:** {min_text_man}")
                    
                    st.dataframe(pd.DataFrame(deck_data_man), use_container_width=True)
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Sum (Gross)", f"{t_u_man:.2f} kg/m")
                    c2.metric("Sum (Perpendicular)", f"{t_pe_man:.2f} kg/m")
                    c3.metric("Sum (Parallel)", f"{t_pa_man:.2f} kg/m")
                    
        else:
            st.write("Check if a custom deck thickness and stiffener size meets the DNV rules for a specific deck load.")
            if len(valid_df) == 0:
                st.error("Please add load data in the table above first.")
            else:
                deck_options = [f"Deck {i+1} (Q = {row[0]}t)" for i, row in enumerate(valid_df.values)]
                sel_deck_idx = st.selectbox("Select Deck Load to Check:", range(len(deck_options)), format_func=lambda x: deck_options[x])
                
                st.markdown("**Input Custom Section Dimensions:**")
                c_sec1, c_sec2, c_sec3 = st.columns(3)
                with c_sec1: custom_t = st.number_input("Deck Thickness t (mm)", min_value=1.0, value=1.0, step=0.5)
                with c_sec2: custom_h = st.number_input("Stiffener Height hw (mm)", min_value=10.0, value=100.0, step=1.0)
                with c_sec3: custom_ts = st.number_input("Stiffener Thickness tw (mm)", min_value=1.0, value=1.0, step=0.5)
                
                if st.button("Validate Custom Section"):
                    sel_row = valid_df.values[sel_deck_idx]
                    p_base = {'alphaa': alphaa, 'betaa': betaa, 'Ca_max': Ca_max, 'sigmahg': sigmahg, 'ReH': ReH, 'D': D_val, 'alphas': alphas, 'betas': betas, 'Cs_max': Cs_max}
                    
                    # 调取需求参数
                    req_pe, req_pa, req_u = get_deck_reqs(man_l, man_b, sel_row, p_base, state_var, g_val, az)
                    
                    # 计算实际参数
                    actual_z = calc_actual_section_modulus(custom_t, custom_h, custom_ts, man_b)
                    
                    #输出实际值
                    st.success(f"**Actual Provided Dimensions:** Section Modulus **$Z_{{act}}$ = {actual_z:.2f} cm³** | Deck Thickness **$t$ = {custom_t:.1f} mm**")
                    
                    #构建比对表格
                    def pass_fail(act, req):
                        return "✅ Pass" if act >= req else "❌ Fail"
                        
                    compare_data = [
                        {"Condition": "Perpendicular", 
                         "Req Z (cm³)": round(req_pe[0], 2), "Z Check": pass_fail(actual_z, req_pe[0]), 
                         "Req t (mm)": round(req_pe[1], 2), "t Check": pass_fail(custom_t, req_pe[1])},
                         
                        {"Condition": "Parallel", 
                         "Req Z (cm³)": round(req_pa[0], 2), "Z Check": pass_fail(actual_z, req_pa[0]), 
                         "Req t (mm)": round(req_pa[1], 2), "t Check": pass_fail(custom_t, req_pa[1])},
                         
                        {"Condition": "Gross", 
                         "Req Z (cm³)": round(req_u[0], 2), "Z Check": pass_fail(actual_z, req_u[0]), 
                         "Req t (mm)": round(req_u[1], 2), "t Check": pass_fail(custom_t, req_u[1])}
                    ]
                    
                    st.table(pd.DataFrame(compare_data))

if __name__ == "__main__":
    main()
