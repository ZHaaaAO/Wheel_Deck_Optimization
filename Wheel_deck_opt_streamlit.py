import streamlit as st
import numpy as np
import pandas as pd
import io

def round_quarter(x):
    # 向上0.5的取整
    return np.ceil(np.asarray(x, dtype=float) * 2) / 2

def stiffener(l, b, par, stn):
    a1, b1, P = par['a1'], par['b1'], par['P']
    alphaa, betaa, Ca_max = par['alphaa'], par['betaa'], par['Ca_max']
    sigmahg, ReH, D = par['sigmahg'], par['ReH'], par['D']
    alphas, betas, Cs_max_val = par['alphas'], par['betas'], par['Cs_max']
    
    kZ = 1.0
    if 0.6 < b1/b <= 1:
        kZ = (1.15 - 0.25 * b1/b)
    elif 1 < b1/b < 3.4:
        kZ = (1.15 - 0.25 * b1/b) * b1/b

    c = b if b1 > b else b1
    d = a1 if a1 < l else l
    
    ratio = a1/l
    if ratio <= 1:
        m = 38 / (ratio**2 - 4.7*ratio + 6.5)
    elif 1 < ratio <= 1.2:
        m1 = 38 / (1**2 - 4.7*1 + 6.5)
        m12 = 87 / (1.2**2 * (1.2**2 - 6.3*1.2 + 10.9))
        k1 = (m12 - m1) / 0.2
        k2 = m1 - k1 * 1
        m = k1 * ratio + k2
    elif 1.2 < ratio <= 2.5:
        m = 87 / (ratio**2 * (ratio**2 - 6.3*ratio + 10.9))
    elif 2.5 < ratio <= 3.5:
        m1 = 87 / (2.5**2 * (2.5**2 - 6.3*2.5 + 10.9))
        k1 = (12 - m1) / 1
        k2 = 12 - k1 * 3.5
        m = k1 * ratio + k2
    else:
        m = 12

    Cs = betas - alphas * (abs(sigmahg) / ReH)
    Cs = min(Cs, Cs_max_val)
    Cs = max(Cs, 1e-5) # Prevent division by zero

    Z = P * kZ * c * d * l / m / Cs / ReH * 1e-6
    
    alphap = min(1.0, 1.2 - b / (2.1 * l))
    kw = 1.0 if a1 >= 1.94*b else max(1.0, 1.3 - 4.2 / (a1/b + 1.8)**2)
    m_t = 13.57 if b1 > b else 38 / ((b1/b)**2 - 4.7*(b1/b) + 6.5)
    
    Ca = betaa - alphaa * abs(sigmahg) / ReH
    Ca = min(Ca, Ca_max)
    Ca = max(Ca, 1e-5) # Prevent invalid sqrt
    
    t = round_quarter(77.4 * alphap * np.sqrt(kw * c * b * P) / np.sqrt(m_t * Ca * ReH) * 1e-3)

    inertia = 0
    height, thickness = 0, 0
    A_net = 0
    
    for i in range(stn.shape[0]):
        bhw, btw = stn[i, 0], stn[i, 1]
        hw = bhw - bhw/9.2 + 2
        alpha = 1.1 + (120 - bhw)**2 / 3000 if bhw <= 120 else 1.0
        bf = alpha * (btw + bhw/6.7 - 2)
        tf = bhw/9.2 - 2
        tw = btw
        
        A = bf*tf + b*t + hw*tw
        z = (t**2/2*b + tw*hw*(hw/2+t) + bf*tf*(tf/2+hw+t)) / A
        Iy = (b*t**3/12 + b*t*(z-t/2)**2) + (tw/12*hw**3 + hw*tw*(hw/2+t-z)**2) + (bf*tf**3/12 + bf*tf*(tf/2+hw+t-z)**2)
        Wy = Iy / (hw + t + tf/2 - z) * 1e-3
        
        if Wy >= Z:
            inertia = 1
            height, thickness = stn[i, 0], stn[i, 1]
            A_net = A
            break

    num = np.floor(D * 1000 / b)
    return num, height, thickness, t, inertia, A_net

def fitness(sol, par, stn):
    l, b = sol[0], sol[1]
    num, height, thickness, t, inertia, A = stiffener(l, b, par, stn)
    mass = (A - b*t) * 7850 * inertia * 1e-6 * np.floor(par['D']*1000/b) + par['D']*1000*t*7850*inertia*1e-6
    sec_mass = mass if mass > 0 else np.inf
    fitness_val = 1.0 / mass if mass > 0 else 0
    return sol, t, sec_mass, fitness_val

def snap_to_step(val, step_val, min_val, max_val):
    val = np.round(val / step_val) * step_val
    return np.clip(val, min_val, max_val)

def run_ga(bounds, parameters, initpop, gen_max, stn, step):
    pop_size = initpop.shape[0]
    startPop = initpop.copy()
    best_history = []
    avg_history = [] # 新增：记录种群平均水平
    
    for gen in range(1, gen_max + 1):
        sorted_idx = np.argsort(startPop[:, 3])[::-1]
        startPop = startPop[sorted_idx]
        best = startPop[0].copy()
        
        best_history.append(best[3])
        
        # 新增：计算当前代种群中有效解的平均适应度
        valid_fits = startPop[startPop[:, 3] > 0][:, 3]
        avg_fit = np.mean(valid_fits) if len(valid_fits) > 0 else 0
        avg_history.append(avg_fit)
        
        elite_num = max(1, int(pop_size * 0.05))
        endPop = np.zeros_like(startPop)
        endPop[:elite_num] = startPop[:elite_num].copy()
        
        def tournament_selection():
            candidates = np.random.choice(pop_size, size=3, replace=False)
            best_cand = candidates[np.argmax(startPop[candidates, 3])]
            return startPop[best_cand].copy()
        
        for i in range(elite_num, pop_size):
            if np.random.rand() < 0.8: 
                p1 = tournament_selection()
                p2 = tournament_selection()
                a = np.random.rand()
                child = p1.copy()
                child[0:2] = p1[0:2]*a + p2[0:2]*(1-a)
            else:
                child = tournament_selection()
                
            # 修改：稍微降低大尺度变异的概率，促使曲线更显“阶梯状”爬行
            if np.random.rand() < 0.15: 
                mut_scale = 1.0 - (gen / gen_max)**1.5
                range_l = (bounds[0,1] - bounds[0,0]) * 0.15 * mut_scale
                range_b = (bounds[1,1] - bounds[1,0]) * 0.15 * mut_scale
                child[0] += np.random.uniform(-range_l, range_l)
                child[1] += np.random.uniform(-range_b, range_b)
                
            child[0] = snap_to_step(child[0], step[0], bounds[0,0], bounds[0,1])
            child[1] = snap_to_step(child[1], step[1], bounds[1,0], bounds[1,1])
            
            _, t, _, fit_val = fitness(child[0:2], parameters, stn)
            child[2] = t
            child[3] = fit_val
            
            endPop[i] = child
            
        startPop = endPop.copy()
        
    best_history.append(np.max(startPop[:, 3]))
    valid_fits_final = startPop[startPop[:, 3] > 0][:, 3]
    avg_history.append(np.mean(valid_fits_final) if len(valid_fits_final) > 0 else 0)
    
    return startPop, best_history, avg_history # 返回值增加了一个

def get_top10_unique(end_pop):
    end_pop = end_pop[np.argsort(end_pop[:, 3])[::-1]]
    
    unique_cands = []
    seen_pairs = set() 
    
    for row in end_pop:
        l, b = row[0], row[1]
        
        if (l, b) not in seen_pairs:
            seen_pairs.add((l, b))
            unique_cands.append(row)
            
        if len(unique_cands) == 10:
            break

    return np.array(unique_cands)[:, 0:2]

def main():
    st.set_page_config(page_title="Wheel Deck Optimization", layout="wide")
    
    st.title("Wheel Deck Optimization")
    st.markdown("""
    **Overview:** This program uses a **Genetic Algorithm (GA)** to optimize the structural weight (linear density) of Ro-Ro wheel decks.
    The calculation theory and constraints strictly follow **DNV-RU-SHIP-Pt3-Ch10-Sec5**.
    """)

    stn = np.array([
        [100, 6], [100, 7], [100, 8], [120, 6], [120, 7], [120, 8], [140, 7], [140, 8], [140, 9], 
        [160, 7], [160, 8], [160, 9], [180, 8], [180, 9], [180, 10], [200, 9], [200, 10], [200, 11.5],
        [220, 10], [220, 11.5], [240, 10], [240, 11], [240, 12], [260, 10], [260, 11], [260, 12], 
        [280, 11], [280, 12], [300, 11], [300, 12], [300, 13], [320, 12], [320, 13]
    ])

    st.sidebar.header("Algorithm Settings")
    individuals = st.sidebar.number_input(
        "Population Size =", 
        value=100, 
        min_value=10,
        help="The number of candidate solutions per iteration. A larger population maintains diversity and prevents premature convergence."
    )

    gen = st.sidebar.number_input(
        "Max iterations =", 
        value=30, 
        min_value=1,
        help="The number of iterations for GA. Larger values increase the chance of finding the global optimum but require more computation time."
    )

    D_val = st.sidebar.number_input("**D** =", value=30.0, step=1.0, help="Wheel Deck Width (m). Total transverse width of the deck used to calculate the total number of stiffeners and overall structural weight.")
    l_min = st.sidebar.number_input("**l_min** =", value=800.0, step=10.0, help="Min Panel Length Limit (mm). The minimum allowed stiffener span in the search space.")
    l_max = st.sidebar.number_input("**l_max** =", value=1500.0, step=10.0, help="Max Panel Length Limit (mm). The maximum allowed stiffener span in the search space.")
    b_min = st.sidebar.number_input("**b_min** =", value=500.0, step=10.0, help="Min Panel Width Limit (mm). The minimum allowed stiffener spacing in the search space.")
    b_max = st.sidebar.number_input("**b_max** =", value=1000.0, step=10.0, help="Max Panel Width Limit (mm). The maximum allowed stiffener spacing in the search space.")
    l_step = st.sidebar.number_input("**l_step** =", value=10.0, step=1.0, help="Panel Length Search Step (mm). The increment step for the stiffener span **l** in the genetic algorithm.")
    b_step = st.sidebar.number_input("**b_step** =", value=5.0, step=1.0, help="Panel Width Search Step (mm). The increment step for the stiffener spacing **b** in the genetic algorithm.")


    st.sidebar.header("Calculation Coefficients")
    alphas = st.sidebar.selectbox(
        "**αs** =", 
        [0.00, 1.00], 
        index=1,
        help="Used to calculate the permissible bending stress coefficient **Cs** as defined in **DNV-RU-SHIP-Pt3-Ch6-Sec5-Page27-Table 3 & Table 4** for stiffeners."
    )
    betas = st.sidebar.selectbox(
        "**βs** =", 
        [0.85, 0.95, 1.00, 1.10, 1.15, 1.20], 
        index=1,
        help="Used to calculate the permissible bending stress coefficient **Cs** as defined in **DNV-RU-SHIP-Pt3-Ch6-Sec5-Page27-Table 3 & Table 4** for stiffeners."
    )
    Cs_max = st.sidebar.selectbox(
        "**Cs_max** =", 
        [0.85, 0.95, 1.00, 1.15], 
        index=0,
        help="Upper limit for the permissible bending stress coefficient **Cs** as defined in **DNV-RU-SHIP-Pt3-Ch6-Sec5-Page27-Table 4**."
    )
    Ca_max = st.sidebar.selectbox(
        "**Ca_max** =", 
        [1.80, 2.00], 
        index=0,
        help="Maximum permissible bending stress coefficient for AC-I and AC-II as defined in **DNV-RU-SHIP-Pt3-Ch10-Sec5-Page46-Table1**."
    )
    alphaa = st.sidebar.selectbox(
        "**αa** =", 
        [0.00, 0.50, 1.00], 
        index=1,
        help="Used to calculate the permissible bending stress coefficient **Ca** for plate. See **DNV-RU-SHIP-Pt3-Ch10-Sec5-Page46-Table1** for details."
    )
    betaa = st.sidebar.selectbox(
        "**βa** =", 
        [1.80, 1.90, 2.00, 2.10], 
        index=1,
        help="Used to calculate the permissible bending stress coefficient **Ca** for plate. See **DNV-RU-SHIP-Pt3-Ch10-Sec5-Page46-Table1** for details."
    )

    sigmahg = st.sidebar.number_input(
        "**σhg** =", 
        value=80.0,
        help="Hull Girder Longitudinal Stress (N/mm²). The normal stress on the deck due to hull girder bending. Usually calculated from global longitudinal strength."
    )
    g_val = st.sidebar.number_input(
        "**g** =", 
        value=9.81,
        help = "Gravity Acceleration (m/s²)."
    )
    ReH = st.sidebar.selectbox(
        "**ReH** =", 
        [235.0, 315.0, 355.0, 390.0, 460.0], 
        index=0,
        help = 'Material Yield Strength (N/mm²). Used to calculate the net deck thickness *t*, the permissible bending stress coefficient *Cs* and the net section modulus *Z*, etc. See **DNV-RU-SHIP-Pt3-Ch10-Sec5** for details.'
    )
    az = st.sidebar.number_input(
        "**az** =", 
        value=4.5,
        help="Vertical Dynamic Acceleration (m/s²). Vertical acceleration determined in accordance with design load set for wheel load, as defined in **DNV-RU-SHIP-Pt3-Ch10-Sec5-2 Wheel loads**."
    )

    st.info("""
    * **Q**: Maximum axle load, t.
    * **n**: Number of  load areas on the axle.
    * **a1**: Extent in mm of the load area parallel to the stiffeners, mm.
    * **b1**: Extent in mm of the load area perpendicular to the stiffeners, mm.
    """)
    
    if 'input_df' not in st.session_state:
        st.session_state.input_df = pd.DataFrame([
            [30.0, 2, 300.0, 150.0]
        ], columns=["Q", "n", "a1", "b1"])

    if 'last_uploaded' not in st.session_state:
        st.session_state.last_uploaded = None

    uploaded_file = st.file_uploader("Import Excel/CSV File (Optional)", type=["xlsx", "xls", "csv"], help="You can import an Excel or CSV file containing 4 columns: Q, n, a1, b1.")
    if uploaded_file is None:
        st.session_state.last_uploaded = None
    elif uploaded_file.name != st.session_state.last_uploaded:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            df = df.dropna(how='all')
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna()
            
            if df.shape[1] >= 4:
                df = df.iloc[:, :4]
                df.columns = ["Q", "n", "a1", "b1"]
                st.session_state.input_df = df
                st.session_state.last_uploaded = uploaded_file.name
            else:
                st.error("The imported file must contain at least 4 numerical columns (Q, n, a1, b1).")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    edited_df = st.data_editor(st.session_state.input_df, num_rows="dynamic", use_container_width=True)

    #Radio Button: Load Case Selection
    state_var = st.radio(
        "Load Case:",
        options=[1, 2],
        format_func=lambda x: "At Harbour" if x==1 else "At Seas",
        horizontal=True,
            )

    if st.button("Run", type="primary"):
        for col in edited_df.columns:
            edited_df[col] = pd.to_numeric(edited_df[col], errors='coerce')
        edited_df = edited_df.dropna()
        
        if len(edited_df) == 0:
            st.warning("The input table contains no valid numerical data. Please check your inputs.")
        else:
            with st.spinner("Optimizing... This may take a moment."):
                bounds = np.array([[l_min, l_max], [b_min, b_max]])
                step = [l_step, b_step]
                table_data = edited_df.values
                num_loads = table_data.shape[0]

                Q0, n0_val, a10, b10 = table_data[0]
                P0 = (Q0/n0_val/a10/b10 * (g_val + 3/np.sqrt(Q0)) * 1e6) if state_var == 1 else (Q0/n0_val/a10/b10 * (g_val + az) * 1e6)
                
                def init_pop(par_dict):
                    pop = np.zeros((individuals, 4))
                    random1 = np.arange(bounds[0, 0], bounds[0, 1] + step[0], step[0])
                    random2 = np.arange(bounds[1, 0], bounds[1, 1] + step[1], step[1])
                    pop[:, 0] = np.random.choice(random1, individuals)
                    pop[:, 1] = np.random.choice(random2, individuals)
                    for i in range(individuals):
                        _, t, _, fit_val = fitness(pop[i, 0:2], par_dict, stn)
                        pop[i, 2] = t
                        pop[i, 3] = fit_val
                    return pop
                
                # --- Perpendicular ---
                par_perp = {
                    'a1': a10, 'b1': b10, 'P': P0, 'alphaa': alphaa, 'betaa': betaa, 
                    'Ca_max': Ca_max, 'sigmahg': sigmahg, 'ReH': ReH, 'D': D_val,
                    'alphas': alphas, 'betas': betas, 'Cs_max': Cs_max
                }
                pop_perp = init_pop(par_perp)
                end_perp, hist_perp, avg_hist_perp = run_ga(bounds, par_perp, pop_perp, gen, stn, step) 
                cands_perp = get_top10_unique(end_perp)

                # --- Parallel ---
                par_para = {
                    'a1': b10, 'b1': a10, 'P': P0, 'alphaa': alphaa, 'betaa': betaa, 
                    'Ca_max': Ca_max, 'sigmahg': sigmahg, 'ReH': ReH, 'D': D_val,
                    'alphas': alphas, 'betas': betas, 'Cs_max': Cs_max 
                }
                pop_para = init_pop(par_para)
                end_para, hist_para, avg_hist_para = run_ga(bounds, par_para, pop_para, gen, stn, step) # 接收 avg_hist_para
                cands_para = get_top10_unique(end_para)

                def evaluate_candidates(candidates, is_parallel=False):
                    results = []
                    for cand in candidates:
                        tr = np.zeros((num_loads, 7))
                        valid = True
                        for ite in range(num_loads):
                            Q, n0_i, a1, b1 = table_data[ite]
                            P = (Q/n0_i/a1/b1 * (g_val + 3/np.sqrt(Q)) * 1e6) if state_var == 1 else (Q/n0_i/a1/b1 * (g_val + az) * 1e6)
                            
                            par = {
                                'a1': b1 if is_parallel else a1,
                                'b1': a1 if is_parallel else b1,
                                'P': P, 'alphaa': alphaa, 'betaa': betaa, 'Ca_max': Ca_max,
                                'sigmahg': sigmahg, 'ReH': ReH, 'D': D_val,
                                'alphas': alphas, 'betas': betas, 'Cs_max': Cs_max
                            }
                                
                            sol, t, secm, _ = fitness(cand, par, stn)
                            n11, h11, t11, _, _, _ = stiffener(cand[0], cand[1], par, stn)
                            
                            if secm == np.inf:
                                valid = False; break
                                
                            tr[ite, :] = [cand[0], cand[1], round_quarter(t), n11, h11, t11, secm]
                        
                        if valid:
                            total_mass = np.sum(tr[:, 6])
                            results.append((total_mass, tr))
                    results.sort(key=lambda x: x[0])
                    return results

                res_perp = evaluate_candidates(cands_perp, False)
                res_para = evaluate_candidates(cands_para, True)
                
                if not res_perp and not res_para:
                    st.error("All combinations failed the section modulus or plate thickness strength constraints. Please expand the dimension bounds or reduce the load requirements.")
                else:
                    st.session_state.res_perp = res_perp
                    st.session_state.res_para = res_para
                    st.session_state.hist_perp = hist_perp
                    st.session_state.hist_para = hist_para
                    st.session_state.avg_hist_perp = avg_hist_perp # 新增
                    st.session_state.avg_hist_para = avg_hist_para # 新增
                    st.success("Optimization Complete!")

    # Results Display Area
    if 'res_perp' in st.session_state and 'res_para' in st.session_state:
        res_perp = st.session_state.res_perp
        res_para = st.session_state.res_para
        hist_perp = st.session_state.hist_perp
        hist_para = st.session_state.hist_para

        def check_convergence(hist):
            if len(hist) < 5: return True
            return (hist[-1] - hist[-5]) < 1e-6

        if not check_convergence(hist_perp) or not check_convergence(hist_para):
            st.error("Convergence Note: The GA has not fully converged. It is recommended to increase 'Iterations' or 'Population Size' in the sidebar.")

        st.divider()
        st.subheader("Optimization Convergence ")
        
        avg_hist_perp = st.session_state.avg_hist_perp
        avg_hist_para = st.session_state.avg_hist_para
        
        mass_best_perp = [1.0 / f if f > 0 else np.nan for f in hist_perp]
        mass_avg_perp = [1.0 / f if f > 0 else np.nan for f in avg_hist_perp]

        mass_best_para = [1.0 / f if f > 0 else np.nan for f in hist_para]
        mass_avg_para = [1.0 / f if f > 0 else np.nan for f in avg_hist_para]
        
        view_option = st.radio("Select Layout for Convergence Plot:", ["Perpendicular", "Parallel"], horizontal=True)

        if view_option == "Perpendicular":
            df_hist = pd.DataFrame({
                "Best Mass": mass_best_perp,
                "Average Mass ": mass_avg_perp
            })
        else:
            df_hist = pd.DataFrame({
                "Best Mass": mass_best_para,
                "Average Mass ": mass_avg_para
            })
        df_hist.index.name = "Generation"

        st.line_chart(df_hist)
        st.divider()
        # ==========================================================
        col1, col2 = st.columns(2)
        table_columns = ['Span l (mm)', 'Spacing b (mm)', 'Thickness t (mm)', 'Num of Stiffeners n', 'Web Height hs (mm)', 'Stiffener Thickness ts (mm)', 'Linear Density (kg/m)']

        with col1:
            st.subheader("Perpendicular Layout")
            if res_perp:
                perp_opts = [f"Candidate {i+1} (Mass: {m:.2f} kg/m)" for i, (m, _) in enumerate(res_perp)]
                sel_perp = st.selectbox("Select Candidate to View:", perp_opts, key="sel_perp")
                idx = perp_opts.index(sel_perp)
                mass, tr = res_perp[idx]
                
                df_perp = pd.DataFrame(tr, columns=table_columns)
                st.dataframe(df_perp.style.format("{:.2f}"), use_container_width=True)
                st.markdown(f"#### **Total Linear Density = {mass:.2f} kg/m**")
            else:
                st.write("No valid solutions found.")
                
        with col2:
            st.subheader("Parallel Layout")
            if res_para:
                para_opts = [f"Candidate {i+1} (Mass: {m:.2f} kg/m)" for i, (m, _) in enumerate(res_para)]
                sel_para = st.selectbox("Select Candidate to View:", para_opts, key="sel_para")
                idx = para_opts.index(sel_para)
                mass, tr = res_para[idx]
                
                df_para = pd.DataFrame(tr, columns=table_columns)
                st.dataframe(df_para.style.format("{:.2f}"), use_container_width=True)
                st.markdown(f"#### **Total Linear Density = {mass:.2f} kg/m**")
            else:
                st.write("No valid solutions found.")
        
        st.warning("**Note**: The number of candidates depends on the **Algorithm Settings**. The above table only shows the various optimal candidates obtained in the final iteration.")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            
            perp_data = []
            for i, (m, tr) in enumerate(res_perp):
                for row in tr:
                    perp_data.append([f"Candidate {i+1}"] + list(row))
            if perp_data:
                pd.DataFrame(perp_data, columns=['Candidate'] + table_columns).to_excel(writer, sheet_name='Perpendicular', index=False)
                
            para_data = []
            for i, (m, tr) in enumerate(res_para):
                for row in tr:
                    para_data.append([f"Candidate {i+1}"] + list(row))
            if para_data:
                pd.DataFrame(para_data, columns=['Candidate'] + table_columns).to_excel(writer, sheet_name='Parallel', index=False)
                
        output.seek(0)
        st.download_button(
            "Export Results to Excel", 
            data=output, 
            file_name="Wheel_deck_optimization_results.xlsx", 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        

    # Manual Quick Calculation Module
    st.divider()
    with st.expander("Manual Size Verification (Quick Check)"):
        st.write("Enter a specific span (l) and spacing (b) to calculate the required plate thickness, selected stiffener profile, and structural weight:")
        col_l, col_b, col_dir = st.columns(3)
        with col_l: man_l = st.number_input("Specific Span l (mm) =", value=2000.0, step=10.0)
        with col_b: man_b = st.number_input("Specific Spacing b (mm) =", value=500.0, step=10.0)
        with col_dir: man_dir = st.selectbox("Stiffener Direction", ["Perpendicular", "Parallel"])
        
        if st.button("Calculate Manual"):
            for col in edited_df.columns:
                edited_df[col] = pd.to_numeric(edited_df[col], errors='coerce')
            edited_df = edited_df.dropna()
            
            if len(edited_df) == 0:
                st.error("The input table contains no valid numerical data.")
            else:
                is_parallel = man_dir == "Parallel"
                table_data = edited_df.values
                total_mass = 0
                
                tr_res = []
                for row in table_data:
                    Q, n0_i, a1, b1 = row
                    P = (Q/n0_i/a1/b1 * (g_val + 3/np.sqrt(Q)) * 1e6) if state_var == 1 else (Q/n0_i/a1/b1 * (g_val + az) * 1e6)
                    
                    par = {
                        'a1': b1 if is_parallel else a1,
                        'b1': a1 if is_parallel else b1,
                        'P': P, 'alphaa': alphaa, 'betaa': betaa, 'Ca_max': Ca_max,
                        'sigmahg': sigmahg, 'ReH': ReH, 'D': D_val,
                        'alphas': alphas, 'betas': betas, 'Cs_max': Cs_max
                    }
                    
                    _, t, secm, _ = fitness([man_l, man_b], par, stn)
                    n11, h11, t11, _, _, _ = stiffener(man_l, man_b, par, stn)
                    
                    tr_row = [man_l, man_b, round_quarter(t), n11, h11, t11, secm]
                    tr_res.append(tr_row)
                    
                    if secm != np.inf:
                        total_mass += secm
                
                df_man = pd.DataFrame(tr_res, columns=['Span l (mm)', 'Spacing b (mm)', 'Plate Thickness t (mm)', 'Num of Stiffeners n', 'Web Height hs (mm)', 'Stiffener Thickness ts (mm)', 'Linear Density (kg/m)'])
                st.dataframe(df_man.style.format("{:.2f}"), use_container_width=True)
                st.markdown(f"#### **Total Linear Density = {total_mass:.2f} kg/m**")

if __name__ == "__main__":
    main()
