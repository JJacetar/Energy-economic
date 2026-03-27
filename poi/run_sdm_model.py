import pandas as pd
import numpy as np
import libpysal
from spreg import GM_Lag
from esda.moran import Moran
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import warnings
warnings.filterwarnings("ignore")
# ================= 1. 加载面板数据 =================
try:
    df = pd.read_csv("附图及csv/太原市_第四阶段_SDM面板.csv")
except Exception as e:
    print(f"错误: {e}")
    exit()
y_name = 'Y_空间失配度'
x_names = ['ln_X1_房价', 'ln_X2_人口', 'ln_X3_房龄', 'ln_X4_距离', 'ln_X5_交通']
Y = df[[y_name]].values
X = df[x_names].values
# ================= 2. VIF 多重共线性检验 =================
X_df = df[x_names]
X_with_const = add_constant(X_df)
vif_data = pd.DataFrame()
vif_data["变量"] = x_names
vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i + 1) for i in range(len(x_names))]
print("-" * 40)
print(vif_data.to_string(index=False, float_format="%.2f"))
print("-" * 40)
if vif_data['VIF'].max() < 10:
    print(" 所有变量 VIF < 10 ，证明不存在多重共线性，特征选取完美！")
else:
    print(" 存在 VIF > 10 的变量，可能需要剔除！")
# ================= 3. 构建行标准化空间权重矩阵 =================
coords = df[['Center_Lng', 'Center_Lat']].values
w = libpysal.weights.KNN.from_array(coords, k=8)
w.transform = 'r'
# ================= 4. 构造空间溢出项与求解 SDM 模型 =================
WX = w.sparse.dot(X)
X_sdm = np.hstack((X, WX))
x_sdm_names = x_names + [f"W_{name}" for name in x_names]
# 核心求解
sdm_model = GM_Lag(Y, X_sdm, w=w, y_name=y_name, x_name=x_sdm_names, name_w='KNN_8', robust='white')
# 提取参数
k = len(x_names)
betas_all = sdm_model.betas.flatten()
p_values_all = [stat[1] for stat in sdm_model.z_stat]
X_coefs = betas_all[1:k + 1]
WX_coefs = betas_all[k + 1:2 * k + 1]
rho = betas_all[-1]
p_val_X = p_values_all[1:k + 1]
p_val_WX = p_values_all[k + 1:2 * k + 1]
p_val_rho = p_values_all[-1]
print(f"\n   模型拟合优度 (Pseudo R-squared): {sdm_model.pr2:.4f}")
print(f"   空间自回归系数 (Rho, ρ): {rho:.4f} (P-value: {p_val_rho:.4e})")
# ================= 5. 残差的 Moran's I 检验 =================
# 提取模型残差
residuals = sdm_model.u.flatten()
moran_res = Moran(residuals, w)
print(f" 残差 Moran's I: {moran_res.I:.4f} (P-value: {moran_res.p_sim:.4f})")
if moran_res.p_sim > 0.05:
    print("   残差不存在显著的空间自相关")
else:
    print("   残差仍有轻微空间自相关")
# ================= 6. LeSage & Pace 偏微分效应分解 =================
N = len(df)
I = np.eye(N)
W_dense = w.full()[0]
V = np.linalg.inv(I - rho * W_dense)
direct_effects = []
indirect_effects = []
total_effects = []
def get_stars(p_val):
    if p_val < 0.01: return "***"
    if p_val < 0.05: return "** "
    if p_val < 0.10: return "* "
    return "   "
sig_flags_dir = []
sig_flags_ind = []
for i in range(k):
    beta_k = X_coefs[i]
    theta_k = WX_coefs[i]
    S_k = np.dot(V, (I * beta_k + W_dense * theta_k))
    de = np.trace(S_k) / N
    te = (beta_k + theta_k) / (1 - rho)
    ie = te - de
    direct_effects.append(de)
    indirect_effects.append(ie)
    total_effects.append(te)
    sig_flags_dir.append(get_stars(p_val_X[i]))
    sig_flags_ind.append(get_stars(p_val_WX[i]))
# ================= 7. 输出表 =================
print("\n==================================================================================")
print("太原市充电设施空间失配 SDM 模型归因结果 —— 【偏微分效应分解矩阵表】")
print("==================================================================================")
print(f"{'变量名称':<12} | {'直接效应 (Direct)':<18} | {'溢出效应 (Indirect)':<18} | {'总效应 (Total)':<15}")
print("-" * 82)
for i, name in enumerate(x_names):
    var_clean = name.replace("ln_", "").replace("_", " ")
    print(
        f"{var_clean:<14} | {direct_effects[i]:>10.4f} {sig_flags_dir[i]}       | {indirect_effects[i]:>10.4f} {sig_flags_ind[i]}       | {total_effects[i]:>10.4f}")
print("==================================================================================")
print(" 注: *** p<0.01, ** p<0.05, * p<0.1。")
# 导出三线表 CSV
results_df = pd.DataFrame({
    '解释变量': [n.replace("ln_", "") for n in x_names],
    'VIF共线性': vif_data['VIF'].values.round(2),
    '直接效应': direct_effects,
    '直接效应_显著': [s.strip() for s in sig_flags_dir],
    '溢出效应': indirect_effects,
    '溢出效应_显著': [s.strip() for s in sig_flags_ind],
    '总效应': total_effects
})
results_df.to_csv("太原市_SDM结果.csv", index=False, encoding='utf-8-sig')
