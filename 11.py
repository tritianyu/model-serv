import pandas as pd
import numpy as np
import re
from pathlib import Path

# -------- 可调参数 --------
excel_path = Path("/Users/cuitianyu/Desktop/工作簿2.xlsx")
sheet1_name = "Sheet1"
sheet2_name = "Sheet2"
sheet3_output = "Sheet3"

key_col = "[材料代号]"
compare_cols = ["[3天缺料]", "[7天缺料]", "[14天缺料]", "[30天缺料]"]

# 数值比较容差（需要严格相等可把 atol 设为 0）
ATOL = 1e-9
RTOL = 0.0
# -------------------------

def normalize_key(x):
    """将材料代号标准化为可匹配的字符串：
    - 去前后空格（含全角空格）
    - 全角数字/连字符转半角
    - 去掉逗号、内部空格
    - 去掉结尾的 .0/.00 等
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("\u3000", " ").strip()
    # 全角->半角
    trans = str.maketrans("０１２３４５６７８９－—", "0123456789--")
    s = s.translate(trans)
    # 去千分位逗号、内部空格
    s = s.replace(",", "").replace(" ", "")
    # 去掉末尾 .0/.00
    if re.fullmatch(r"\d+\.(0)+", s):
        s = s.split(".")[0]
    return s

def equal_series(a: pd.Series, b: pd.Series) -> pd.Series:
    """稳健相等性判断：
    - 若两边都能转成数值：用 np.isclose 比较
    - 否则：去空格的字符串比较
    - NaN 与 NaN 视为相等
    """
    a_num = pd.to_numeric(a, errors="coerce")
    b_num = pd.to_numeric(b, errors="coerce")
    both_num = a_num.notna() & b_num.notna()
    eq_num = np.isclose(a_num, b_num, rtol=RTOL, atol=ATOL)

    # 字符串比较（仅对非同时为数值的情况）
    a_str = a.astype(str).str.strip().where(~a.isna(), other=np.nan)
    b_str = b.astype(str).str.strip().where(~b.isna(), other=np.nan)
    eq_str = (a_str == b_str)

    both_nan = a.isna() & b.isna()
    return (both_num & eq_num) | (~both_num & eq_str) | both_nan

# 读取
with pd.ExcelFile(excel_path) as xls:
    df1 = pd.read_excel(xls, sheet_name=sheet1_name)
    df2 = pd.read_excel(xls, sheet_name=sheet2_name)

# 字段校验
need_cols = [key_col] + compare_cols
miss1 = [c for c in need_cols if c not in df1.columns]
miss2 = [c for c in need_cols if c not in df2.columns]
if miss1 or miss2:
    raise ValueError(f"字段缺失：sheet1缺{miss1}；sheet2缺{miss2}")

# 仅保留所需列，并规范材料代号
df1_use = df1[need_cols].copy()
df2_use = df2[need_cols].copy()
df1_use[key_col] = df1_use[key_col].map(normalize_key)
df2_use[key_col] = df2_use[key_col].map(normalize_key)

# 诊断：看看唯一键数量
print(f"[诊断] sheet1 唯一材料代号：{df1_use[key_col].nunique()}，sheet2 唯一材料代号：{df2_use[key_col].nunique()}")

# 以材料代号内连接（确保 dtype 一致）
m = pd.merge(df1_use, df2_use, on=key_col, how="inner", suffixes=("_sheet1", "_sheet2"))
print(f"[诊断] 成功匹配到的材料代号数量：{m[key_col].nunique()}；合并后行数：{len(m)}")

# 对比并生成标记
flag_cols = []
for col in compare_cols:
    c1, c2 = f"{col}_sheet1", f"{col}_sheet2"
    same_flag = equal_series(m[c1], m[c2])
    m[f"{col}_是否相同"] = same_flag
    flag_cols.append(f"{col}_是否相同")

# 汇总差异列
def diff_cols_row(row):
    diffs = []
    for col in compare_cols:
        if not bool(row[f"{col}_是否相同"]):
            diffs.append(col)
    return ",".join(diffs)

m["差异列"] = m.apply(diff_cols_row, axis=1)
m["存在差异"] = m["差异列"].str.len() > 0

# 输出仅差异行到 sheet3
out_cols = [key_col]
for c in compare_cols:
    out_cols += [f"{c}_sheet1", f"{c}_sheet2", f"{c}_是否相同"]
out_cols += ["差异列"]

out = m.loc[m["存在差异"], out_cols].copy()

with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    out.to_excel(writer, index=False, sheet_name=sheet3_output)

print(f"完成：已将存在差异的记录写入「{sheet3_output}」。共 {len(out)} 行。")

# ---- 可选：快速核查你提到的材料代号是否进入对比并被标记 ----
target = "3151000695"
target_norm = normalize_key(target)
if target_norm in set(m[key_col]):
    sub = m.loc[m[key_col] == target_norm, out_cols]
    print("[诊断] 3151000695 对比结果：")
    print(sub)
else:
    print("[诊断] 未在匹配结果中找到 3151000695。大概率是两表材料代号写法不同（空格/全角/逗号/结尾.0等），请检查源表此键的原始文本。")