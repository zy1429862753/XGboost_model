import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. 变量映射配置 (修复显示问题)
# ==========================================
# 修改点：在 ">" 前面加上反斜杠 "\>" 以防止被识别为 Markdown 引用
# 或者使用中文全角 "＞" 也是很好的替代方案

VAR_CONFIG = {
    # --- 分类变量 (Categorical) ---
    "BMI": {
        "<18.5 (Code: 0)": 0,
        "18.5-23.9 (Code: 1)": 1,
        "≥24 (Code: 2)": 2
    },
    "Duration_of_operation": {
        "≤60 min (Code: 0)": 0,
        r"\>60 min (Code: 1)": 1  # 修复：加了 \ 防止 > 被吞掉
    },
    "Duration_of_anesthesia": {
        "≤70 min (Code: 0)": 0,
        r"\>70 min (Code: 1)": 1  # 修复：加了 \ 防止 > 被吞掉
    },
    "Age": {
        "<60 (Code: 0)": 0,
        "≥60 (Code: 1)": 1
    },
    "Endoscopic_technique": {
        "ESD (Code: 0)": 0,
        "EFTR (Code: 1)": 1
    },
    "Sex": {
        "Female (Code: 0)": 0,
        "Male (Code: 1)": 1
    },

    # --- 连续变量 (Continuous) ---
}

# 连续变量的范围设置 [最小值, 最大值, 默认值, 步长]
SLIDER_SETTINGS = {
    "Operating_room_temperature": [18.0, 30.0, 22.0, 0.1],
    "Basal_body_temperature": [35.0, 42.0, 36.5, 0.1]
}

# ==========================================
# 2. 页面配置
# ==========================================
st.set_page_config(page_title="Hypothermia Prediction Tool", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 2rem !important;}
    .main-header {
        text-align: center; color: #333; margin-bottom: 20px; 
        font-weight: 700; font-size: 28px;
    }
    .custom-label {
        font-size: 16px !important; font-weight: 600; 
        color: #444; margin-top: 15px; margin-bottom: 5px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #f8f9fa; border: 1px solid #ddd; border-radius: 8px; padding: 15px;
    }
    div.stButton > button {
        background-color: #2ca02c; color: white; font-size: 18px; 
        height: 3em; border-radius: 8px; width: 100%; font-weight: bold;
    }
    /* 滑块颜色 */
    div.stSlider > div[data-baseweb = "slider"] > div > div > div > div {
        background-color: #2ca02c !important;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. 加载模型
# ==========================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("xgboost_model_deploy.pkl")
        features = joblib.load("feature_names_xgboost.pkl")
        return model, features
    except Exception as e:
        return None, []


model, feature_names = load_model()

# ==========================================
# 4. 界面逻辑
# ==========================================
st.markdown("<div class='main-header'>Inadvertent Intraoperative Hypothermia (IIH) Risk Prediction</div>",
            unsafe_allow_html=True)

user_input_values = {}

if not model:
    st.error("⚠️ 未找到模型文件，请先运行 generate_xgboost_model.py")
else:
    # 修改：将左侧列宽从 1.5 改为 2，给文字更多显示空间
    col_input, col_result = st.columns([2, 1], gap="large")

    # --- 左侧：输入控件 ---
    with col_input:
        with st.container(border=True):
            st.markdown("### Patient Parameters")

            # 使用两列布局
            cols = st.columns(2)

            for idx, feature in enumerate(feature_names):
                current_col = cols[idx % 2]

                with current_col:
                    st.markdown(f"<div class='custom-label'>{feature}</div>", unsafe_allow_html=True)

                    # 1. 检查是否为分类变量 (Radio)
                    if feature in VAR_CONFIG:
                        options_map = VAR_CONFIG[feature]
                        options_labels = list(options_map.keys())

                        selected_label = st.radio(
                            label=f"radio_{feature}",
                            options=options_labels,
                            key=feature,
                            label_visibility="collapsed",
                            horizontal=True
                        )
                        user_input_values[feature] = options_map[selected_label]

                    # 2. 否则视为连续变量 (Slider)
                    else:
                        settings = SLIDER_SETTINGS.get(feature, [0.0, 100.0, 0.0, 1.0])
                        min_v, max_v, def_v, step_v = settings

                        val = st.slider(
                            label=f"slider_{feature}",
                            min_value=float(min_v),
                            max_value=float(max_v),
                            value=float(def_v),
                            step=float(step_v),
                            key=feature,
                            label_visibility="collapsed"
                        )
                        user_input_values[feature] = val

    # --- 右侧：预测结果 ---
    with col_result:
        # 空白占位，让结果框稍微下移，对齐左侧
        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("### Prediction Result")

            chart_ph = st.empty()
            res_ph = st.empty()

            if st.button("🚀 Calculate Risk"):
                try:
                    # 构造输入数据
                    input_df = pd.DataFrame([user_input_values], columns=feature_names)

                    # 预测概率
                    pred_prob = model.predict_proba(input_df)[0][1]
                    risk_percent = pred_prob * 100

                    # 绘制仪表盘
                    # 修复：将 title 移出 Plotly，改用 layout 调整 margin
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=risk_percent,
                        number={'suffix': "%", 'font': {'size': 40, 'color': "#333"}},
                        # title={'text': "Hypothermia Probability"}, # 这一行注释掉，不用 Plotly 自带标题
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#2ca02c" if risk_percent < 50 else "#d62728"},  # 低风险绿，高风险红
                            'bgcolor': "white",
                            'steps': [
                                {'range': [0, 100], 'color': '#f0f2f6'}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 3},
                                'thickness': 0.75,
                                'value': risk_percent
                            }
                        }
                    ))

                    # 修复：增加 top margin (t=50) 避免顶部被遮挡
                    # 并且在这里手动加一个标题注释，或者直接依赖外面的 Markdown 标题
                    fig.update_layout(
                        height=220,
                        margin=dict(l=20, r=20, t=60, b=10),
                        title={
                            'text': "Probability",
                            'y': 0.9,
                            'x': 0.5,
                            'xanchor': 'center',
                            'yanchor': 'top',
                            'font': {'size': 20}
                        }
                    )
                    chart_ph.plotly_chart(fig, use_container_width=True)

                    # 文字提示
                    if risk_percent < 30:
                        res_ph.success(f"**Low Risk**: {risk_percent:.1f}%")
                    elif risk_percent < 70:
                        res_ph.warning(f"**Medium Risk**: {risk_percent:.1f}%")
                    else:
                        res_ph.error(f"**High Risk**: {risk_percent:.1f}%")

                except Exception as e:
                    st.error(f"Prediction Error: {str(e)}")
            else:
                chart_ph.info("Click 'Calculate Risk' to see the result.")