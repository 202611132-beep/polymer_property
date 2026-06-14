
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
import koreanize_matplotlib  # 한국어 폰트 자동 적용
 
# ── 페이지 설정
st.set_page_config(
    page_title="고분자 복합체 인장강도 예측",
    page_icon="🧪",
    layout="wide",
)
 
# ── 스타일
st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e8eaf0; }
 
    .hero {
        background: linear-gradient(135deg, #1a1f2e 0%, #0d1b2a 100%);
        border: 1px solid #2a3550;
        border-radius: 12px;
        padding: 2.4rem 2.8rem 2rem;
        margin-bottom: 2rem;
    }
    .hero h1 { font-size: 1.9rem; font-weight: 700; color: #e8eaf0; margin: 0 0 0.4rem; letter-spacing: -0.5px; }
    .hero p  { color: #8a9bb5; font-size: 0.95rem; margin: 0; }
 
    .input-card {
        background: #1a1f2e;
        border: 1px solid #2a3550;
        border-radius: 10px;
        padding: 1.4rem 1.6rem 0.6rem;
        margin-bottom: 1rem;
    }
    .input-card .label    { font-size: 0.75rem; font-weight: 600; color: #5a7fa8; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.3rem; }
    .input-card .var-name { font-size: 1.05rem; font-weight: 600; color: #c5d0e0; margin-bottom: 0.2rem; }
    .input-card .var-desc { font-size: 0.82rem; color: #6a7f99; margin-bottom: 0.6rem; }
 
    .derived-box {
        background: #12192a;
        border: 1px dashed #2a4060;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-top: 0.8rem;
    }
    .derived-box .d-title { font-size: 0.78rem; font-weight: 600; color: #4a6a88; letter-spacing: 0.07em; text-transform: uppercase; margin-bottom: 0.8rem; }
    .derived-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
    .derived-row .d-name { font-size: 0.85rem; color: #7a9ab8; }
    .derived-row .d-val  { font-size: 0.9rem; font-weight: 600; color: #4db8ff; }
 
    .result-box {
        background: linear-gradient(135deg, #0d1b2a 0%, #0a1520 100%);
        border: 1.5px solid #2e6ca4;
        border-radius: 12px;
        padding: 2rem 2.2rem;
        text-align: center;
    }
    .result-box .result-label { font-size: 0.8rem; font-weight: 600; letter-spacing: 0.1em; color: #4a7fa8; text-transform: uppercase; margin-bottom: 0.6rem; }
    .result-box .result-value { font-size: 3.4rem; font-weight: 800; color: #4db8ff; line-height: 1.1; }
    .result-box .result-unit  { font-size: 1.1rem; color: #6a9fc0; margin-top: 0.2rem; }
    .result-box .result-grade { display: inline-block; margin-top: 1rem; padding: 0.35rem 1.1rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
    .model-badge { display: inline-block; margin-top: 0.6rem; padding: 0.2rem 0.8rem; border-radius: 12px; font-size: 0.75rem; color: #5a8aaa; background: #0d2030; border: 1px solid #1e4060; }
 
    .insight-item {
        background: #151b28;
        border-left: 3px solid #2e6ca4;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.88rem;
        color: #a0b4c8;
    }
 
    hr { border-color: #2a3550 !important; }
    label { color: #8a9bb5 !important; }
    .stButton > button {
        background: linear-gradient(135deg, #1e5080, #2e6ca4);
        color: white; border: none; border-radius: 8px;
        padding: 0.65rem 2rem; font-size: 1rem; font-weight: 600;
        width: 100%; cursor: pointer; transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)
 
 
# ── 모델 로드
@st.cache_resource
def load_model():
    model      = joblib.load("polymer_best_model.pkl")
    scaler     = joblib.load("polymer_scaler.pkl")
    feat_all   = joblib.load("polymer_feature_cols.pkl")   # 파생변수 포함 7개
    feat_base  = joblib.load("polymer_base_cols.pkl")      # 사용자 입력 4개
    return model, scaler, feat_all, feat_base
 
try:
    model, scaler, feat_all, feat_base = load_model()
    model_loaded = True
    model_name = type(model).__name__
    # 사람이 읽기 좋은 이름으로 변환
    name_map = {
        "LinearRegression": "선형회귀",
        "Ridge": "릿지회귀",
        "RandomForestRegressor": "랜덤포레스트",
        "GradientBoostingRegressor": "그라디언트부스팅",
        "SVR": "SVR",
    }
    model_label = name_map.get(model_name, model_name)
except FileNotFoundError:
    model_loaded = False
    model_label = ""
 
 
# ── 헤더
st.markdown(f"""
<div class="hero">
    <h1>🧪 고분자 복합체 인장강도 예측</h1>
    <p>핵심 가동 조건 4개를 입력하면 파생 변수를 자동 계산하여 인장강도(MPa)를 예측합니다 &nbsp;·&nbsp; 채택 모델: <b>{"선형회귀  R² = 0.908" if model_loaded else "—"}</b></p>
</div>
""", unsafe_allow_html=True)
 
if not model_loaded:
    st.error("⚠️ 모델 파일(.pkl)을 찾을 수 없습니다. 같은 폴더에 네 개의 .pkl 파일을 업로드해 주세요.")
    st.stop()
 
 
# ── 변수 정의
VARS = [
    {
        "key": "filler",
        "label": "가동 조건 01",
        "name": "충전제 함량 (wt%)",
        "desc": "복합체에 첨가하는 충전제의 무게 비율. 높을수록 강도 상승 경향.",
        "col": "충전제_함량(wt%)",
        "min": 0.0, "max": 40.0, "default": 20.0, "step": 0.1,
        "importance": 38.3,
    },
    {
        "key": "interfacial",
        "label": "가동 조건 02",
        "name": "계면 결합 강도",
        "desc": "고분자 기지와 충전제 사이 계면의 접착 강도 (1 ~ 10 척도).",
        "col": "계면_결합_강도",
        "min": 1.0, "max": 10.0, "default": 5.5, "step": 0.1,
        "importance": 23.3,
    },
    {
        "key": "void",
        "label": "가동 조건 03",
        "name": "공극률 (%)",
        "desc": "복합체 내부 기공 비율. 낮을수록 인장강도에 유리.",
        "col": "공극률(%)",
        "min": 0.0, "max": 15.0, "default": 7.5, "step": 0.1,
        "importance": 16.9,
    },
    {
        "key": "dispersion",
        "label": "가동 조건 04",
        "name": "분산 균일도 (%)",
        "desc": "충전제가 기지 내에 얼마나 고르게 분포했는지의 비율.",
        "col": "분산_균일도",
        "min": 20.0, "max": 100.0, "default": 60.0, "step": 0.5,
        "importance": 14.9,
    },
]
 
st.caption("📊 5개 알고리즘 비교 후 최고 성능 모델 자동 선정 · 기본 4변수 + 파생 3변수 = 총 7변수 입력")
st.markdown("<hr>", unsafe_allow_html=True)
 
 
# ── 레이아웃
col_in, col_out = st.columns([1.1, 0.9], gap="large")
 
with col_in:
    st.markdown("### 가동 조건 입력")
    values = {}
    for v in VARS:
        st.markdown(f"""
        <div class="input-card">
            <div class="label">{v['label']} &nbsp; 중요도 {v['importance']}%</div>
            <div class="var-name">{v['name']}</div>
            <div class="var-desc">{v['desc']}</div>
        </div>
        """, unsafe_allow_html=True)
        values[v["key"]] = st.slider(
            v["name"],
            min_value=v["min"], max_value=v["max"],
            value=v["default"], step=v["step"],
            label_visibility="collapsed",
            key=v["key"],
        )
 
    # 파생 변수 실시간 표시
    f = values["filler"]; i = values["interfacial"]
    v_ = values["void"];  d = values["dispersion"]
    strength_idx = f * i
    defect_idx   = v_ / (d + 1)
    disp_bond    = d * i
 
    st.markdown(f"""
    <div class="derived-box">
        <div class="d-title">⚙️ 파생 변수 (자동 계산)</div>
        <div class="derived-row">
            <span class="d-name">강도 지수 = 충전제 함량 × 계면 결합 강도</span>
            <span class="d-val">{strength_idx:.2f}</span>
        </div>
        <div class="derived-row">
            <span class="d-name">결함 지수 = 공극률 ÷ (분산 균일도 + 1)</span>
            <span class="d-val">{defect_idx:.4f}</span>
        </div>
        <div class="derived-row">
            <span class="d-name">분산·결합 = 분산 균일도 × 계면 결합 강도</span>
            <span class="d-val">{disp_bond:.2f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("인장강도 예측하기", use_container_width=True)
 
 
with col_out:
    st.markdown("### 예측 결과")
 
    # 입력 → 파생 변수 포함 DataFrame → 스케일링 → 예측
    input_raw = pd.DataFrame(
        [[values["filler"], values["interfacial"], values["void"], values["dispersion"]]],
        columns=feat_base
    )
    input_df = input_raw.copy()
    input_df['강도_지수'] = input_df['충전제_함량(wt%)'] * input_df['계면_결합_강도']
    input_df['결함_지수'] = input_df['공극률(%)'] / (input_df['분산_균일도'] + 1)
    input_df['분산_결합'] = input_df['분산_균일도'] * input_df['계면_결합_강도']
    input_df = input_df[feat_all]
 
    input_scaled = scaler.transform(input_df)
    pred_val = max(0.0, model.predict(input_scaled)[0])
 
    # 등급 분류
    if pred_val >= 280:
        grade, grade_color, grade_bg = "고강도 ★★★", "#4db8ff", "#0d2a40"
    elif pred_val >= 180:
        grade, grade_color, grade_bg = "중강도 ★★☆", "#7ed48a", "#0d2a1a"
    else:
        grade, grade_color, grade_bg = "저강도 ★☆☆", "#f0a060", "#2a1a0d"
 
    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">예측 인장강도</div>
        <div class="result-value">{pred_val:.1f}</div>
        <div class="result-unit">MPa</div>
        <span class="result-grade" style="background:{grade_bg}; color:{grade_color}; border:1px solid {grade_color}40;">
            {grade}
        </span>
        <br>
        <span class="model-badge">채택 모델 : {model_label}</span>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # 인사이트
    st.markdown("#### 개선 인사이트")
    insights = []
    if values["filler"] < 25:
        insights.append(f"충전제 함량({values['filler']:.1f}%)을 25~35% 수준으로 높이면 강도 향상 기대")
    if values["interfacial"] < 7:
        insights.append(f"계면 결합 강도({values['interfacial']:.1f})가 낮습니다. 커플링제 등 표면 처리를 검토하세요")
    if values["void"] > 5:
        insights.append(f"공극률({values['void']:.1f}%)이 높습니다. 가공 압력 상승 또는 탈기 공정 최적화를 권장합니다")
    if values["dispersion"] < 60:
        insights.append(f"분산 균일도({values['dispersion']:.1f}%)가 낮습니다. 혼합 시간·속도 조정을 검토하세요")
    if not insights:
        insights.append("모든 조건이 양호한 범위입니다. 현재 설정으로 안정적인 강도를 기대할 수 있습니다 ✅")
 
    for ins in insights:
        st.markdown(f'<div class="insight-item">💡 {ins}</div>', unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # 입력값 현황 차트
    st.markdown("#### 입력값 현황")
    fig, ax = plt.subplots(figsize=(5, 2.4))
    fig.patch.set_facecolor("#1a1f2e")
    ax.set_facecolor("#1a1f2e")
 
    labels = ["충전제\n함량(%)", "계면\n결합강도", "공극률\n(%)", "분산\n균일도(%)"]
    norms  = [
        values["filler"] / 40,
        values["interfacial"] / 10,
        1 - values["void"] / 15,
        (values["dispersion"] - 20) / 80,
    ]
    colors = ["#4db8ff", "#7ed48a", "#f0a060", "#b48aff"]
    x = np.arange(len(labels))
    ax.bar(x, norms, color=colors, width=0.5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="#8a9bb5", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("정규화 값\n(1 = 최적)", color="#8a9bb5", fontsize=8)
    ax.tick_params(colors="#8a9bb5", labelsize=8)
    ax.axhline(0.7, color="#ffffff20", linestyle="--", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3550")
    plt.tight_layout(pad=0.5)
    st.pyplot(fig)
    plt.close()
 
 
# ── 하단
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"""
<p style="color:#4a5a72; font-size:0.8rem; text-align:center;">
  채택 모델: {model_label} (5개 알고리즘 비교 후 자동 선정) &nbsp;·&nbsp; 학습 데이터 12,700건 &nbsp;·&nbsp; 기본 4변수 + 파생 3변수 = 총 7변수 &nbsp;·&nbsp; R² 0.908
</p>
""", unsafe_allow_html=True)
 