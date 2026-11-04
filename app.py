import math
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 1. 시스템 기본 설정 및 질병 카테고리별 약물 DB
# ==========================================
st.set_page_config(
    page_title="BioPharm-Sim Pro | 임상 약동학 다중 약물 시뮬레이터",
    page_icon="🧬",
    layout="wide",
)

# 카테고리별 상세 약물 DB (약물명, 표준 반감기(h), 1회 용량(mg), MEC(최소유효농도), MTC(최소독성농도), 주요 대사/배설 경로)
DISEASE_MEDICINE_DB = {
    "심혈관계 질환 (Cardiovascular)": {
        "아스피린 (항혈전제)": {
            "half_life": 3.0,
            "standard_dose": 100,
            "mec": 50,
            "mtc": 300,
            "elimination": "Renal/Hepatic",
        },
        "아테놀롤 (고혈압/베타차단제)": {
            "half_life": 6.0,
            "standard_dose": 50,
            "mec": 20,
            "mtc": 150,
            "elimination": "Renal",
        },
        "암로디핀 (고혈압/칼슘차단제)": {
            "half_life": 35.0,
            "standard_dose": 5,
            "mec": 2,
            "mtc": 15,
            "elimination": "Hepatic",
        },
        "아토르바스타틴 (고지혈증)": {
            "half_life": 14.0,
            "standard_dose": 20,
            "mec": 5,
            "mtc": 80,
            "elimination": "Hepatic",
        },
    },
    "감염성 질환 & 항생제 (Infectious Diseases)": {
        "아목시실린 (페니실린계 항생제)": {
            "half_life": 1.0,
            "standard_dose": 500,
            "mec": 100,
            "mtc": 1000,
            "elimination": "Renal",
        },
        "세파클러 (세파계 항생제)": {
            "half_life": 0.8,
            "standard_dose": 250,
            "mec": 50,
            "mtc": 600,
            "elimination": "Renal",
        },
        "아지트로마이신 (마크롤라이드계)": {
            "half_life": 40.0,
            "standard_dose": 250,
            "mec": 30,
            "mtc": 400,
            "elimination": "Hepatic",
        },
        "시프로플록사신 (퀴놀론계 항생제)": {
            "half_life": 4.0,
            "standard_dose": 500,
            "mec": 80,
            "mtc": 800,
            "elimination": "Renal/Hepatic",
        },
    },
    "내분비 & 대사 질환 (Endocrine & Metabolic)": {
        "메트포르민 (당뇨병)": {
            "half_life": 5.0,
            "standard_dose": 500,
            "mec": 150,
            "mtc": 1200,
            "elimination": "Renal",
        },
        "글리메피리드 (당뇨병)": {
            "half_life": 5.0,
            "standard_dose": 2,
            "mec": 0.5,
            "mtc": 6,
            "elimination": "Renal/Hepatic",
        },
        "레보티록신 (갑상선저하증)": {
            "half_life": 168.0,
            "standard_dose": 0.1,
            "mec": 0.03,
            "mtc": 0.2,
            "elimination": "Hepatic",
        },
    },
    "호흡기 & 소화기 질환 (Respiratory & GI)": {
        "테오필린 (천식치료제-좁은 치료창)": {
            "half_life": 8.0,
            "standard_dose": 200,
            "mec": 100,
            "mtc": 250,
            "elimination": "Hepatic",
        },
        "몬테루카스트 (알레르기/천식)": {
            "half_life": 4.5,
            "standard_dose": 10,
            "mec": 2,
            "mtc": 30,
            "elimination": "Hepatic",
        },
        "오메프라졸 (위산분비억제제)": {
            "half_life": 1.0,
            "standard_dose": 20,
            "mec": 5,
            "mtc": 60,
            "elimination": "Hepatic",
        },
    },
    "통증 & 근골격계 질환 (Pain & Inflammation)": {
        "아세트아미노펜 (해열진통제)": {
            "half_life": 2.5,
            "standard_dose": 500,
            "mec": 150,
            "mtc": 800,
            "elimination": "Hepatic",
        },
        "이부프로펜 (소염진통제)": {
            "half_life": 2.0,
            "standard_dose": 400,
            "mec": 100,
            "mtc": 600,
            "elimination": "Renal/Hepatic",
        },
        "트라마돌 (진통제)": {
            "half_life": 6.0,
            "standard_dose": 50,
            "mec": 15,
            "mtc": 120,
            "elimination": "Renal/Hepatic",
        },
    },
}

st.title(
    "🧬 BioPharm-Sim Pro v2.0: 질병군별 정밀 약동학 & 임상 지표 시뮬레이터"
)
st.caption(
    "의학·생명공학 수행평가 프로젝트 | Cockcroft-Gault 수식 기반 CrCl 신장 보정 및 임상 병력 반영 알고리즘"
)
st.markdown("---")


# ==========================================
# 2. 예외 처리 검증 함수 (평가 요소 ⑥)
# ==========================================
def validate_clinical_inputs(dose, interval, times, age, weight, scr):
    if dose <= 0 or interval <= 0 or times < 1:
        raise ValueError(
            "투약량, 투약 간격, 복용 횟수는 반드시 0보다 큰 양수여야 합니다."
        )
    if age < 1 or age > 120:
        raise ValueError("환자 연령은 1세 이상 120세 이하로 입력해야 합니다.")
    if weight < 5 or weight > 200:
        raise ValueError("환자 체중은 5kg 이상 200kg 이하로 입력해야 합니다.")
    if scr < 0.2 or scr > 15.0:
        raise ValueError(
            "혈청 크레아티닌 수치는 0.2 ~ 15.0 mg/dL 범위를 유지해야 합니다."
        )


# ==========================================
# 3. 사이드바 - 상세 생체 지표 및 병력 입력
# ==========================================
st.sidebar.header("👤 환자 기본 생체 지표")
gender = st.sidebar.radio("성별", ["남성 (Male)", "여성 (Female)"])
user_age = st.sidebar.number_input("연령 (세)", 1, 120, 55)
user_weight = st.sidebar.number_input("체중 (kg)", 5, 200, 68)

st.sidebar.subheader("🩺 임상 혈액 검사 및 병력")
serum_creatinine = st.sidebar.number_input(
    "혈청 크레아티닌 (Serum Cr, mg/dL)", 0.2, 10.0, 1.0, step=0.1
)
hepatic_status = st.sidebar.select_slider(
    "간 기능 평가 (Child-Pugh Class)",
    options=["Class A (정상/경증)", "Class B (중등도)", "Class C (중증 장애)"],
)

st.sidebar.markdown("---")
st.sidebar.header("💊 질병군 및 약물 처방 선택")
selected_category = st.sidebar.selectbox(
    "질병 카테고리 선택", list(DISEASE_MEDICINE_DB.keys())
)
selected_med_name = st.sidebar.selectbox(
    "약물 선택", list(DISEASE_MEDICINE_DB[selected_category].keys())
)

med_info = DISEASE_MEDICINE_DB[selected_category][selected_med_name]

dose_mg = st.sidebar.number_input(
    "1회 투여량 (mg)", 0.01, 3000.0, float(med_info["standard_dose"])
)
interval_hrs = st.sidebar.number_input("투여 간격 (시간)", 1, 48, 8)
total_doses = st.sidebar.number_input("총 투여 횟수 (회)", 1, 30, 5)


# ==========================================
# 4. 임상 알고리즘 (Cockcroft-Gault & 보정식)
# ==========================================
try:
    # ① 입력 예외 처리 검증
    validate_clinical_inputs(
        dose_mg,
        interval_hrs,
        total_doses,
        user_age,
        user_weight,
        serum_creatinine,
    )

    # [의학 수식] Cockcroft-Gault 공식을 이용한 크레아티닌 청소율(CrCl) 자동 산출
    crcl = ((140 - user_age) * user_weight) / (72 * serum_creatinine)
    if "여성" in gender:
        crcl *= 0.85  # 여성 보정 계수

    # 배설 경로별 약물 반감기 자동 보정 알고리즘
    base_half_life = med_info["half_life"]
    adjusted_half_life = base_half_life

    # 신장 배설 보정 (CrCl 정상 범위: 90~120 mL/min)
    if "Renal" in med_info["elimination"]:
        if crcl < 30:  # 중증 신장장애
            adjusted_half_life *= 2.5
        elif crcl < 60:  # 중등도 신장장애
            adjusted_half_life *= 1.6

    # 간 대사 보정
    if "Hepatic" in med_info["elimination"]:
        if hepatic_status == "Class B (중등도)":
            adjusted_half_life *= 1.4
        elif hepatic_status == "Class C (중증 장애)":
            adjusted_half_life *= 2.2

    # 약동학 기본 연산
    k = math.log(2) / adjusted_half_life
    mec_val = med_info["mec"]
    mtc_val = med_info["mtc"]

    # 타임라인 생성
    total_time = (total_doses * interval_hrs) + (adjusted_half_life * 3)
    t_timeline = np.arange(0, total_time, 0.1)

    # [알고리즘 A] 정석 복약 농도 계산
    ideal_conc = np.zeros_like(t_timeline)
    for i in range(int(total_doses)):
        dose_time = i * interval_hrs
        mask = t_timeline >= dose_time
        ideal_conc[mask] += dose_mg * np.exp(-k * (t_timeline[mask] - dose_time))

    # [알고리즘 B] 몬테카를로 복약 지연 시뮬레이션
    random.seed(42)
    real_conc = np.zeros_like(t_timeline)
    for i in range(int(total_doses)):
        if random.random() > 0.25:  # 75% 확률로 제때 복용
            dose_time = max(
                0, (i * interval_hrs) + random.uniform(-1.0, 1.5)
            )
            mask = t_timeline >= dose_time
            real_conc[mask] += dose_mg * np.exp(
                -k * (t_timeline[mask] - dose_time)
            )

    # ==========================================
    # 5. 메인 대시보드 출력
    # ==========================================

    # 상단 요약 카드
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("산출된 CrCl (신기능)", f"{crcl:.1f} mL/min")
    c2.metric(
        "보정된 반감기",
        f"{adjusted_half_life:.1f} h",
        delta=f"원래: {base_half_life}h",
    )
    c3.metric("소실 속도 상수(k)", f"{k:.4f} /h")
    c4.metric("치료적 창 (MEC~MTC)", f"{mec_val} ~ {mtc_val} mg")

    st.markdown("---")

    # 탭 구성
    tab1, tab2, tab3 = st.tabs([
        "📈 혈중 농도 추이 시뮬레이션",
        "🏥 임상 진단 및 위험 평가",
        "📄 시간대별 정밀 데이터",
    ])

    with tab1:
        st.subheader(f"[{selected_med_name}] 체내 약물 농도 추이")

        fig = go.Figure()
        # 치료적 창 Band
        fig.add_hrect(
            y0=mec_val,
            y1=mtc_val,
            fillcolor="green",
            opacity=0.15,
            line_width=0,
            annotation_text="치료적 유효 구간 (Therapeutic Window)",
        )
        fig.add_hline(
            y=mtc_val,
            line_dash="dash",
            line_color="red",
            annotation_text="최소 독성 농도 (MTC)",
        )
        fig.add_hline(
            y=mec_val,
            line_dash="dot",
            line_color="blue",
            annotation_text="최소 약효 농도 (MEC)",
        )

        fig.add_trace(
            go.Scatter(
                x=t_timeline,
                y=ideal_conc,
                mode="lines",
                name="정상 복약 스케줄",
                line=dict(color="blue", width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t_timeline,
                y=real_conc,
                mode="lines",
                name="복약 지연/오차 시뮬레이션",
                line=dict(color="orange", width=2, dash="dot"),
            )
        )

        fig.update_layout(
            xaxis_title="시간 (Hours)",
            yaxis_title="혈중 농도 (mg)",
            hovermode="x unified",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("📋 환자 맞춤형 임상 안전 진단")
        col_x, col_y = st.columns(2)

        with col_x:
            st.write("#### 1. 장기 기능 상태에 따른 위험성 평가")
            if crcl < 30 and "Renal" in med_info["elimination"]:
                st.error(
                    f"⛔ **중증 신장 기능 저하 (CrCl: {crcl:.1f} mL/min)**\n\n"
                    "약물 배출이 매우 지연되어 체내 축적 및 독성 위험이 큽니다. 투여 간격을 대폭 늘려야 합니다."
                )
            elif crcl < 60 and "Renal" in med_info["elimination"]:
                st.warning(
                    f"⚠️ **경증~중등도 신장 기능 저하 (CrCl: {crcl:.1f} mL/min)**\n\n"
                    "용량 감량 또는 투여 간격 조정을 고려하세요."
                )
            else:
                st.success("✅ **신장 기능 정상:** 표준 반감기가 적용됩니다.")

        with col_y:
            st.write("#### 2. 혈중 농도 안전성 판정")
            max_c = np.max(ideal_conc)
            if max_c > mtc_val:
                st.error(
                    f"🚨 **독성 농도 초과 위험!** (예측 최고 농도: {max_c:.1f}mg > MTC: {mtc_val}mg)"
                )
            elif max_c < mec_val:
                st.warning(
                    f"⚠️ **약효 미달!** (예측 최고 농도: {max_c:.1f}mg < MEC: {mec_val}mg)"
                )
            else:
                st.success(
                    "✅ 예측 최고 농도가 안전한 치료적 유효 범위 내에 유지됩니다."
                )

    with tab3:
        st.subheader("📊 상세 타임라인 연산 데이터")
        df_res = pd.DataFrame({
            "시간(h)": np.round(t_timeline, 1),
            "정상_혈중농도(mg)": np.round(ideal_conc, 2),
            "실제반영_혈중농도(mg)": np.round(real_conc, 2),
        })
        st.dataframe(df_res.iloc[::10], use_container_width=True)

except ValueError as e:
    st.error(f"❌ 입력값 유효성 에러: {e}")
except Exception as e:
    st.error(f"❌ 예기치 않은 시스템 에러: {e}")
