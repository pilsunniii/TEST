import math
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 1. 시스템 설정 및 의약품 DB 정의
# ==========================================
st.set_page_config(
    page_title="BioPharm-Sim Pro | 임상 약동학 통합 시뮬레이터",
    page_icon="🧬",
    layout="wide",
)

# 고급 약물 DB (반감기, 권장 용량, 독성 농도 기준 포함)
MEDICINE_DB = {
    "아세트아미노펜 (타이레놀)": {
        "half_life": 2.5,
        "standard_dose": 500,
        "mec": 150,  # 최소 유효 농도
        "mtc": 800,  # 최소 독성 농도
    },
    "이부프로펜 (이지엔6)": {
        "half_life": 2.0,
        "standard_dose": 400,
        "mec": 100,
        "mtc": 600,
    },
    "아스피린 (항혈전/소염)": {
        "half_life": 3.0,
        "standard_dose": 100,
        "mec": 50,
        "mtc": 300,
    },
    "테오필린 (천식치료제-좁은 치료창)": {
        "half_life": 8.0,
        "standard_dose": 200,
        "mec": 100,
        "mtc": 250,
    },
}

# 헤더 UI
st.title("🧬 BioPharm-Sim Pro: 임상 약동학 & 복약 순응도 통합 시뮬레이터")
st.caption(
    "의학·생명공학 수행평가 프로젝트 | 1차 소실 반응 모델 및 몬테카를로 임상 확률 시뮬레이션 연동"
)
st.markdown("---")


# ==========================================
# 2. 예외 처리 검증 함수 (평가 요소 ⑥)
# ==========================================
def validate_clinical_inputs(dose, interval, times, age, weight):
    """입력 데이터의 생체 유효 범위 검증"""
    if dose <= 0 or interval <= 0 or times < 1:
        raise ValueError(
            "투약량, 투약 간격, 복용 횟수는 반드시 양수여야 합니다."
        )
    if age < 0 or age > 120:
        raise ValueError("환자 연령은 0세 이상 120세 이하로 입력해야 합니다.")
    if weight < 5 or weight > 250:
        raise ValueError("환자 체중은 5kg 이상 250kg 이하로 입력해야 합니다.")


# ==========================================
# 3. 사이드바 - 고급 환자/약물 데이터 입력
# ==========================================
st.sidebar.header("👤 환자 임상 데이터")
user_age = st.sidebar.number_input("환자 연령 (세)", 1, 120, 45)
user_weight = st.sidebar.number_input("환자 체중 (kg)", 5, 200, 65)

st.sidebar.subheader("🏥 병력 및 생체 지표")
is_renal_impaired = st.sidebar.checkbox(
    "신장 기능 저하 (eGFR < 60 mL/min)"
)
is_hepatic_impaired = st.sidebar.checkbox(
    "간 기능 장애 (Child-Pugh B 이상)"
)

st.sidebar.markdown("---")
st.sidebar.header("💊 처방 정보 설정")
primary_med = st.sidebar.selectbox("주 약물 선택", list(MEDICINE_DB.keys()))

# 환자 생체 지표 기반 반감기 자동 보정 계산 (생명공학 알고리즘)
base_half_life = MEDICINE_DB[primary_med]["half_life"]
adjusted_half_life = base_half_life

if is_renal_impaired:
    adjusted_half_life *= 1.5  # 신장 배출 지연
if is_hepatic_impaired:
    adjusted_half_life *= 1.3  # 간 대사 지연

dose_mg = st.sidebar.number_input(
    "1회 투여량 (mg)", 10, 2000, MEDICINE_DB[primary_med]["standard_dose"]
)
interval_hrs = st.sidebar.number_input("투여 간격 (시간)", 1, 24, 8)
total_doses = st.sidebar.number_input("총 투여 횟수 (회)", 1, 14, 6)

# ==========================================
# 4. 연산 알고리즘 및 시뮬레이션 구현
# ==========================================
try:
    # ① 입력 예외 처리 검증 실행
    validate_clinical_inputs(
        dose_mg, interval_hrs, total_doses, user_age, user_weight
    )

    # 약동학 기본 상수 계산
    k = math.log(2) / adjusted_half_life  # 소실 속도 상수
    mec_val = MEDICINE_DB[primary_med]["mec"]
    mtc_val = MEDICINE_DB[primary_med]["mtc"]

    # 시간 타임라인 (0.1시간 단위)
    total_time = (total_doses * interval_hrs) + 24
    t_timeline = np.arange(0, total_time, 0.1)

    # [알고리즘 A] 정석 복약 시 체내 농도 추이
    ideal_conc = np.zeros_like(t_timeline)
    for i in range(int(total_doses)):
        dose_time = i * interval_hrs
        mask = t_timeline >= dose_time
        ideal_conc[mask] += dose_mg * np.exp(-k * (t_timeline[mask] - dose_time))

    # [알고리즘 B] 몬테카를로 복약 지연/누락 시뮬레이션 (80% 준수율 가정)
    random.seed(42)  # 재현성 보장
    real_conc = np.zeros_like(t_timeline)
    actual_dose_times = []

    for i in range(int(total_doses)):
        # 20% 확률로 약을 잊거나 지연 복용
        if random.random() > 0.2:  # 제때 복용
            dose_time = (i * interval_hrs) + random.uniform(-0.5, 0.5)
            dose_time = max(0, dose_time)
            actual_dose_times.append(dose_time)
            mask = t_timeline >= dose_time
            real_conc[mask] += dose_mg * np.exp(
                -k * (t_timeline[mask] - dose_time)
            )

    # ==========================================
    # 5. 대시보드 화면 구성 (Streamlit 고급 레이아웃)
    # ==========================================

    # (1) 상단 요약 KPI 카운터
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("보정된 반감기", f"{adjusted_half_life:.2f} 시간")
    kpi2.metric("소실 속도 상수(k)", f"{k:.4f} /h")
    kpi3.metric("최고 예측 농도", f"{np.max(ideal_conc):.1f} mg")
    kpi4.metric(
        "치료적 유효 구간",
        f"{mec_val} ~ {mtc_val} mg",
        delta="Therapeutic Window",
    )

    st.markdown("---")

    # (2) 탭 구성 (시각화 / 위험 분석 / 상세 데이터)
    tab1, tab2, tab3 = st.tabs([
        "📈 혈중 농도 시뮬레이션",
        "⚠️ 독성 및 병용 위험 분석",
        "📄 상세 데이터 리포트",
    ])

    with tab1:
        st.subheader("체내 약물 잔류량 및 치료창 시각화")

        # Plotly 차트 생성
        fig = go.Figure()

        # 치료적 창 (Therapeutic Window) 영역 표시
        fig.add_hrect(
            y0=mec_val,
            y1=mtc_val,
            fillcolor="green",
            opacity=0.15,
            line_width=0,
            annotation_text="치료적 유효 범위 (Therapeutic Range)",
        )

        # 독성 기준선
        fig.add_hline(
            y=mtc_val,
            line_dash="dash",
            line_color="red",
            annotation_text="최소 독성 농도 (MTC)",
        )

        # 정석 복약 곡선
        fig.add_trace(
            go.Scatter(
                x=t_timeline,
                y=ideal_conc,
                mode="lines",
                name="100% 정석 복약 시",
                line=dict(color="blue", width=3),
            )
        )

        # 몬테카를로 현실 복약 곡선
        fig.add_trace(
            go.Scatter(
                x=t_timeline,
                y=real_conc,
                mode="lines",
                name="실제 복약 순응도 반영 (오차 발생)",
                line=dict(color="orange", width=2, dash="dot"),
            )
        )

        fig.update_layout(
            title=f"<b>[{primary_med}] 시간에 따른 혈중 농도 추이 분석</b>",
            xaxis_title="시간 (Hours)",
            yaxis_title="혈중 약물 농도 (mg)",
            hovermode="x unified",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🚨 Clinical Safety Warning & 약물 상호작용 분석")

        col_a, col_b = st.columns(2)

        with col_a:
            st.write("#### 1. 과다 투여 및 독성 위험성 검증")
            max_c = np.max(ideal_conc)
            if max_c > mtc_val:
                st.error(
                    f"⛔ **경고: 독성 농도 초과!** (최대 농도: {max_c:.1f}mg > 독성 기준: {mtc_val}mg)\n\n"
                    "투여 간격을 늘리거나 1회 투여량을 감량해야 합니다."
                )
            elif max_c < mec_val:
                st.warning(
                    f"⚠️ **경고: 약효 미달!** (최대 농도: {max_c:.1f}mg < 최소 유효 기준: {mec_val}mg)\n\n"
                    "약물 혈중 농도가 치료 범위에 도달하지 못합니다."
                )
            else:
                st.success(
                    "✅ **안전:** 혈중 최고 농도가 안전한 치료적 유효 범위 내에 존재합니다."
                )

        with col_b:
            st.write("#### 2. 약물 상호작용 (Drug-Drug Interaction) 체크")
            secondary_med = st.selectbox(
                "함께 복용 중인 다른 약물 선택",
                ["없음", "아스피린 (항혈전제)", "이부프로펜 (소염진통제)"],
            )

            if primary_med.startswith("이부프로펜") and secondary_med.startswith(
                "아스피린"
            ):
                st.error(
                    "🚨 **중대 약물 상호작용 위험!**\n\n"
                    "이부프로펜과 아스피린을 동시 복용 시 위장관 출혈 위험이 격증하며 아스피린의 항혈전 효과가 저하될 수 있습니다."
                )
            elif secondary_med != "없음":
                st.info(
                    "ℹ️ 선택한 약물 조합에서는 알려진 병용 금기 상호작용이 감지되지 않았습니다."
                )

    with tab3:
        st.subheader("📊 시뮬레이션 타임라인 데이터")
        df_export = pd.DataFrame({
            "시간(h)": np.round(t_timeline, 1),
            "정석_혈중농도(mg)": np.round(ideal_conc, 2),
            "현실반영_혈중농도(mg)": np.round(real_conc, 2),
        })
        st.dataframe(df_export.iloc[::10], use_container_width=True)

except ValueError as val_err:
    st.error(f"❌ 입력값 오류: {val_err}")
except Exception as sys_err:
    st.error(f"❌ 예상치 못한 시스템 오류 발생: {sys_err}")
