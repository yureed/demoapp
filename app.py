import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Analytics Demo | Data Science Dojo",
    page_icon="https://datasciencedojo.com/wp-content/uploads/dsd-favicon-80x80.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DATA SCIENCE DOJO BRAND COLORS
# =============================================================================
DSD_NAVY = "#0A1628"
DSD_BLUE = "#1E88E5"
DSD_LIGHT_BLUE = "#4FC3F7"
DSD_WHITE = "#FFFFFF"
DSD_GRAY = "#F5F5F5"
DSD_TEXT = "#212121"
DSD_TEXT_SECONDARY = "#616161"
CHART_COLORS = [DSD_BLUE, DSD_LIGHT_BLUE, "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]

# =============================================================================
# CUSTOM CSS - LIGHT BACKGROUND, DARK TEXT, DSD COLORS
# =============================================================================
st.markdown("""
<style>
/* Aggressive Streamlit icon fix - paste this whole block and reload (Ctrl/Cmd+F5) */

/* keep main background light */
.main, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
  background-color: #ffffff !important;
}

/* keep normal text dark (target text nodes only) */
.main, .block-container, [data-testid="stAppViewContainer"] { color: #1a1a2e !important; }
.main p, .main span, .main li, .main a, .main label,
.main input, .main button, .main textarea,
.main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
  color: #1a1a2e !important;
}

/* ========== AGGRESSIVE ICON OVERRIDES ========== */
/* Target header / top-right toolbar icons used by Streamlit (GitHub, Share, etc.) */
[data-testid="stHeader"] svg,
[data-testid="stHeader"] svg * ,
[data-testid="stAppViewContainer"] svg.header-icon,
[data-testid="stAppViewContainer"] svg[role="img"],
[data-testid="stAppViewContainer"] svg[role="img"] * {
  fill: #1a1a2e !important;
  color: #1a1a2e !important;
  stroke: #1a1a2e !important;
  opacity: 1 !important;
  -webkit-text-fill-color: #1a1a2e !important;
}

/* Force any <path>, <circle>, <rect>, <polygon> inside header/toolbar to dark */
[data-testid="stHeader"] path,
[data-testid="stHeader"] circle,
[data-testid="stHeader"] rect,
[data-testid="stHeader"] polygon,
[data-testid="stAppViewContainer"] path,
[data-testid="stAppViewContainer"] circle,
[data-testid="stAppViewContainer"] rect,
[data-testid="stAppViewContainer"] polygon {
  fill: #1a1a2e !important;
  stroke: #1a1a2e !important;
  color: #1a1a2e !important;
  opacity: 1 !important;
}

/* If SVGs have inline `fill` attributes, override them too */
svg[fill], svg[fill] * { fill: #1a1a2e !important; }

/* Buttons that include icons (share, GitHub button icons, etc.) */
button svg, button svg *, .stButton button svg, .stButton button svg * {
  fill: #1a1a2e !important;
  stroke: #1a1a2e !important;
  color: #1a1a2e !important;
  opacity: 1 !important;
}

/* Some icons are inside divs with role='button' */
div[role="button"] svg, div[role="button"] svg * {
  fill: #1a1a2e !important;
  stroke: #1a1a2e !important;
}

/* Avoid touching sidebar icons/text (keep them white) */
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
  background: linear-gradient(180deg, #0a1628 0%, #0d2137 100%) !important;
}
section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] svg,
section[data-testid="stSidebar"] svg * {
  color: #ffffff !important;
  fill: #ffffff !important;
  stroke: #ffffff !important;
}

/* If a third-party stylesheet is setting SVGs via mask / background-image,
   this fallback gives contrast by inverting them inside header (last resort). */
[data-testid="stHeader"] img,
[data-testid="stHeader"] .css-1* img {
  filter: invert(0%) !important; /* makes images dark if they were inverted */
}

/* keep other UI pieces styled as desired (uploader & cards) */
.stFileUploader section { border: 2px dashed #1E88E5 !important; background: #F1F5F9 !important; border-radius: 12px !important; }
.stFileUploader button { background-color: #1E88E5 !important; color: #FFFFFF !important; }
.metric-card { background: #FFFFFF !important; border-radius: 16px; padding: 20px; border: 1px solid #E2E8F0; box-shadow: 0 4px 15px rgba(0,0,0,0.08); }
</style>


""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def create_metric_card(value, label):
    return f'<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>'

def format_number(num, is_count=False):
    if pd.isna(num): return "N/A"
    if is_count:
        return f"{int(num)}" if num < 1000 else (f"{num/1000:.1f}K" if num < 1_000_000 else f"{num/1_000_000:.1f}M")
    return f"{num:.2f}" if num < 1000 else (f"{num/1000:.2f}K" if num < 1_000_000 else f"{num/1_000_000:.2f}M")

def style_chart(fig, height=400):
    """Apply consistent light styling with dark text to plotly figures"""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,250,252,0.5)',
        font=dict(family="DM Sans, sans-serif", size=12, color="#1a1a2e"),
        legend=dict(font=dict(color="#1a1a2e")),
        height=height,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    # Ensure axis labels are dark
    fig.update_xaxes(
        title_font=dict(color="#1a1a2e"),
        tickfont=dict(color="#64748B"),
        gridcolor="#E2E8F0"
    )
    fig.update_yaxes(
        title_font=dict(color="#1a1a2e"),
        tickfont=dict(color="#64748B"),
        gridcolor="#E2E8F0"
    )
    # Hide colorbar title (prevents "undefined"), keep tick labels dark
    fig.update_coloraxes(
        colorbar_title_text="",
        colorbar_tickfont_color="#1a1a2e"
    )
    return fig

# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_superstore():
    df = pd.read_csv('data/Sample_-_Superstore.csv', encoding='latin-1')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=False, errors='coerce')
    df['Year'] = df['Order Date'].dt.year
    return df

@st.cache_data
def load_hr():
    df = pd.read_csv('data/WA_Fn-UseC_-HR-Employee-Attrition.csv', encoding='utf-8-sig')
    df.columns = df.columns.str.replace('ï»¿', '')
    return df

@st.cache_data
def load_marketing():
    df = pd.read_csv('data/marketing_campaign_dataset.csv', encoding='latin-1')
    df['Acquisition_Cost'] = df['Acquisition_Cost'].replace(r'[\$,]', '', regex=True).astype(float)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df['Year'] = df['Date'].dt.year
    return df

@st.cache_data
def load_ecommerce():
    df = pd.read_csv('data/E-commerce_Customer_Behavior_-_Sheet1.csv', encoding='latin-1')
    df['Satisfaction Level'] = df['Satisfaction Level'].fillna('Unknown')
    df['Age_Group'] = pd.cut(df['Age'], bins=[18,25,35,45,55,100], labels=['18-25','26-35','36-45','46-55','55+'])
    return df

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## Data Science Dojo")
    st.markdown("Analytics Demo")
    st.markdown("---")
    dashboard = st.selectbox("Select Dashboard", ["Upload Your Data", "Retail Sales", "HR Analytics", "Marketing", "E-commerce"])
    st.markdown("---")
    st.markdown("#### About")
    st.markdown("Interactive analytics demos showcasing data visualization capabilities.")

# =============================================================================
# UPLOAD YOUR DATA
# =============================================================================
def render_upload():
    st.markdown('<h1 class="dsd-header">Try With Your Data</h1>', unsafe_allow_html=True)
    st.markdown('<p class="dsd-subheader">Upload a CSV file to instantly explore your data with interactive visualizations</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is None:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### What You Can Do")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="feature-card"><h4>Automatic Analysis</h4><p>We detect your column types and suggest relevant visualizations automatically.</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="feature-card"><h4>Interactive Filtering</h4><p>Filter and drill down into your data with dynamic controls.</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="feature-card"><h4>Instant Insights</h4><p>Get summary statistics, distributions, and correlations in seconds.</p></div>', unsafe_allow_html=True)
        return
    
    # Load file with encoding detection
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    df = None
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=enc)
            break
        except: pass
    
    if df is None:
        st.error("Could not read file. Please ensure it's a valid CSV.")
        return
    
    # Detect date columns
    date_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(100)
            try:
                parsed = pd.to_datetime(sample, dayfirst=True, errors='coerce')
                if parsed.dropna().shape[0] > len(sample) * 0.5:
                    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
                    date_cols.append(col)
            except: pass
        elif 'datetime' in str(df[col].dtype):
            date_cols.append(col)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.select_dtypes(include=['object']).columns if c not in date_cols]
    
    # Data Overview
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Data Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(create_metric_card(format_number(len(df), True), "Rows"), unsafe_allow_html=True)
    with c2: st.markdown(create_metric_card(format_number(len(df.columns), True), "Columns"), unsafe_allow_html=True)
    with c3: st.markdown(create_metric_card(format_number(len(numeric_cols), True), "Numeric"), unsafe_allow_html=True)
    with c4: st.markdown(create_metric_card(format_number(len(date_cols), True), "Date"), unsafe_allow_html=True)
    
    fdf = df.copy()
    
    # Date filter
    if date_cols:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Date Range")
        dcol = date_cols[0] if len(date_cols) == 1 else st.selectbox("Date column", date_cols)
        valid = df[dcol].dropna()
        if len(valid) > 0:
            c1, c2 = st.columns(2)
            with c1: start = st.date_input("Start", value=valid.min(), min_value=valid.min(), max_value=valid.max())
            with c2: end = st.date_input("End", value=valid.max(), min_value=valid.min(), max_value=valid.max())
            fdf = fdf[(fdf[dcol] >= pd.to_datetime(start)) & (fdf[dcol] <= pd.to_datetime(end))]
    
    # Category filters
    if categorical_cols:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Filters")
        fcols = [c for c in categorical_cols if df[c].nunique() <= 50][:4]
        if fcols:
            cols = st.columns(len(fcols))
            for i, col in enumerate(fcols):
                with cols[i]:
                    val = st.selectbox(col, ['All'] + sorted(df[col].dropna().astype(str).unique().tolist()), key=f"f_{col}")
                    if val != 'All': fdf = fdf[fdf[col].astype(str) == val]
    
    st.caption(f"Showing {len(fdf):,} of {len(df):,} rows")
    
    # Tabs
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Explore Your Data")
    
    has_dates = len(date_cols) > 0 and len(numeric_cols) > 0
    
    if has_dates:
        t1, t2, t3, t4 = st.tabs(["Time Series", "Distribution", "Relationships", "Summary"])
    else:
        t2, t3, t4 = st.tabs(["Distribution", "Relationships", "Summary"])
        t1 = None
    
    # Time Series
    if t1 is not None:
        with t1:
            if numeric_cols:
                c1, c2 = st.columns([1, 3])
                with c1:
                    ts_date = st.selectbox("Date column", date_cols, key="ts_d")
                    ts_val = st.selectbox("Value", numeric_cols, key="ts_v")
                    ts_agg = st.selectbox("Aggregation", ["Sum", "Mean", "Count"], key="ts_a")
                    ts_per = st.selectbox("Period", ["Day", "Week", "Month", "Quarter", "Year"], index=2, key="ts_p")
                with c2:
                    tdf = fdf.dropna(subset=[ts_date])
                    if len(tdf) > 0:
                        pmap = {"Day": "D", "Week": "W", "Month": "M", "Quarter": "Q", "Year": "Y"}
                        tdf['P'] = tdf[ts_date].dt.to_period(pmap[ts_per])
                        amap = {"Sum": "sum", "Mean": "mean", "Count": "count"}
                        grp = tdf.groupby('P')[ts_val].agg(amap[ts_agg]).reset_index().sort_values('P')
                        
                        # Format period labels properly
                        def fmt_period(p, per):
                            if per == "Month": return p.strftime('%b %Y')
                            elif per == "Quarter": return f"Q{p.quarter} {p.year}"
                            elif per == "Year": return str(p.year)
                            elif per == "Week": return p.start_time.strftime('%d %b %Y')
                            else: return p.strftime('%d %b %Y')
                        
                        grp['Label'] = grp['P'].apply(lambda x: fmt_period(x, ts_per))
                        fig = go.Figure(go.Scatter(x=grp['Label'], y=grp[ts_val], mode='lines+markers', line=dict(color='#1E88E5', width=2), marker=dict(size=6)))
                        style_chart(fig)
                        fig.update_layout(title=f'{ts_agg} of {ts_val} by {ts_per}', xaxis=dict(type='category'), xaxis_title=ts_per, yaxis_title=ts_val)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No data available for the selected date range.")
            else:
                st.info("No numeric columns available for time series analysis.")
    
    # Distribution
    with t2:
        if numeric_cols:
            c1, c2 = st.columns([1, 3])
            with c1:
                ncol = st.selectbox("Column", numeric_cols, key="d_n")
                ccol = st.selectbox("Color by", ["None"] + categorical_cols, key="d_c") if categorical_cols else "None"
            with c2:
                fig = px.histogram(fdf, x=ncol, color=None if ccol == "None" else ccol, nbins=30, opacity=0.7, color_discrete_sequence=['#1E88E5', '#4FC3F7', '#FF9800', '#4CAF50', '#E91E63'], title=f'Distribution of {ncol}', labels={ncol: ncol, 'count': 'Count'})
                style_chart(fig)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available for distribution analysis.")
    
    # Relationships
    with t3:
        if len(numeric_cols) >= 2:
            c1, c2 = st.columns([1, 3])
            with c1:
                xc = st.selectbox("X-axis", numeric_cols, key="r_x")
                yc = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1), key="r_y")
                cc = st.selectbox("Color", ["None"] + categorical_cols, key="r_c") if categorical_cols else "None"
            with c2:
                fig = px.scatter(fdf, x=xc, y=yc, color=None if cc == "None" else cc, opacity=0.6, color_discrete_sequence=['#1E88E5', '#4FC3F7', '#FF9800', '#4CAF50', '#E91E63'], title=f'{yc} vs {xc}', labels={xc: xc, yc: yc})
                style_chart(fig)
                st.plotly_chart(fig, use_container_width=True)
        elif len(numeric_cols) == 1:
            st.info("Need at least 2 numeric columns for relationship analysis.")
        else:
            st.info("No numeric columns available.")
    
    # Summary
    with t4:
        if numeric_cols:
            st.markdown("#### Numeric Summary")
            st.dataframe(fdf[numeric_cols].describe().round(2), use_container_width=True)
        if categorical_cols:
            st.markdown("#### Category Counts")
            cat = st.selectbox("Category", categorical_cols, key="s_c")
            cts = fdf[cat].value_counts().head(10).reset_index()
            cts.columns = [cat, 'Count']
            fig = px.bar(cts, x=cat, y='Count', color_discrete_sequence=[DSD_BLUE], title=f'Top {cat} by Count', labels={cat: cat, 'Count': 'Count'})
            style_chart(fig, 350)
            st.plotly_chart(fig, use_container_width=True)
        if not numeric_cols and not categorical_cols:
            st.info("No columns available for summary.")
    
    # Drill-down
    if categorical_cols and numeric_cols:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Drill-Down Analysis")
        c1, c2, c3 = st.columns(3)
        with c1: gcol = st.selectbox("Group by", categorical_cols, key="dd_g")
        with c2: mcol = st.selectbox("Measure", numeric_cols, key="dd_m")
        with c3: agg = st.selectbox("Aggregation", ["Sum", "Mean", "Count"], key="dd_a")
        
        amap = {"Sum": "sum", "Mean": "mean", "Count": "count"}
        dd = fdf.groupby(gcol)[mcol].agg(amap[agg]).reset_index().sort_values(mcol, ascending=False).head(15)
        dd.columns = [gcol, f'{agg} of {mcol}']
        fig = px.bar(dd, x=gcol, y=f'{agg} of {mcol}', color_discrete_sequence=[DSD_BLUE], title=f'{agg} of {mcol} by {gcol}', labels={gcol: gcol, f'{agg} of {mcol}': f'{agg} of {mcol}'})
        style_chart(fig)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# RETAIL SALES
# =============================================================================
def render_retail():
    df = load_superstore()
    st.markdown('<h1 class="dsd-header">Retail Sales Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="dsd-subheader">Analyze sales performance across regions, categories, and time periods</p>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: region = st.selectbox("Region", ['All'] + sorted(df['Region'].unique().tolist()))
    with c2: category = st.selectbox("Category", ['All'] + sorted(df['Category'].unique().tolist()))
    with c3: segment = st.selectbox("Segment", ['All'] + sorted(df['Segment'].unique().tolist()))
    with c4: year = st.selectbox("Year", ['All'] + sorted(df['Year'].dropna().unique().tolist()))
    
    fdf = df.copy()
    if region != 'All': fdf = fdf[fdf['Region'] == region]
    if category != 'All': fdf = fdf[fdf['Category'] == category]
    if segment != 'All': fdf = fdf[fdf['Segment'] == segment]
    if year != 'All': fdf = fdf[fdf['Year'] == year]
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(create_metric_card(f"${format_number(fdf['Sales'].sum())}", "Total Sales"), unsafe_allow_html=True)
    with c2: st.markdown(create_metric_card(f"${format_number(fdf['Profit'].sum())}", "Total Profit"), unsafe_allow_html=True)
    with c3: st.markdown(create_metric_card(format_number(fdf['Order ID'].nunique(), True), "Orders"), unsafe_allow_html=True)
    with c4:
        margin = (fdf['Profit'].sum() / fdf['Sales'].sum() * 100) if fdf['Sales'].sum() > 0 else 0
        st.markdown(create_metric_card(f"{margin:.1f}%", "Margin"), unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### Sales Trend")
        m = fdf.groupby(fdf['Order Date'].dt.to_period('M')).agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
        m['Order Date'] = m['Order Date'].astype(str)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m['Order Date'], y=m['Sales'], name='Sales', line=dict(color='#1E88E5', width=2), mode='lines+markers'))
        fig.add_trace(go.Scatter(x=m['Order Date'], y=m['Profit'], name='Profit', line=dict(color='#4FC3F7', width=2), mode='lines+markers'))
        style_chart(fig, 350)
        fig.update_layout(legend=dict(orientation="h", y=1.02), yaxis_title="", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown("### Sales by Region")
        r = fdf.groupby('Region').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
        r['Margin'] = (r['Profit'] / r['Sales'] * 100).round(1)
        fig = px.bar(r, x='Region', y='Sales', color='Margin', color_continuous_scale='RdYlGn')
        style_chart(fig, 350)
        st.plotly_chart(fig, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Category Treemap")
        fig = px.treemap(fdf, path=['Category', 'Sub-Category'], values='Sales', color='Profit', color_continuous_scale='RdYlGn', color_continuous_midpoint=0, labels={'Profit': 'Profit $'})
        style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown("### Top Sub-Categories")
        sc = fdf.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=True).tail(10)
        fig = go.Figure(go.Bar(x=sc.values, y=sc.index, orientation='h', marker_color='#1E88E5'))
        style_chart(fig)
        fig.update_layout(yaxis_title="", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Geographic Drill-Down")
    c1, c2 = st.columns(2)
    with c1: dr = st.selectbox("Region", ['All Regions'] + sorted(fdf['Region'].unique().tolist()), key="rd_r")
    with c2:
        if dr != 'All Regions':
            ds = st.selectbox("State", ['All States'] + sorted(fdf[fdf['Region'] == dr]['State'].unique().tolist()), key="rd_s")
        else:
            ds = st.selectbox("State", ["Select Region first"], disabled=True, key="rd_s")
    
    if dr == 'All Regions':
        dd = fdf.groupby('Region').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
        xcol, title = 'Region', 'Sales by Region'
    elif ds in ['All States', 'Select Region first']:
        dd = fdf[fdf['Region'] == dr].groupby('State').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index().sort_values('Sales', ascending=False).head(15)
        xcol, title = 'State', f'Sales by State ({dr})'
    else:
        dd = fdf[(fdf['Region'] == dr) & (fdf['State'] == ds)].groupby('City').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index().sort_values('Sales', ascending=False).head(15)
        xcol, title = 'City', f'Sales by City ({ds})'
    
    dd['Margin'] = (dd['Profit'] / dd['Sales'] * 100).round(1)
    fig = px.bar(dd, x=xcol, y='Sales', color='Margin', color_continuous_scale='RdYlGn', title=title, labels={'Margin': 'Margin %'})
    style_chart(fig)
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# HR ANALYTICS
# =============================================================================
def render_hr():
    df = load_hr()
    st.markdown('<h1 class="dsd-header">HR Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="dsd-subheader">Understand employee attrition patterns and workforce dynamics</p>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1: dept = st.selectbox("Department", ['All'] + sorted(df['Department'].unique().tolist()))
    with c2: role = st.selectbox("Job Role", ['All'] + sorted(df['JobRole'].unique().tolist()))
    with c3: gender = st.selectbox("Gender", ['All'] + sorted(df['Gender'].unique().tolist()))
    
    fdf = df.copy()
    if dept != 'All': fdf = fdf[fdf['Department'] == dept]
    if role != 'All': fdf = fdf[fdf['JobRole'] == role]
    if gender != 'All': fdf = fdf[fdf['Gender'] == gender]
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    att = (fdf['Attrition'] == 'Yes').sum() / len(fdf) * 100 if len(fdf) > 0 else 0
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(create_metric_card(format_number(len(fdf), True), "Employees"), unsafe_allow_html=True)
    with c2: st.markdown(create_metric_card(f"{att:.1f}%", "Attrition"), unsafe_allow_html=True)
    with c3: st.markdown(create_metric_card(f"${format_number(fdf['MonthlyIncome'].mean())}", "Avg Salary"), unsafe_allow_html=True)
    with c4: st.markdown(create_metric_card(f"{fdf['YearsAtCompany'].mean():.1f}", "Avg Tenure"), unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### Attrition by Department")
        d = fdf.groupby(['Department', 'Attrition']).size().unstack(fill_value=0)
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Stayed', x=d.index, y=d.get('No', [0]*len(d)), marker_color='#4CAF50'))
        fig.add_trace(go.Bar(name='Left', x=d.index, y=d.get('Yes', [0]*len(d)), marker_color='#E53935'))
        style_chart(fig, 350)
        fig.update_layout(barmode='stack', yaxis_title="", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown("### Salary vs Attrition")
        fig = px.box(fdf, x='Attrition', y='MonthlyIncome', color='Attrition', color_discrete_map={'Yes': '#E53935', 'No': '#4CAF50'})
        style_chart(fig, 350)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Age Distribution")
        fig = px.histogram(fdf, x='Age', color='Attrition', nbins=20, color_discrete_map={'Yes': '#E53935', 'No': '#4CAF50'}, barmode='overlay', opacity=0.7)
        style_chart(fig, 350)
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown("### Satisfaction Impact")
        s = fdf.groupby('JobSatisfaction').apply(lambda x: (x['Attrition'] == 'Yes').sum() / len(x) * 100).reset_index()
        s.columns = ['Satisfaction', 'Rate']
        s['Satisfaction'] = s['Satisfaction'].map({1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'})
        fig = px.bar(s, x='Satisfaction', y='Rate', color='Rate', color_continuous_scale='RdYlGn_r', labels={'Rate': 'Attrition %'})
        style_chart(fig, 350)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Organizational Drill-Down")
    dd = st.selectbox("Department", ['All Departments'] + sorted(fdf['Department'].unique().tolist()), key="hr_dd")
    
    if dd == 'All Departments':
        d = fdf.groupby('Department').apply(lambda x: pd.Series({'Rate': (x['Attrition']=='Yes').sum()/len(x)*100, 'Salary': x['MonthlyIncome'].mean()})).reset_index()
        xcol, title = 'Department', 'Attrition by Department'
    else:
        d = fdf[fdf['Department'] == dd].groupby('JobRole').apply(lambda x: pd.Series({'Rate': (x['Attrition']=='Yes').sum()/len(x)*100, 'Salary': x['MonthlyIncome'].mean()})).reset_index()
        xcol, title = 'JobRole', f'Attrition by Role ({dd})'
    
    fig = px.bar(d, x=xcol, y='Rate', color='Salary', color_continuous_scale='Viridis', title=title, labels={'Rate': 'Attrition %', 'Salary': 'Avg Salary'})
    style_chart(fig)
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# MARKETING
# =============================================================================
def render_marketing():
    df = load_marketing()
    st.markdown('<h1 class="dsd-header">Marketing Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="dsd-subheader">Evaluate campaign performance across channels and audiences</p>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1: ch = st.selectbox("Channel", ['All'] + sorted(df['Channel_Used'].unique().tolist()))
    with c2: ct = st.selectbox("Campaign Type", ['All'] + sorted(df['Campaign_Type'].unique().tolist()))
    with c3: yr = st.selectbox("Year", ['All'] + sorted(df['Year'].dropna().unique().tolist()), key="m_y")
    
    fdf = df.copy()
    if ch != 'All': fdf = fdf[fdf['Channel_Used'] == ch]
    if ct != 'All': fdf = fdf[fdf['Campaign_Type'] == ct]
    if yr != 'All': fdf = fdf[fdf['Year'] == yr]
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(create_metric_card(format_number(len(fdf), True), "Campaigns"), unsafe_allow_html=True)
    with c2: st.markdown(create_metric_card(f"{fdf['ROI'].mean():.2f}x", "Avg ROI"), unsafe_allow_html=True)
    with c3: st.markdown(create_metric_card(f"{fdf['Conversion_Rate'].mean()*100:.1f}%", "Conversion"), unsafe_allow_html=True)
    with c4: st.markdown(create_metric_card(f"${format_number(fdf['Acquisition_Cost'].sum())}", "Ad Spend"), unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### ROI by Channel")
        d = fdf.groupby('Channel_Used').agg({'ROI': 'mean', 'Conversion_Rate': 'mean'}).reset_index()
        d['Conversion_Rate'] = (d['Conversion_Rate'] * 100).round(1)
        fig = px.bar(d, x='Channel_Used', y='ROI', color='Conversion_Rate', color_continuous_scale='Viridis', labels={'Conversion_Rate': 'Conv %'})
        style_chart(fig, 350)
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown("### Campaign Type Performance")
        d = fdf.groupby('Campaign_Type').agg({'ROI': 'mean', 'Engagement_Score': 'mean'}).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(name='ROI', x=d['Campaign_Type'], y=d['ROI'], marker_color='#1E88E5'))
        fig.add_trace(go.Bar(name='Engagement', x=d['Campaign_Type'], y=d['Engagement_Score'], marker_color='#4FC3F7'))
        style_chart(fig, 350)
        fig.update_layout(barmode='group', yaxis_title="", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Audience Performance")
        d = fdf.groupby('Target_Audience').agg({'ROI': 'mean', 'Conversion_Rate': 'mean', 'Acquisition_Cost': 'mean'}).reset_index()
        d['Conversion_Rate'] = (d['Conversion_Rate'] * 100).round(1)
        fig = px.scatter(d, x='Conversion_Rate', y='ROI', size='Acquisition_Cost', text='Target_Audience', color='Acquisition_Cost', color_continuous_scale='Viridis', labels={'Acquisition_Cost': 'Cost', 'Conversion_Rate': 'Conv %'})
        fig.update_traces(textposition='top center', textfont_size=9)
        style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown("### Monthly Trends")
        m = fdf.groupby(fdf['Date'].dt.to_period('M')).agg({'ROI': 'mean', 'Campaign_ID': 'count'}).reset_index()
        m['Date'] = m['Date'].astype(str)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=m['Date'], y=m['ROI'], name='ROI', line=dict(color='#1E88E5', width=2)), secondary_y=False)
        fig.add_trace(go.Bar(x=m['Date'], y=m['Campaign_ID'], name='Campaigns', marker_color='#4FC3F7', opacity=0.5), secondary_y=True)
        style_chart(fig)
        fig.update_yaxes(title_text="ROI", secondary_y=False)
        fig.update_yaxes(title_text="Campaigns", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Campaign Drill-Down")
    dc = st.selectbox("Channel", ['All Channels'] + sorted(fdf['Channel_Used'].unique().tolist()), key="m_dd")
    
    if dc == 'All Channels':
        d = fdf.groupby('Channel_Used').agg({'ROI': 'mean', 'Conversion_Rate': 'mean'}).reset_index()
        d['Conversion_Rate'] = (d['Conversion_Rate'] * 100).round(1)
        xcol, title = 'Channel_Used', 'ROI by Channel'
    else:
        d = fdf[fdf['Channel_Used'] == dc].groupby('Campaign_Type').agg({'ROI': 'mean', 'Conversion_Rate': 'mean'}).reset_index()
        d['Conversion_Rate'] = (d['Conversion_Rate'] * 100).round(1)
        xcol, title = 'Campaign_Type', f'ROI by Type ({dc})'
    
    fig = px.bar(d, x=xcol, y='ROI', color='Conversion_Rate', color_continuous_scale='Viridis', title=title, labels={'Conversion_Rate': 'Conv %'})
    style_chart(fig)
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# E-COMMERCE
# =============================================================================
def render_ecommerce():
    df = load_ecommerce()
    st.markdown('<h1 class="dsd-header">E-commerce Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="dsd-subheader">Understand customer behavior, satisfaction, and spending patterns</p>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1: city = st.selectbox("City", ['All'] + sorted(df['City'].unique().tolist()))
    with c2: mem = st.selectbox("Membership", ['All'] + sorted(df['Membership Type'].unique().tolist()))
    with c3: sat = st.selectbox("Satisfaction", ['All'] + sorted(df['Satisfaction Level'].dropna().unique().tolist()))
    
    fdf = df.copy()
    if city != 'All': fdf = fdf[fdf['City'] == city]
    if mem != 'All': fdf = fdf[fdf['Membership Type'] == mem]
    if sat != 'All': fdf = fdf[fdf['Satisfaction Level'] == sat]
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(create_metric_card(format_number(len(fdf), True), "Customers"), unsafe_allow_html=True)
    with c2: st.markdown(create_metric_card(f"${format_number(fdf['Total Spend'].sum())}", "Revenue"), unsafe_allow_html=True)
    with c3: st.markdown(create_metric_card(f"${fdf['Total Spend'].mean():.0f}", "Avg Spend"), unsafe_allow_html=True)
    with c4: st.markdown(create_metric_card(f"{fdf['Average Rating'].mean():.1f}", "Avg Rating"), unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### Revenue by Membership")
        d = fdf.groupby('Membership Type')['Total Spend'].sum().reset_index()
        fig = px.pie(d, values='Total Spend', names='Membership Type', color_discrete_sequence=[DSD_BLUE, DSD_LIGHT_BLUE, '#FF9800'], hole=0.4)
        style_chart(fig, 350)
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown("### Satisfaction Distribution")
        d = fdf['Satisfaction Level'].value_counts().reset_index()
        d.columns = ['Satisfaction', 'Count']
        colors = {'Satisfied': '#4CAF50', 'Neutral': '#FF9800', 'Unsatisfied': '#E53935', 'Unknown': '#9E9E9E'}
        fig = px.bar(d, x='Satisfaction', y='Count', color='Satisfaction', color_discrete_map=colors)
        style_chart(fig, 350)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Spend vs Items")
        fig = px.scatter(fdf, x='Items Purchased', y='Total Spend', color='Membership Type', size='Days Since Last Purchase', color_discrete_sequence=['#1E88E5', '#4FC3F7', '#FF9800', '#4CAF50', '#E91E63'])
        style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown("### Spend by Age Group")
        d = fdf.groupby('Age_Group')['Total Spend'].mean().reset_index()
        fig = px.bar(d, x='Age_Group', y='Total Spend', color='Total Spend', color_continuous_scale='Viridis', labels={'Total Spend': 'Avg Spend'})
        style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Customer Drill-Down")
    dc = st.selectbox("City", ['All Cities'] + sorted(fdf['City'].unique().tolist()), key="e_dd")
    
    if dc == 'All Cities':
        d = fdf.groupby('City').agg({'Total Spend': 'sum', 'Average Rating': 'mean'}).reset_index()
        xcol, title = 'City', 'Revenue by City'
    else:
        d = fdf[fdf['City'] == dc].groupby('Membership Type').agg({'Total Spend': 'sum', 'Average Rating': 'mean'}).reset_index()
        xcol, title = 'Membership Type', f'Revenue by Membership ({dc})'
    
    fig = px.bar(d, x=xcol, y='Total Spend', color='Average Rating', color_continuous_scale='Viridis', title=title, labels={'Average Rating': 'Rating'})
    style_chart(fig)
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# MAIN
# =============================================================================
if dashboard == "Upload Your Data": render_upload()
elif dashboard == "Retail Sales": render_retail()
elif dashboard == "HR Analytics": render_hr()
elif dashboard == "Marketing": render_marketing()
elif dashboard == "E-commerce": render_ecommerce()
