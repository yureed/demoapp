import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="Data Analytics Demo", page_icon="◆", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap');
    .main { background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%); }
    .stApp { background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%); }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #ffffff !important; }
    .metric-card { background: linear-gradient(145deg, #1e1e4a 0%, #2a2a5a 100%); border-radius: 16px; padding: 20px; border: 1px solid rgba(255,255,255,0.1); }
    .metric-value { font-size: 2.5rem; font-weight: 700; color: #00d4ff; font-family: 'Space Grotesk', sans-serif; }
    .metric-label { font-size: 0.9rem; color: #a0a0c0; text-transform: uppercase; letter-spacing: 1px; margin-top: 8px; }
    .section-divider { height: 2px; background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.5), transparent); margin: 30px 0; }
    .capability-card { background: linear-gradient(145deg, #1e1e4a 0%, #2a2a5a 100%); border-radius: 12px; padding: 20px; border: 1px solid rgba(255,255,255,0.1); margin: 10px 0; }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #12122e 0%, #1a1a40 100%); }
    .dashboard-header { background: linear-gradient(90deg, #8b5cf6 0%, #06b6d4 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

def create_metric_card(value, label):
    return f'<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>'

def format_number(num, is_count=False):
    if is_count:
        return f"{int(num)}" if num < 1000 else (f"{num/1000:.1f}K" if num < 1000000 else f"{num/1000000:.1f}M")
    return f"{num:.2f}" if num < 1000 else (f"{num/1000:.2f}K" if num < 1000000 else f"{num/1000000:.2f}M")

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
    df['Acquisition_Cost'] = df['Acquisition_Cost'].replace('[\$,]', '', regex=True).astype(float)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df['Year'] = df['Date'].dt.year
    return df

@st.cache_data
def load_ecommerce():
    df = pd.read_csv('data/E-commerce_Customer_Behavior_-_Sheet1.csv', encoding='latin-1')
    df['Satisfaction Level'] = df['Satisfaction Level'].fillna('Unknown')
    df['Age_Group'] = pd.cut(df['Age'], bins=[18,25,35,45,55,100], labels=['18-25','26-35','36-45','46-55','55+'])
    return df

with st.sidebar:
    st.markdown("## Analytics Demo")
    dashboard = st.selectbox("Choose Dashboard", ["Upload Your Data", "Retail Sales", "HR Analytics", "Marketing", "E-commerce"])

# ============ UPLOAD YOUR DATA ============
def render_upload():
    st.markdown('<h1 class="dashboard-header">Try With Your Data</h1>', unsafe_allow_html=True)
    st.markdown("*Upload a CSV file and instantly explore your data*")
    
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file is None:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="capability-card"><h4 style="color:#8b5cf6">Automatic Analysis</h4><p style="color:#a0a0c0">Auto-detect column types and visualizations</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="capability-card"><h4 style="color:#06b6d4">Interactive Filtering</h4><p style="color:#a0a0c0">Filter and drill down dynamically</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="capability-card"><h4 style="color:#10b981">Instant Insights</h4><p style="color:#a0a0c0">Summary stats and distributions</p></div>', unsafe_allow_html=True)
        return
    
    # Try multiple encodings
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    df = None
    for encoding in encodings:
        try:
            uploaded_file.seek(0)  # Reset file pointer
            df = pd.read_csv(uploaded_file, encoding=encoding)
            break
        except (UnicodeDecodeError, Exception):
            continue
    
    if df is None:
        st.error("Could not read file. Please ensure it's a valid CSV.")
        return
    
    # Detect and convert date columns
    date_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to parse as date
            sample = df[col].dropna().head(100)
            try:
                parsed = pd.to_datetime(sample, infer_datetime_format=True)
                # Check if it looks like dates (not just numbers that converted)
                if parsed.dt.year.min() > 1900 and parsed.dt.year.max() < 2100:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    date_cols.append(col)
            except:
                pass
        elif 'datetime' in str(df[col].dtype):
            date_cols.append(col)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.select_dtypes(include=['object']).columns.tolist() if c not in date_cols]
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(create_metric_card(format_number(len(df), True), "Rows"), unsafe_allow_html=True)
    with col2: st.markdown(create_metric_card(format_number(len(df.columns), True), "Columns"), unsafe_allow_html=True)
    with col3: st.markdown(create_metric_card(format_number(len(numeric_cols), True), "Numeric"), unsafe_allow_html=True)
    with col4: st.markdown(create_metric_card(format_number(len(date_cols), True), "Date"), unsafe_allow_html=True)
    
    # Filters section
    filtered_df = df.copy()
    
    # Date filter (if date columns exist)
    if date_cols:
        st.markdown("### Date Range")
        date_col = st.selectbox("Date column", date_cols, key="date_col_select") if len(date_cols) > 1 else date_cols[0]
        
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        
        if pd.notna(min_date) and pd.notna(max_date):
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date, key="start_d")
            with col2:
                end_date = st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date, key="end_d")
            
            filtered_df = filtered_df[(filtered_df[date_col] >= pd.to_datetime(start_date)) & 
                                      (filtered_df[date_col] <= pd.to_datetime(end_date))]
    
    # Categorical filters
    if categorical_cols:
        st.markdown("### Filter Data")
        filters = {}
        filter_cols = [c for c in categorical_cols if df[c].nunique() <= 50]  # Only show filters for columns with <= 50 unique values
        
        if filter_cols:
            cols = st.columns(min(4, len(filter_cols)))
            for i, col in enumerate(filter_cols[:4]):
                with cols[i]:
                    filters[col] = st.selectbox(col, ['All'] + sorted(df[col].dropna().astype(str).unique().tolist()), key=f"f_{col}")
            
            for col, val in filters.items():
                if val != 'All': filtered_df = filtered_df[filtered_df[col].astype(str) == val]
    
    st.caption(f"Showing {len(filtered_df):,} of {len(df):,} rows")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Create tabs based on available column types
    if date_cols and numeric_cols:
        tab1, tab2, tab3, tab4 = st.tabs(["Time Series", "Distribution", "Relationships", "Summary"])
    else:
        tab1, tab2, tab3 = st.tabs(["Distribution", "Relationships", "Summary"])
        tab4 = None
    
    # Time Series tab (only if date columns exist)
    if date_cols and numeric_cols:
        with tab1:
            col1, col2 = st.columns([1, 3])
            with col1:
                ts_date_col = st.selectbox("Date column", date_cols, key="ts_date")
                ts_value_col = st.selectbox("Value column", numeric_cols, key="ts_value")
                ts_agg = st.selectbox("Aggregation", ["Sum", "Mean", "Count"], key="ts_agg")
                ts_period = st.selectbox("Group by", ["Day", "Week", "Month", "Quarter", "Year"], index=2, key="ts_period")
            
            with col2:
                # Aggregate by time period
                ts_df = filtered_df.copy()
                ts_df = ts_df.dropna(subset=[ts_date_col])
                
                if len(ts_df) > 0:
                    period_map = {"Day": "D", "Week": "W", "Month": "M", "Quarter": "Q", "Year": "Y"}
                    ts_df['Period'] = ts_df[ts_date_col].dt.to_period(period_map[ts_period])
                    
                    agg_map = {"Sum": "sum", "Mean": "mean", "Count": "count"}
                    ts_grouped = ts_df.groupby('Period')[ts_value_col].agg(agg_map[ts_agg]).reset_index()
                    
                    # Sort by period
                    ts_grouped = ts_grouped.sort_values('Period')
                    
                    # Format period labels nicely
                    def format_period(p, period_type):
                        if period_type == "Month":
                            return p.strftime('%b %Y')  # "Jan 2019"
                        elif period_type == "Quarter":
                            return f"Q{p.quarter} {p.year}"  # "Q1 2019"
                        elif period_type == "Year":
                            return str(p.year)  # "2019"
                        elif period_type == "Week":
                            return p.start_time.strftime('%d %b %Y')  # "01 Jan 2019"
                        else:  # Day
                            return p.strftime('%d %b %Y')  # "01 Jan 2019"
                    
                    ts_grouped['Period_Label'] = ts_grouped['Period'].apply(lambda x: format_period(x, ts_period))
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ts_grouped['Period_Label'], y=ts_grouped[ts_value_col], 
                                            mode='lines+markers', line=dict(color='#8b5cf6', width=3),
                                            hovertemplate='%{x}<br>Value: %{y:,.2f}<extra></extra>'))
                    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', 
                                     plot_bgcolor='rgba(0,0,0,0)', height=400,
                                     title=f'{ts_agg} of {ts_value_col} by {ts_period}',
                                     xaxis_title=ts_period, yaxis_title=ts_value_col,
                                     xaxis=dict(type='category'))  # Keep order, don't auto-sort
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data available for the selected filters.")
        
        # Distribution tab
        with tab2:
            if numeric_cols:
                col1, col2 = st.columns([1,3])
                with col1:
                    num_col = st.selectbox("Numeric column", numeric_cols)
                    color = st.selectbox("Color by", ["None"] + categorical_cols) if categorical_cols else "None"
                with col2:
                    fig = px.histogram(filtered_df, x=num_col, color=None if color=="None" else color, nbins=30, barmode='overlay', opacity=0.7)
                    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Relationships tab
        with tab3:
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns([1,3])
                with col1:
                    x = st.selectbox("X-axis", numeric_cols, key="sx")
                    y = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1), key="sy")
                    c = st.selectbox("Color", ["None"] + categorical_cols, key="sc") if categorical_cols else "None"
                with col2:
                    fig = px.scatter(filtered_df, x=x, y=y, color=None if c=="None" else c, opacity=0.6)
                    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Summary tab
        with tab4:
            if numeric_cols:
                st.dataframe(filtered_df[numeric_cols].describe().round(2), use_container_width=True)
    else:
        # No date columns - original layout
        with tab1:
            if numeric_cols:
                col1, col2 = st.columns([1,3])
                with col1:
                    num_col = st.selectbox("Numeric column", numeric_cols)
                    color = st.selectbox("Color by", ["None"] + categorical_cols) if categorical_cols else "None"
                with col2:
                    fig = px.histogram(filtered_df, x=num_col, color=None if color=="None" else color, nbins=30, barmode='overlay', opacity=0.7)
                    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns([1,3])
                with col1:
                    x = st.selectbox("X-axis", numeric_cols, key="sx")
                    y = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1), key="sy")
                    c = st.selectbox("Color", ["None"] + categorical_cols, key="sc") if categorical_cols else "None"
                with col2:
                    fig = px.scatter(filtered_df, x=x, y=y, color=None if c=="None" else c, opacity=0.6)
                    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            if numeric_cols:
                st.dataframe(filtered_df[numeric_cols].describe().round(2), use_container_width=True)
    
    if categorical_cols and numeric_cols:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Drill-Down Analysis")
        col1, col2, col3 = st.columns(3)
        with col1: group = st.selectbox("Group by", categorical_cols)
        with col2: measure = st.selectbox("Measure", numeric_cols)
        with col3: agg = st.selectbox("Aggregation", ["Sum", "Mean", "Count"])
        
        agg_map = {"Sum": "sum", "Mean": "mean", "Count": "count"}
        drill = filtered_df.groupby(group)[measure].agg(agg_map[agg]).reset_index()
        drill.columns = [group, f'{agg} of {measure}']
        drill = drill.sort_values(f'{agg} of {measure}', ascending=False).head(20)
        
        fig = px.bar(drill, x=group, y=f'{agg} of {measure}', color=f'{agg} of {measure}', color_continuous_scale='Viridis')
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

# ============ RETAIL SALES ============
def render_retail():
    df = load_superstore()
    st.markdown('<h1 class="dashboard-header">Retail Sales Analytics</h1>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: region = st.selectbox("Region", ['All'] + sorted(df['Region'].unique().tolist()))
    with col2: category = st.selectbox("Category", ['All'] + sorted(df['Category'].unique().tolist()))
    with col3: segment = st.selectbox("Segment", ['All'] + sorted(df['Segment'].unique().tolist()))
    with col4: year = st.selectbox("Year", ['All'] + sorted(df['Year'].unique().tolist()))
    
    fdf = df.copy()
    if region != 'All': fdf = fdf[fdf['Region'] == region]
    if category != 'All': fdf = fdf[fdf['Category'] == category]
    if segment != 'All': fdf = fdf[fdf['Segment'] == segment]
    if year != 'All': fdf = fdf[fdf['Year'] == year]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(create_metric_card(f"${format_number(fdf['Sales'].sum())}", "Sales"), unsafe_allow_html=True)
    with col2: st.markdown(create_metric_card(f"${format_number(fdf['Profit'].sum())}", "Profit"), unsafe_allow_html=True)
    with col3: st.markdown(create_metric_card(format_number(fdf['Order ID'].nunique(), True), "Orders"), unsafe_allow_html=True)
    with col4: st.markdown(create_metric_card(f"{(fdf['Profit'].sum()/fdf['Sales'].sum()*100):.2f}%", "Margin"), unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Sales Trend")
        m = fdf.groupby(fdf['Order Date'].dt.to_period('M')).agg({'Sales':'sum','Profit':'sum'}).reset_index()
        m['Order Date'] = m['Order Date'].astype(str)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m['Order Date'], y=m['Sales'], name='Sales', line=dict(color='#8b5cf6', width=3)))
        fig.add_trace(go.Scatter(x=m['Order Date'], y=m['Profit'], name='Profit', line=dict(color='#06b6d4', width=3)))
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### By Region")
        r = fdf.groupby('Region').agg({'Sales':'sum','Profit':'sum'}).reset_index()
        r['Margin'] = (r['Profit']/r['Sales']*100).round(2)
        fig = px.bar(r, x='Region', y='Sales', color='Margin', color_continuous_scale='RdYlGn')
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Category Treemap")
        fig = px.treemap(fdf, path=['Category','Sub-Category'], values='Sales', color='Profit', color_continuous_scale='RdYlGn', color_continuous_midpoint=0)
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Top Sub-Categories")
        s = fdf.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=True).tail(10)
        fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation='h', marker_color='#8b5cf6'))
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Geographic Drill-Down")
    col1, col2 = st.columns(2)
    with col1: dr = st.selectbox("Region", ['All Regions'] + sorted(fdf['Region'].unique().tolist()), key="dr")
    with col2:
        if dr != 'All Regions':
            ds = st.selectbox("State", ['All States'] + sorted(fdf[fdf['Region']==dr]['State'].unique().tolist()), key="ds")
        else:
            ds = st.selectbox("State", ["Select Region first"], disabled=True, key="ds")
    
    if dr == 'All Regions':
        dd = fdf.groupby('Region').agg({'Sales':'sum','Profit':'sum'}).reset_index()
        x, title = 'Region', 'Sales by Region'
    elif ds in ['All States', 'Select Region first']:
        dd = fdf[fdf['Region']==dr].groupby('State').agg({'Sales':'sum','Profit':'sum'}).reset_index().sort_values('Sales', ascending=False).head(15)
        x, title = 'State', f'Sales by State ({dr})'
    else:
        dd = fdf[(fdf['Region']==dr)&(fdf['State']==ds)].groupby('City').agg({'Sales':'sum','Profit':'sum'}).reset_index().sort_values('Sales', ascending=False).head(15)
        x, title = 'City', f'Sales by City ({ds})'
    
    dd['Margin'] = (dd['Profit']/dd['Sales']*100).round(2)
    fig = px.bar(dd, x=x, y='Sales', color='Margin', color_continuous_scale='RdYlGn', title=title)
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# ============ HR ANALYTICS ============
def render_hr():
    df = load_hr()
    st.markdown('<h1 class="dashboard-header">HR Analytics</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1: dept = st.selectbox("Department", ['All'] + sorted(df['Department'].unique().tolist()))
    with col2: role = st.selectbox("Job Role", ['All'] + sorted(df['JobRole'].unique().tolist()))
    with col3: gender = st.selectbox("Gender", ['All'] + sorted(df['Gender'].unique().tolist()))
    
    fdf = df.copy()
    if dept != 'All': fdf = fdf[fdf['Department'] == dept]
    if role != 'All': fdf = fdf[fdf['JobRole'] == role]
    if gender != 'All': fdf = fdf[fdf['Gender'] == gender]
    
    att_rate = (fdf['Attrition']=='Yes').sum() / len(fdf) * 100 if len(fdf) > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(create_metric_card(format_number(len(fdf), True), "Employees"), unsafe_allow_html=True)
    with col2: st.markdown(create_metric_card(f"{att_rate:.2f}%", "Attrition Rate"), unsafe_allow_html=True)
    with col3: st.markdown(create_metric_card(f"${format_number(fdf['MonthlyIncome'].mean())}", "Avg Salary"), unsafe_allow_html=True)
    with col4: st.markdown(create_metric_card(f"{fdf['YearsAtCompany'].mean():.2f}", "Avg Tenure"), unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Attrition by Department")
        d = fdf.groupby(['Department','Attrition']).size().unstack(fill_value=0)
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Stayed', x=d.index, y=d.get('No',[]), marker_color='#10b981'))
        fig.add_trace(go.Bar(name='Left', x=d.index, y=d.get('Yes',[]), marker_color='#ef4444'))
        fig.update_layout(barmode='stack', template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Salary vs Attrition")
        fig = px.box(fdf, x='Attrition', y='MonthlyIncome', color='Attrition', color_discrete_map={'Yes':'#ef4444','No':'#10b981'})
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Age Distribution")
        fig = px.histogram(fdf, x='Age', color='Attrition', nbins=20, color_discrete_map={'Yes':'#ef4444','No':'#10b981'}, barmode='overlay', opacity=0.7)
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Satisfaction Impact")
        s = fdf.groupby('JobSatisfaction').apply(lambda x: (x['Attrition']=='Yes').sum()/len(x)*100).reset_index()
        s.columns = ['Satisfaction','Attrition Rate']
        s['Satisfaction'] = s['Satisfaction'].map({1:'Low',2:'Medium',3:'High',4:'Very High'})
        fig = px.bar(s, x='Satisfaction', y='Attrition Rate', color='Attrition Rate', color_continuous_scale='RdYlGn_r')
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Organizational Drill-Down")
    dd = st.selectbox("Select Department", ['All Departments'] + sorted(fdf['Department'].unique().tolist()), key="hrd")
    
    if dd == 'All Departments':
        data = fdf.groupby('Department').apply(lambda x: pd.Series({'Rate': (x['Attrition']=='Yes').sum()/len(x)*100, 'Salary': x['MonthlyIncome'].mean()})).reset_index()
        x, title = 'Department', 'Attrition by Department'
    else:
        data = fdf[fdf['Department']==dd].groupby('JobRole').apply(lambda x: pd.Series({'Rate': (x['Attrition']=='Yes').sum()/len(x)*100, 'Salary': x['MonthlyIncome'].mean()})).reset_index()
        x, title = 'JobRole', f'Attrition by Role ({dd})'
    
    fig = px.bar(data, x=x, y='Rate', color='Salary', color_continuous_scale='Viridis', title=title)
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# ============ MARKETING ============
def render_marketing():
    df = load_marketing()
    st.markdown('<h1 class="dashboard-header">Marketing Analytics</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1: channel = st.selectbox("Channel", ['All'] + sorted(df['Channel_Used'].unique().tolist()))
    with col2: ctype = st.selectbox("Campaign Type", ['All'] + sorted(df['Campaign_Type'].unique().tolist()))
    with col3: year = st.selectbox("Year", ['All'] + sorted(df['Year'].unique().tolist()), key="my")
    
    fdf = df.copy()
    if channel != 'All': fdf = fdf[fdf['Channel_Used'] == channel]
    if ctype != 'All': fdf = fdf[fdf['Campaign_Type'] == ctype]
    if year != 'All': fdf = fdf[fdf['Year'] == year]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(create_metric_card(format_number(len(fdf), True), "Campaigns"), unsafe_allow_html=True)
    with col2: st.markdown(create_metric_card(f"{fdf['ROI'].mean():.2f}x", "Avg ROI"), unsafe_allow_html=True)
    with col3: st.markdown(create_metric_card(f"{fdf['Conversion_Rate'].mean()*100:.2f}%", "Conversion"), unsafe_allow_html=True)
    with col4: st.markdown(create_metric_card(f"${format_number(fdf['Acquisition_Cost'].sum())}", "Ad Spend"), unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ROI by Channel")
        c = fdf.groupby('Channel_Used').agg({'ROI':'mean','Conversion_Rate':'mean'}).reset_index()
        c['Conversion_Rate'] = (c['Conversion_Rate']*100).round(2)
        fig = px.bar(c, x='Channel_Used', y='ROI', color='Conversion_Rate', color_continuous_scale='Viridis')
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Campaign Type Performance")
        t = fdf.groupby('Campaign_Type').agg({'ROI':'mean','Engagement_Score':'mean'}).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(name='ROI', x=t['Campaign_Type'], y=t['ROI'], marker_color='#8b5cf6'))
        fig.add_trace(go.Bar(name='Engagement', x=t['Campaign_Type'], y=t['Engagement_Score'], marker_color='#06b6d4'))
        fig.update_layout(barmode='group', template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Audience Performance")
        a = fdf.groupby('Target_Audience').agg({'ROI':'mean','Conversion_Rate':'mean','Acquisition_Cost':'mean'}).reset_index()
        a['Conversion_Rate'] = (a['Conversion_Rate']*100).round(2)
        fig = px.scatter(a, x='Conversion_Rate', y='ROI', size='Acquisition_Cost', text='Target_Audience', color='Acquisition_Cost', color_continuous_scale='Viridis')
        fig.update_traces(textposition='top center', textfont_size=8)
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Monthly Trends")
        m = fdf.groupby(fdf['Date'].dt.to_period('M')).agg({'ROI':'mean','Campaign_ID':'count'}).reset_index()
        m['Date'] = m['Date'].astype(str)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=m['Date'], y=m['ROI'], name='ROI', line=dict(color='#8b5cf6', width=3)), secondary_y=False)
        fig.add_trace(go.Bar(x=m['Date'], y=m['Campaign_ID'], name='Campaigns', marker_color='rgba(6,182,212,0.5)'), secondary_y=True)
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Campaign Drill-Down")
    dc = st.selectbox("Select Channel", ['All Channels'] + sorted(fdf['Channel_Used'].unique().tolist()), key="mdc")
    
    if dc == 'All Channels':
        data = fdf.groupby('Channel_Used').agg({'ROI':'mean','Conversion_Rate':'mean'}).reset_index()
        data['Conversion_Rate'] = (data['Conversion_Rate']*100).round(2)
        x, title = 'Channel_Used', 'ROI by Channel'
    else:
        data = fdf[fdf['Channel_Used']==dc].groupby('Campaign_Type').agg({'ROI':'mean','Conversion_Rate':'mean'}).reset_index()
        data['Conversion_Rate'] = (data['Conversion_Rate']*100).round(2)
        x, title = 'Campaign_Type', f'ROI by Type ({dc})'
    
    fig = px.bar(data, x=x, y='ROI', color='Conversion_Rate', color_continuous_scale='Viridis', title=title)
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
    st.plotly_chart(fig, use_container_width=True)

# ============ E-COMMERCE ============
def render_ecommerce():
    df = load_ecommerce()
    st.markdown('<h1 class="dashboard-header">E-commerce Analytics</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1: city = st.selectbox("City", ['All'] + sorted(df['City'].unique().tolist()))
    with col2: membership = st.selectbox("Membership", ['All'] + sorted(df['Membership Type'].unique().tolist()))
    with col3: satisfaction = st.selectbox("Satisfaction", ['All'] + sorted(df['Satisfaction Level'].dropna().unique().tolist()))
    
    fdf = df.copy()
    if city != 'All': fdf = fdf[fdf['City'] == city]
    if membership != 'All': fdf = fdf[fdf['Membership Type'] == membership]
    if satisfaction != 'All': fdf = fdf[fdf['Satisfaction Level'] == satisfaction]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(create_metric_card(format_number(len(fdf), True), "Customers"), unsafe_allow_html=True)
    with col2: st.markdown(create_metric_card(f"${format_number(fdf['Total Spend'].sum())}", "Revenue"), unsafe_allow_html=True)
    with col3: st.markdown(create_metric_card(f"${fdf['Total Spend'].mean():.2f}", "Avg Spend"), unsafe_allow_html=True)
    with col4: st.markdown(create_metric_card(f"{fdf['Average Rating'].mean():.2f}", "Avg Rating"), unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Revenue by Membership")
        m = fdf.groupby('Membership Type')['Total Spend'].sum().reset_index()
        fig = px.pie(m, values='Total Spend', names='Membership Type', color_discrete_sequence=['#8b5cf6','#06b6d4','#f59e0b'], hole=0.4)
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Satisfaction Distribution")
        s = fdf['Satisfaction Level'].value_counts().reset_index()
        s.columns = ['Satisfaction','Count']
        colors = {'Satisfied':'#10b981','Neutral':'#f59e0b','Unsatisfied':'#ef4444','Unknown':'#6b7280'}
        fig = px.bar(s, x='Satisfaction', y='Count', color='Satisfaction', color_discrete_map=colors)
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Spend vs Items")
        fig = px.scatter(fdf, x='Items Purchased', y='Total Spend', color='Membership Type', size='Days Since Last Purchase', color_discrete_sequence=['#8b5cf6','#06b6d4','#f59e0b'])
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Spend by Age Group")
        a = fdf.groupby('Age_Group')['Total Spend'].mean().reset_index()
        fig = px.bar(a, x='Age_Group', y='Total Spend', color='Total Spend', color_continuous_scale='Viridis')
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Customer Drill-Down")
    dc = st.selectbox("Select City", ['All Cities'] + sorted(fdf['City'].unique().tolist()), key="edc")
    
    if dc == 'All Cities':
        data = fdf.groupby('City').agg({'Total Spend':'sum','Average Rating':'mean'}).reset_index()
        x, title = 'City', 'Revenue by City'
    else:
        data = fdf[fdf['City']==dc].groupby('Membership Type').agg({'Total Spend':'sum','Average Rating':'mean'}).reset_index()
        x, title = 'Membership Type', f'Revenue by Membership ({dc})'
    
    fig = px.bar(data, x=x, y='Total Spend', color='Average Rating', color_continuous_scale='Viridis', title=title)
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
    st.plotly_chart(fig, use_container_width=True)

# ============ MAIN ============
if dashboard == "Upload Your Data": render_upload()
elif dashboard == "Retail Sales": render_retail()
elif dashboard == "HR Analytics": render_hr()
elif dashboard == "Marketing": render_marketing()
elif dashboard == "E-commerce": render_ecommerce()
