import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime


# PAGE CONFIG & STYLING

st.set_page_config(
    page_title="Data Analytics Demo",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%);
    }
    
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #ffffff !important;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1e1e4a 0%, #2a2a5a 100%);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00d4ff;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #a0a0c0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 8px;
    }
    
    .filter-section {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .stSelectbox > div > div {
        background-color: #1e1e4a;
        border-color: rgba(255,255,255,0.1);
    }
    
    .stMultiSelect > div > div {
        background-color: #1e1e4a;
        border-color: rgba(255,255,255,0.1);
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12122e 0%, #1a1a40 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    .sidebar-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 10px;
    }
    
    .dashboard-header {
        background: linear-gradient(90deg, #8b5cf6 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.5), transparent);
        margin: 30px 0;
    }
</style>
""", unsafe_allow_html=True)


# DATA LOADING FUNCTIONS

@st.cache_data
def load_superstore():
    df = pd.read_csv('data/Sample_-_Superstore.csv', encoding='latin-1')
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%Y')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%m/%d/%Y')
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Month_Name'] = df['Order Date'].dt.strftime('%B')
    df['Quarter'] = df['Order Date'].dt.quarter
    df['Profit_Margin'] = (df['Profit'] / df['Sales'] * 100).round(2)
    return df

@st.cache_data
def load_hr_attrition():
    df = pd.read_csv('data/WA_Fn-UseC_-HR-Employee-Attrition.csv', encoding='utf-8-sig')
    df.columns = df.columns.str.replace('ï»¿', '')
    # Create age groups
    df['Age_Group'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 100], 
                             labels=['18-25', '26-35', '36-45', '46-55', '55+'])
    # Income brackets
    df['Income_Bracket'] = pd.cut(df['MonthlyIncome'], bins=[0, 3000, 6000, 10000, 20000], 
                                   labels=['< $3K', '$3K-$6K', '$6K-$10K', '$10K+'])
    # Tenure groups
    df['Tenure_Group'] = pd.cut(df['YearsAtCompany'], bins=[-1, 2, 5, 10, 40], 
                                 labels=['0-2 yrs', '3-5 yrs', '6-10 yrs', '10+ yrs'])
    return df

@st.cache_data
def load_marketing():
    df = pd.read_csv('data/marketing_campaign_dataset.csv', encoding='latin-1')
    # Clean acquisition cost
    df['Acquisition_Cost'] = df['Acquisition_Cost'].replace('[\$,]', '', regex=True).astype(float)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    # Calculate CTR
    df['CTR'] = (df['Clicks'] / df['Impressions'] * 100).round(2)
    return df

@st.cache_data
def load_ecommerce():
    df = pd.read_csv('data/E-commerce_Customer_Behavior_-_Sheet1.csv', encoding='latin-1')
    # Handle missing satisfaction levels
    df['Satisfaction Level'] = df['Satisfaction Level'].fillna('Unknown')
    # Create age groups
    df['Age_Group'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 100], 
                             labels=['18-25', '26-35', '36-45', '46-55', '55+'])
    # Spend tier
    df['Spend_Tier'] = pd.cut(df['Total Spend'], bins=[0, 500, 1000, 2000, 5000], 
                               labels=['Low', 'Medium', 'High', 'Premium'])
    # Recency groups
    df['Recency_Group'] = pd.cut(df['Days Since Last Purchase'], bins=[-1, 7, 30, 60, 365], 
                                  labels=['Active (0-7d)', 'Recent (8-30d)', 'At Risk (31-60d)', 'Dormant (60d+)'])
    return df


# HELPER FUNCTIONS

def create_metric_card(value, label, prefix="", suffix="", delta=None):
    delta_html = ""
    if delta is not None:
        color = "#10b981" if delta >= 0 else "#ef4444"
        arrow = "↑" if delta >= 0 else "↓"
        delta_html = f'<div style="color: {color}; font-size: 0.9rem; margin-top: 5px;">{arrow} {abs(delta):.2f}%</div>'
    
    return f"""
    <div class="metric-card">
        <div class="metric-value">{prefix}{value}{suffix}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """

def format_number(num, is_count=False):
    """Format numbers - use is_count=True for whole numbers like customers, orders"""
    if is_count:
        if num >= 1_000_000:
            return f"{num/1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num/1_000:.1f}K"
        return f"{int(num)}"
    else:
        if num >= 1_000_000:
            return f"{num/1_000_000:.2f}M"
        elif num >= 1_000:
            return f"{num/1_000:.2f}K"
        return f"{num:.2f}"

# Color schemes
COLORS = {
    'primary': ['#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#ec4899'],
    'sequential': px.colors.sequential.Viridis,
    'diverging': px.colors.diverging.RdYlGn
}


# SIDEBAR

with st.sidebar:
    st.markdown('<p class="sidebar-title">Analytics Demo</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    dashboard = st.selectbox(
        "Choose Dashboard",
        ["Retail Sales", "HR Analytics", "Marketing Performance", "E-commerce Customers"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    <div style="color: #a0a0c0; font-size: 0.85rem; line-height: 1.6;">
    Explore our interactive data analytics demos showcasing the power of modern business intelligence.
    <br><br>
    Each dashboard demonstrates filtering, drill-down capabilities, and actionable insights.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Capabilities")
    st.markdown("""
    - Interactive Filtering
    - Drill-down Analysis
    - Trend Detection
    - Segmentation
    - KPI Tracking
    """)



# DASHBOARD 1: RETAIL SALES (SUPERSTORE)

def render_superstore_dashboard():
    df = load_superstore()
    
    st.markdown('<h1 class="dashboard-header">Retail Sales Analytics</h1>', unsafe_allow_html=True)
    st.markdown("*Analyze sales performance across regions, categories, and time periods*")
    
    # Filters
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        regions = ['All'] + sorted(df['Region'].unique().tolist())
        selected_region = st.selectbox("Region", regions, key="ss_region")
    
    with col2:
        categories = ['All'] + sorted(df['Category'].unique().tolist())
        selected_category = st.selectbox("Category", categories, key="ss_category")
    
    with col3:
        segments = ['All'] + sorted(df['Segment'].unique().tolist())
        selected_segment = st.selectbox("Customer Segment", segments, key="ss_segment")
    
    with col4:
        years = ['All'] + sorted(df['Year'].unique().tolist())
        selected_year = st.selectbox("Year", years, key="ss_year")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Apply filters
    filtered_df = df.copy()
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]
    if selected_segment != 'All':
        filtered_df = filtered_df[filtered_df['Segment'] == selected_segment]
    if selected_year != 'All':
        filtered_df = filtered_df[filtered_df['Year'] == selected_year]
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_sales = filtered_df['Sales'].sum()
    total_profit = filtered_df['Profit'].sum()
    total_orders = filtered_df['Order ID'].nunique()
    avg_profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
    
    with col1:
        st.markdown(create_metric_card(f"${format_number(total_sales)}", "Total Sales"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card(f"${format_number(total_profit)}", "Total Profit"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card(format_number(total_orders, is_count=True), "Orders"), unsafe_allow_html=True)
    with col4:
        st.markdown(create_metric_card(f"{avg_profit_margin:.2f}%", "Profit Margin"), unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Sales Trend Over Time")
        monthly_sales = filtered_df.groupby(filtered_df['Order Date'].dt.to_period('M')).agg({
            'Sales': 'sum',
            'Profit': 'sum'
        }).reset_index()
        monthly_sales['Order Date'] = monthly_sales['Order Date'].astype(str)
        monthly_sales['Sales'] = monthly_sales['Sales'].round(2)
        monthly_sales['Profit'] = monthly_sales['Profit'].round(2)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_sales['Order Date'], y=monthly_sales['Sales'], 
                                  mode='lines+markers', name='Sales',
                                  line=dict(color='#8b5cf6', width=3),
                                  marker=dict(size=6),
                                  hovertemplate='%{x}<br>Sales: $%{y:,.2f}<extra></extra>'))
        fig.add_trace(go.Scatter(x=monthly_sales['Order Date'], y=monthly_sales['Profit'], 
                                  mode='lines+markers', name='Profit',
                                  line=dict(color='#06b6d4', width=3),
                                  marker=dict(size=6),
                                  hovertemplate='%{x}<br>Profit: $%{y:,.2f}<extra></extra>'))
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Sales by Region")
        region_sales = filtered_df.groupby('Region').agg({
            'Sales': 'sum',
            'Profit': 'sum'
        }).reset_index()
        region_sales['Sales'] = region_sales['Sales'].round(2)
        region_sales['Profit'] = region_sales['Profit'].round(2)
        region_sales['Profit_Margin'] = ((region_sales['Profit'] / region_sales['Sales']) * 100).round(2)
        
        fig = px.bar(region_sales, x='Region', y='Sales', color='Profit_Margin',
                     color_continuous_scale='RdYlGn',
                     labels={'Profit_Margin': 'Profit Margin %'})
        fig.update_traces(hovertemplate='<b>%{x}</b><br>Sales: $%{y:,.2f}<br>Profit Margin: %{marker.color:.2f}%<extra></extra>')
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Category Performance")
        # Round values for treemap
        treemap_df = filtered_df.copy()
        treemap_df['Sales'] = treemap_df['Sales'].round(2)
        treemap_df['Profit'] = treemap_df['Profit'].round(2)
        
        fig = px.treemap(treemap_df, path=['Category', 'Sub-Category'], values='Sales',
                         color='Profit', color_continuous_scale='RdYlGn',
                         color_continuous_midpoint=0)
        fig.update_traces(hovertemplate='<b>%{label}</b><br>Sales: $%{value:,.2f}<br>Profit: $%{color:,.2f}<extra></extra>')
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Top 10 Sub-Categories by Sales")
        subcat_sales = filtered_df.groupby('Sub-Category')['Sales'].sum().round(2).sort_values(ascending=True).tail(10)
        
        fig = go.Figure(go.Bar(
            x=subcat_sales.values,
            y=subcat_sales.index,
            orientation='h',
            marker=dict(color='#8b5cf6'),
            hovertemplate='<b>%{y}</b><br>Sales: $%{x:,.2f}<extra></extra>'
        ))
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Drill-down Section - Hierarchical: Region → State → City
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Geographic Drill-Down")
    
    # Level 1: Region selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        regions_list = ['All Regions'] + sorted(filtered_df['Region'].unique().tolist())
        selected_drill_region = st.selectbox("Select Region", regions_list, key="drill_region")
    
    # Filter for level 2
    if selected_drill_region == 'All Regions':
        level1_df = filtered_df
        drill_level = 'Region'
    else:
        level1_df = filtered_df[filtered_df['Region'] == selected_drill_region]
        drill_level = 'State'
    
    with col2:
        if selected_drill_region != 'All Regions':
            states_list = ['All States'] + sorted(level1_df['State'].unique().tolist())
            selected_drill_state = st.selectbox("Select State", states_list, key="drill_state")
        else:
            selected_drill_state = None
            st.selectbox("Select State", ["Select a Region first"], disabled=True, key="drill_state_disabled")
    
    # Filter for level 3
    if selected_drill_state and selected_drill_state != 'All States':
        level2_df = level1_df[level1_df['State'] == selected_drill_state]
        drill_level = 'City'
    elif selected_drill_region != 'All Regions':
        level2_df = level1_df
    else:
        level2_df = level1_df
    
    with col3:
        if selected_drill_state and selected_drill_state != 'All States':
            cities_list = ['All Cities'] + sorted(level2_df['City'].unique().tolist())
            selected_drill_city = st.selectbox("Select City", cities_list, key="drill_city")
        else:
            selected_drill_city = None
            st.selectbox("Select City", ["Select a State first"], disabled=True, key="drill_city_disabled")
    
    # Final filter
    if selected_drill_city and selected_drill_city != 'All Cities':
        final_df = level2_df[level2_df['City'] == selected_drill_city]
        drill_level = 'Product'
    elif selected_drill_state and selected_drill_state != 'All States':
        final_df = level2_df
    else:
        final_df = level2_df
    
    # Build chart based on drill level
    if drill_level == 'Region':
        chart_data = final_df.groupby('Region').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
        chart_data['Sales'] = chart_data['Sales'].round(2)
        chart_data['Profit_Margin'] = ((chart_data['Profit'] / chart_data['Sales']) * 100).round(2)
        x_col, title = 'Region', 'Sales by Region'
    elif drill_level == 'State':
        chart_data = final_df.groupby('State').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
        chart_data['Sales'] = chart_data['Sales'].round(2)
        chart_data['Profit_Margin'] = ((chart_data['Profit'] / chart_data['Sales']) * 100).round(2)
        chart_data = chart_data.sort_values('Sales', ascending=False).head(15)
        x_col, title = 'State', f'Sales by State in {selected_drill_region}'
    elif drill_level == 'City':
        chart_data = final_df.groupby('City').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
        chart_data['Sales'] = chart_data['Sales'].round(2)
        chart_data['Profit_Margin'] = ((chart_data['Profit'] / chart_data['Sales']) * 100).round(2)
        chart_data = chart_data.sort_values('Sales', ascending=False).head(15)
        x_col, title = 'City', f'Sales by City in {selected_drill_state}'
    else:  # Product level
        chart_data = final_df.groupby('Product Name').agg({'Sales': 'sum', 'Profit': 'sum', 'Quantity': 'sum'}).reset_index()
        chart_data['Sales'] = chart_data['Sales'].round(2)
        chart_data['Profit_Margin'] = ((chart_data['Profit'] / chart_data['Sales']) * 100).round(2)
        chart_data = chart_data.sort_values('Sales', ascending=False).head(10)
        x_col, title = 'Product Name', f'Top Products in {selected_drill_city}'
    
    # Display the drill-down chart
    fig = px.bar(chart_data, x=x_col, y='Sales', color='Profit_Margin',
                 color_continuous_scale='RdYlGn',
                 labels={'Profit_Margin': 'Profit Margin %'})
    fig.update_traces(hovertemplate=f'<b>%{{x}}</b><br>Sales: $%{{y:,.2f}}<br>Profit Margin: %{{marker.color:.2f}}%<extra></extra>')
    fig.update_layout(
        title=title,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_tickangle=-45 if drill_level in ['State', 'City', 'Product'] else 0
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show breadcrumb trail
    breadcrumb = "All Regions"
    if selected_drill_region != 'All Regions':
        breadcrumb += f" → {selected_drill_region}"
    if selected_drill_state and selected_drill_state != 'All States':
        breadcrumb += f" → {selected_drill_state}"
    if selected_drill_city and selected_drill_city != 'All Cities':
        breadcrumb += f" → {selected_drill_city}"
    st.caption(f"Current view: {breadcrumb}")



# DASHBOARD 2: HR ANALYTICS

def render_hr_dashboard():
    df = load_hr_attrition()
    
    st.markdown('<h1 class="dashboard-header">HR Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("*Understand employee attrition patterns and workforce dynamics*")
    
    # Filters
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        departments = ['All'] + sorted(df['Department'].unique().tolist())
        selected_dept = st.selectbox("Department", departments, key="hr_dept")
    
    with col2:
        job_roles = ['All'] + sorted(df['JobRole'].unique().tolist())
        selected_role = st.selectbox("Job Role", job_roles, key="hr_role")
    
    with col3:
        age_groups = ['All'] + df['Age_Group'].dropna().unique().tolist()
        selected_age = st.selectbox("Age Group", age_groups, key="hr_age")
    
    with col4:
        genders = ['All'] + sorted(df['Gender'].unique().tolist())
        selected_gender = st.selectbox("Gender", genders, key="hr_gender")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Apply filters
    filtered_df = df.copy()
    if selected_dept != 'All':
        filtered_df = filtered_df[filtered_df['Department'] == selected_dept]
    if selected_role != 'All':
        filtered_df = filtered_df[filtered_df['JobRole'] == selected_role]
    if selected_age != 'All':
        filtered_df = filtered_df[filtered_df['Age_Group'] == selected_age]
    if selected_gender != 'All':
        filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_employees = len(filtered_df)
    attrition_count = filtered_df['Attrition'].value_counts().get('Yes', 0)
    attrition_rate = (attrition_count / total_employees * 100) if total_employees > 0 else 0
    avg_salary = filtered_df['MonthlyIncome'].mean()
    avg_tenure = filtered_df['YearsAtCompany'].mean()
    
    with col1:
        st.markdown(create_metric_card(format_number(total_employees, is_count=True), "Total Employees"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card(f"{attrition_rate:.2f}%", "Attrition Rate"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card(f"${format_number(avg_salary)}", "Avg Monthly Salary"), unsafe_allow_html=True)
    with col4:
        st.markdown(create_metric_card(f"{avg_tenure:.2f}", "Avg Tenure (Years)"), unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Attrition by Department")
        dept_attrition = filtered_df.groupby(['Department', 'Attrition']).size().unstack(fill_value=0)
        dept_attrition['Rate'] = (dept_attrition.get('Yes', 0) / dept_attrition.sum(axis=1) * 100).round(2)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Stayed', x=dept_attrition.index, y=dept_attrition.get('No', []),
                             marker_color='#10b981',
                             hovertemplate='<b>%{x}</b><br>Stayed: %{y}<extra></extra>'))
        fig.add_trace(go.Bar(name='Left', x=dept_attrition.index, y=dept_attrition.get('Yes', []),
                             marker_color='#ef4444',
                             hovertemplate='<b>%{x}</b><br>Left: %{y}<extra></extra>'))
        fig.update_layout(
            barmode='stack',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Salary Distribution by Attrition")
        fig = px.box(filtered_df, x='Attrition', y='MonthlyIncome', color='Attrition',
                     color_discrete_map={'Yes': '#ef4444', 'No': '#10b981'})
        fig.update_traces(hovertemplate='Attrition: %{x}<br>Salary: $%{y:,.2f}<extra></extra>')
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False,
            yaxis_title='Monthly Income ($)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Age Distribution")
        fig = px.histogram(filtered_df, x='Age', color='Attrition', nbins=20,
                          color_discrete_map={'Yes': '#ef4444', 'No': '#10b981'},
                          barmode='overlay', opacity=0.7)
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Job Satisfaction vs Attrition")
        sat_attrition = filtered_df.groupby(['JobSatisfaction', 'Attrition']).size().unstack(fill_value=0)
        sat_attrition['Total'] = sat_attrition.sum(axis=1)
        sat_attrition['Attrition Rate'] = (sat_attrition.get('Yes', 0) / sat_attrition['Total'] * 100).round(2)
        
        satisfaction_labels = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
        sat_attrition.index = sat_attrition.index.map(lambda x: satisfaction_labels.get(x, x))
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=sat_attrition.index, y=sat_attrition['Attrition Rate'],
                             marker_color='#8b5cf6',
                             hovertemplate='<b>%{x}</b><br>Attrition Rate: %{y:.2f}%<extra></extra>'))
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title='Attrition Rate (%)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Drill-down section - Hierarchical: Department → Job Role → Details
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Organizational Drill-Down")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dept_list = ['All Departments'] + sorted(filtered_df['Department'].unique().tolist())
        selected_drill_dept = st.selectbox("Select Department", dept_list, key="hr_drill_dept")
    
    with col2:
        if selected_drill_dept != 'All Departments':
            dept_df = filtered_df[filtered_df['Department'] == selected_drill_dept]
            role_list = ['All Roles'] + sorted(dept_df['JobRole'].unique().tolist())
            selected_drill_role = st.selectbox("Select Job Role", role_list, key="hr_drill_role")
        else:
            selected_drill_role = None
            st.selectbox("Select Job Role", ["Select a Department first"], disabled=True, key="hr_drill_role_disabled")
    
    # Determine drill level and prepare data
    if selected_drill_dept == 'All Departments':
        # Level 1: Show departments
        chart_data = filtered_df.groupby('Department').agg({
            'EmployeeNumber': 'count',
            'Attrition': lambda x: (x == 'Yes').sum(),
            'MonthlyIncome': 'mean'
        }).reset_index()
        chart_data.columns = ['Department', 'Total', 'Left', 'Avg Salary']
        chart_data['Attrition Rate'] = (chart_data['Left'] / chart_data['Total'] * 100).round(2)
        chart_data['Avg Salary'] = chart_data['Avg Salary'].round(2)
        x_col, y_col, title = 'Department', 'Attrition Rate', 'Attrition Rate by Department'
        color_col = 'Avg Salary'
    elif selected_drill_role is None or selected_drill_role == 'All Roles':
        # Level 2: Show job roles within department
        dept_df = filtered_df[filtered_df['Department'] == selected_drill_dept]
        chart_data = dept_df.groupby('JobRole').agg({
            'EmployeeNumber': 'count',
            'Attrition': lambda x: (x == 'Yes').sum(),
            'MonthlyIncome': 'mean'
        }).reset_index()
        chart_data.columns = ['Job Role', 'Total', 'Left', 'Avg Salary']
        chart_data['Attrition Rate'] = (chart_data['Left'] / chart_data['Total'] * 100).round(2)
        chart_data['Avg Salary'] = chart_data['Avg Salary'].round(2)
        chart_data = chart_data.sort_values('Attrition Rate', ascending=False)
        x_col, y_col, title = 'Job Role', 'Attrition Rate', f'Attrition Rate by Role in {selected_drill_dept}'
        color_col = 'Avg Salary'
    else:
        # Level 3: Show detailed metrics for selected role
        role_df = filtered_df[(filtered_df['Department'] == selected_drill_dept) & (filtered_df['JobRole'] == selected_drill_role)]
        chart_data = role_df.groupby('JobSatisfaction').agg({
            'EmployeeNumber': 'count',
            'Attrition': lambda x: (x == 'Yes').sum(),
            'MonthlyIncome': 'mean'
        }).reset_index()
        chart_data.columns = ['Satisfaction', 'Total', 'Left', 'Avg Salary']
        chart_data['Attrition Rate'] = (chart_data['Left'] / chart_data['Total'] * 100).round(2)
        chart_data['Avg Salary'] = chart_data['Avg Salary'].round(2)
        satisfaction_labels = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
        chart_data['Satisfaction'] = chart_data['Satisfaction'].map(satisfaction_labels)
        x_col, y_col, title = 'Satisfaction', 'Attrition Rate', f'Attrition by Satisfaction Level: {selected_drill_role}'
        color_col = 'Total'
    
    # Display chart
    fig = px.bar(chart_data, x=x_col, y=y_col, color=color_col,
                 color_continuous_scale='Viridis',
                 labels={color_col: 'Avg Salary ($)' if color_col == 'Avg Salary' else 'Employee Count'})
    
    # Use appropriate format for hover based on color column type
    if color_col == 'Total':
        hover_format = f'<b>%{{x}}</b><br>Attrition Rate: %{{y:.2f}}%<br>Employees: %{{marker.color:,.0f}}<extra></extra>'
    else:
        hover_format = f'<b>%{{x}}</b><br>Attrition Rate: %{{y:.2f}}%<br>Avg Salary: $%{{marker.color:,.2f}}<extra></extra>'
    fig.update_traces(hovertemplate=hover_format)
    fig.update_layout(
        title=title,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_tickangle=-45 if x_col == 'Job Role' else 0,
        yaxis_title='Attrition Rate (%)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Breadcrumb
    breadcrumb = "All Departments"
    if selected_drill_dept != 'All Departments':
        breadcrumb += f" → {selected_drill_dept}"
    if selected_drill_role and selected_drill_role != 'All Roles':
        breadcrumb += f" → {selected_drill_role}"
    st.caption(f"Current view: {breadcrumb}")



# DASHBOARD 3: MARKETING PERFORMANCE

def render_marketing_dashboard():
    df = load_marketing()
    
    st.markdown('<h1 class="dashboard-header">Marketing Campaign Analytics</h1>', unsafe_allow_html=True)
    st.markdown("*Evaluate campaign performance across channels and audiences*")
    
    # Filters
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        channels = ['All'] + sorted(df['Channel_Used'].unique().tolist())
        selected_channel = st.selectbox("Channel", channels, key="mkt_channel")
    
    with col2:
        campaign_types = ['All'] + sorted(df['Campaign_Type'].unique().tolist())
        selected_type = st.selectbox("Campaign Type", campaign_types, key="mkt_type")
    
    with col3:
        audiences = ['All'] + sorted(df['Target_Audience'].unique().tolist())
        selected_audience = st.selectbox("Target Audience", audiences, key="mkt_audience")
    
    with col4:
        segments = ['All'] + sorted(df['Customer_Segment'].unique().tolist())
        selected_segment = st.selectbox("Customer Segment", segments, key="mkt_segment")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Apply filters
    filtered_df = df.copy()
    if selected_channel != 'All':
        filtered_df = filtered_df[filtered_df['Channel_Used'] == selected_channel]
    if selected_type != 'All':
        filtered_df = filtered_df[filtered_df['Campaign_Type'] == selected_type]
    if selected_audience != 'All':
        filtered_df = filtered_df[filtered_df['Target_Audience'] == selected_audience]
    if selected_segment != 'All':
        filtered_df = filtered_df[filtered_df['Customer_Segment'] == selected_segment]
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_campaigns = len(filtered_df)
    avg_roi = filtered_df['ROI'].mean()
    avg_conversion = filtered_df['Conversion_Rate'].mean() * 100
    total_spend = filtered_df['Acquisition_Cost'].sum()
    
    with col1:
        st.markdown(create_metric_card(format_number(total_campaigns, is_count=True), "Total Campaigns"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card(f"{avg_roi:.2f}x", "Average ROI"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card(f"{avg_conversion:.2f}%", "Avg Conversion Rate"), unsafe_allow_html=True)
    with col4:
        st.markdown(create_metric_card(f"${format_number(total_spend)}", "Total Ad Spend"), unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ROI by Channel")
        channel_perf = filtered_df.groupby('Channel_Used').agg({
            'ROI': 'mean',
            'Conversion_Rate': 'mean',
            'Campaign_ID': 'count'
        }).reset_index()
        channel_perf.columns = ['Channel', 'Avg ROI', 'Avg Conversion', 'Campaigns']
        channel_perf['Avg ROI'] = channel_perf['Avg ROI'].round(2)
        channel_perf['Avg Conversion'] = (channel_perf['Avg Conversion'] * 100).round(2)
        
        fig = px.bar(channel_perf, x='Channel', y='Avg ROI', color='Avg Conversion',
                     color_continuous_scale='Viridis',
                     labels={'Avg Conversion': 'Conversion %'})
        fig.update_traces(hovertemplate='<b>%{x}</b><br>ROI: %{y:.2f}x<br>Conversion: %{marker.color:.2f}%<extra></extra>')
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Campaign Type Performance")
        type_perf = filtered_df.groupby('Campaign_Type').agg({
            'ROI': 'mean',
            'Conversion_Rate': 'mean',
            'Engagement_Score': 'mean'
        }).reset_index()
        type_perf['ROI'] = type_perf['ROI'].round(2)
        type_perf['Conversion_Rate'] = (type_perf['Conversion_Rate'] * 100).round(2)
        type_perf['Engagement_Score'] = type_perf['Engagement_Score'].round(2)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='ROI', x=type_perf['Campaign_Type'], y=type_perf['ROI'],
                             marker_color='#8b5cf6',
                             hovertemplate='<b>%{x}</b><br>ROI: %{y:.2f}x<extra></extra>'))
        fig.add_trace(go.Bar(name='Engagement', x=type_perf['Campaign_Type'], y=type_perf['Engagement_Score'],
                             marker_color='#06b6d4',
                             hovertemplate='<b>%{x}</b><br>Engagement: %{y:.2f}<extra></extra>'))
        fig.update_layout(
            barmode='group',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Performance by Target Audience")
        audience_perf = filtered_df.groupby('Target_Audience').agg({
            'ROI': 'mean',
            'Conversion_Rate': 'mean',
            'Acquisition_Cost': 'mean'
        }).reset_index()
        audience_perf['Conversion_Rate'] = (audience_perf['Conversion_Rate'] * 100).round(2)
        audience_perf['ROI'] = audience_perf['ROI'].round(2)
        audience_perf['Acquisition_Cost'] = audience_perf['Acquisition_Cost'].round(2)
        
        fig = px.scatter(audience_perf, x='Conversion_Rate', y='ROI', 
                        text='Target_Audience', size='Acquisition_Cost',
                        color='Acquisition_Cost', color_continuous_scale='Viridis',
                        labels={'Acquisition_Cost': 'Avg Ad Spend ($)'})
        fig.update_traces(textposition='top center', textfont_size=9,
                         hovertemplate='<b>%{text}</b><br>Conversion: %{x:.2f}%<br>ROI: %{y:.2f}x<br>Avg Spend: $%{marker.size:,.2f}<extra></extra>')
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title='Conversion Rate (%)',
            yaxis_title='ROI'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Monthly Campaign Trends")
        monthly_data = filtered_df.groupby(filtered_df['Date'].dt.to_period('M')).agg({
            'ROI': 'mean',
            'Campaign_ID': 'count',
            'Acquisition_Cost': 'sum'
        }).reset_index()
        monthly_data['Date'] = monthly_data['Date'].astype(str)
        monthly_data['ROI'] = monthly_data['ROI'].round(2)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=monthly_data['Date'], y=monthly_data['ROI'],
                                 mode='lines+markers', name='Avg ROI',
                                 line=dict(color='#8b5cf6', width=3),
                                 hovertemplate='%{x}<br>ROI: %{y:.2f}x<extra></extra>'), secondary_y=False)
        fig.add_trace(go.Bar(x=monthly_data['Date'], y=monthly_data['Campaign_ID'],
                            name='Campaigns', marker_color='rgba(6, 182, 212, 0.5)',
                            hovertemplate='%{x}<br>Campaigns: %{y}<extra></extra>'), secondary_y=True)
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(title_text="ROI", secondary_y=False)
        fig.update_yaxes(title_text="Campaigns", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Drill-down - Hierarchical: Channel → Campaign Type → Location
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Campaign Drill-Down")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        channel_list = ['All Channels'] + sorted(filtered_df['Channel_Used'].unique().tolist())
        selected_drill_channel = st.selectbox("Select Channel", channel_list, key="mkt_drill_channel")
    
    with col2:
        if selected_drill_channel != 'All Channels':
            channel_df = filtered_df[filtered_df['Channel_Used'] == selected_drill_channel]
            type_list = ['All Types'] + sorted(channel_df['Campaign_Type'].unique().tolist())
            selected_drill_type = st.selectbox("Select Campaign Type", type_list, key="mkt_drill_type")
        else:
            selected_drill_type = None
            st.selectbox("Select Campaign Type", ["Select a Channel first"], disabled=True, key="mkt_drill_type_disabled")
    
    with col3:
        if selected_drill_type and selected_drill_type != 'All Types':
            type_df = filtered_df[(filtered_df['Channel_Used'] == selected_drill_channel) & 
                                  (filtered_df['Campaign_Type'] == selected_drill_type)]
            location_list = ['All Locations'] + sorted(type_df['Location'].unique().tolist())
            selected_drill_location = st.selectbox("Select Location", location_list, key="mkt_drill_location")
        else:
            selected_drill_location = None
            st.selectbox("Select Location", ["Select a Campaign Type first"], disabled=True, key="mkt_drill_location_disabled")
    
    # Determine drill level and prepare data
    if selected_drill_channel == 'All Channels':
        chart_data = filtered_df.groupby('Channel_Used').agg({
            'ROI': 'mean', 'Conversion_Rate': 'mean', 'Campaign_ID': 'count'
        }).reset_index()
        chart_data.columns = ['Channel', 'Avg ROI', 'Conversion', 'Campaigns']
        chart_data['Avg ROI'] = chart_data['Avg ROI'].round(2)
        chart_data['Conversion'] = (chart_data['Conversion'] * 100).round(2)
        x_col, title = 'Channel', 'ROI by Channel'
    elif selected_drill_type is None or selected_drill_type == 'All Types':
        channel_df = filtered_df[filtered_df['Channel_Used'] == selected_drill_channel]
        chart_data = channel_df.groupby('Campaign_Type').agg({
            'ROI': 'mean', 'Conversion_Rate': 'mean', 'Campaign_ID': 'count'
        }).reset_index()
        chart_data.columns = ['Campaign Type', 'Avg ROI', 'Conversion', 'Campaigns']
        chart_data['Avg ROI'] = chart_data['Avg ROI'].round(2)
        chart_data['Conversion'] = (chart_data['Conversion'] * 100).round(2)
        x_col, title = 'Campaign Type', f'ROI by Campaign Type ({selected_drill_channel})'
    elif selected_drill_location is None or selected_drill_location == 'All Locations':
        type_df = filtered_df[(filtered_df['Channel_Used'] == selected_drill_channel) & 
                              (filtered_df['Campaign_Type'] == selected_drill_type)]
        chart_data = type_df.groupby('Location').agg({
            'ROI': 'mean', 'Conversion_Rate': 'mean', 'Campaign_ID': 'count'
        }).reset_index()
        chart_data.columns = ['Location', 'Avg ROI', 'Conversion', 'Campaigns']
        chart_data['Avg ROI'] = chart_data['Avg ROI'].round(2)
        chart_data['Conversion'] = (chart_data['Conversion'] * 100).round(2)
        chart_data = chart_data.sort_values('Avg ROI', ascending=False)
        x_col, title = 'Location', f'ROI by Location ({selected_drill_channel} - {selected_drill_type})'
    else:
        location_df = filtered_df[(filtered_df['Channel_Used'] == selected_drill_channel) & 
                                  (filtered_df['Campaign_Type'] == selected_drill_type) &
                                  (filtered_df['Location'] == selected_drill_location)]
        chart_data = location_df.groupby('Target_Audience').agg({
            'ROI': 'mean', 'Conversion_Rate': 'mean', 'Campaign_ID': 'count'
        }).reset_index()
        chart_data.columns = ['Target Audience', 'Avg ROI', 'Conversion', 'Campaigns']
        chart_data['Avg ROI'] = chart_data['Avg ROI'].round(2)
        chart_data['Conversion'] = (chart_data['Conversion'] * 100).round(2)
        chart_data = chart_data.sort_values('Avg ROI', ascending=False)
        x_col, title = 'Target Audience', f'ROI by Audience ({selected_drill_location})'
    
    # Display chart
    fig = px.bar(chart_data, x=x_col, y='Avg ROI', color='Conversion',
                 color_continuous_scale='Viridis',
                 labels={'Conversion': 'Conversion %'})
    fig.update_traces(hovertemplate=f'<b>%{{x}}</b><br>ROI: %{{y:.2f}}x<br>Conversion: %{{marker.color:.2f}}%<extra></extra>')
    fig.update_layout(
        title=title,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_tickangle=-45 if len(chart_data) > 5 else 0,
        yaxis_title='Average ROI'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Breadcrumb
    breadcrumb = "All Channels"
    if selected_drill_channel != 'All Channels':
        breadcrumb += f" → {selected_drill_channel}"
    if selected_drill_type and selected_drill_type != 'All Types':
        breadcrumb += f" → {selected_drill_type}"
    if selected_drill_location and selected_drill_location != 'All Locations':
        breadcrumb += f" → {selected_drill_location}"
    st.caption(f"Current view: {breadcrumb}")



# DASHBOARD 4: E-COMMERCE CUSTOMERS

def render_ecommerce_dashboard():
    df = load_ecommerce()
    
    st.markdown('<h1 class="dashboard-header">E-commerce Customer Analytics</h1>', unsafe_allow_html=True)
    st.markdown("*Understand customer behavior, satisfaction, and spending patterns*")
    
    # Filters
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cities = ['All'] + sorted(df['City'].unique().tolist())
        selected_city = st.selectbox("City", cities, key="ec_city")
    
    with col2:
        memberships = ['All'] + sorted(df['Membership Type'].unique().tolist())
        selected_membership = st.selectbox("Membership Type", memberships, key="ec_membership")
    
    with col3:
        satisfaction = ['All'] + sorted(df['Satisfaction Level'].dropna().unique().tolist())
        selected_satisfaction = st.selectbox("Satisfaction Level", satisfaction, key="ec_satisfaction")
    
    with col4:
        genders = ['All'] + sorted(df['Gender'].unique().tolist())
        selected_gender = st.selectbox("Gender", genders, key="ec_gender")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Apply filters
    filtered_df = df.copy()
    if selected_city != 'All':
        filtered_df = filtered_df[filtered_df['City'] == selected_city]
    if selected_membership != 'All':
        filtered_df = filtered_df[filtered_df['Membership Type'] == selected_membership]
    if selected_satisfaction != 'All':
        filtered_df = filtered_df[filtered_df['Satisfaction Level'] == selected_satisfaction]
    if selected_gender != 'All':
        filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(filtered_df)
    total_revenue = filtered_df['Total Spend'].sum()
    avg_spend = filtered_df['Total Spend'].mean()
    avg_rating = filtered_df['Average Rating'].mean()
    
    with col1:
        st.markdown(create_metric_card(format_number(total_customers, is_count=True), "Total Customers"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card(f"${format_number(total_revenue)}", "Total Revenue"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card(f"${avg_spend:.2f}", "Avg Spend per Customer"), unsafe_allow_html=True)
    with col4:
        st.markdown(create_metric_card(f"{avg_rating:.2f}", "Avg Rating", suffix=""), unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Revenue by Membership Type")
        membership_revenue = filtered_df.groupby('Membership Type').agg({
            'Total Spend': 'sum',
            'Customer ID': 'count',
            'Average Rating': 'mean'
        }).reset_index()
        membership_revenue.columns = ['Membership', 'Revenue', 'Customers', 'Avg Rating']
        membership_revenue['Revenue'] = membership_revenue['Revenue'].round(2)
        membership_revenue['Avg Rating'] = membership_revenue['Avg Rating'].round(2)
        
        fig = px.pie(membership_revenue, values='Revenue', names='Membership',
                     color_discrete_sequence=['#8b5cf6', '#06b6d4', '#f59e0b'],
                     hole=0.4)
        fig.update_traces(hovertemplate='<b>%{label}</b><br>Revenue: $%{value:,.2f}<br>Share: %{percent}<extra></extra>')
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            height=350,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Customer Satisfaction Distribution")
        satisfaction_counts = filtered_df['Satisfaction Level'].value_counts()
        colors = {'Satisfied': '#10b981', 'Neutral': '#f59e0b', 'Unsatisfied': '#ef4444', 'Unknown': '#6b7280'}
        
        fig = go.Figure(data=[go.Bar(
            x=satisfaction_counts.index,
            y=satisfaction_counts.values,
            marker_color=[colors.get(x, '#8b5cf6') for x in satisfaction_counts.index],
            hovertemplate='<b>%{x}</b><br>Customers: %{y}<extra></extra>'
        )])
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title='Number of Customers'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Spend vs Items Purchased")
        scatter_df = filtered_df.copy()
        scatter_df['Total Spend'] = scatter_df['Total Spend'].round(2)
        scatter_df['Average Rating'] = scatter_df['Average Rating'].round(2)
        
        fig = px.scatter(scatter_df, x='Items Purchased', y='Total Spend',
                        color='Membership Type', size='Days Since Last Purchase',
                        color_discrete_sequence=['#8b5cf6', '#06b6d4', '#f59e0b'],
                        labels={'Days Since Last Purchase': 'Days Inactive'})
        fig.update_traces(hovertemplate='<b>%{customdata[0]}</b><br>Items: %{x}<br>Spend: $%{y:,.2f}<br>Days Inactive: %{marker.size}<extra></extra>',
                         customdata=scatter_df[['City']])
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title='Total Spend ($)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Customer Segments by Age")
        age_analysis = filtered_df.groupby('Age_Group').agg({
            'Total Spend': 'mean',
            'Customer ID': 'count',
            'Items Purchased': 'mean'
        }).reset_index()
        age_analysis.columns = ['Age Group', 'Avg Spend', 'Count', 'Avg Items']
        age_analysis['Avg Spend'] = age_analysis['Avg Spend'].round(2)
        age_analysis['Avg Items'] = age_analysis['Avg Items'].round(2)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=age_analysis['Age Group'], y=age_analysis['Avg Spend'],
                             marker_color='#8b5cf6',
                             hovertemplate='<b>%{x}</b><br>Avg Spend: $%{y:,.2f}<extra></extra>'))
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title='Average Spend ($)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Drill-down - Hierarchical: City → Membership → Satisfaction Level
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Customer Drill-Down")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        city_list = ['All Cities'] + sorted(filtered_df['City'].unique().tolist())
        selected_drill_city = st.selectbox("Select City", city_list, key="ec_drill_city")
    
    with col2:
        if selected_drill_city != 'All Cities':
            city_df = filtered_df[filtered_df['City'] == selected_drill_city]
            membership_list = ['All Memberships'] + sorted(city_df['Membership Type'].unique().tolist())
            selected_drill_membership = st.selectbox("Select Membership", membership_list, key="ec_drill_membership")
        else:
            selected_drill_membership = None
            st.selectbox("Select Membership", ["Select a City first"], disabled=True, key="ec_drill_membership_disabled")
    
    with col3:
        if selected_drill_membership and selected_drill_membership != 'All Memberships':
            membership_df = filtered_df[(filtered_df['City'] == selected_drill_city) & 
                                        (filtered_df['Membership Type'] == selected_drill_membership)]
            satisfaction_list = ['All Levels'] + sorted(membership_df['Satisfaction Level'].dropna().unique().tolist())
            selected_drill_satisfaction = st.selectbox("Select Satisfaction", satisfaction_list, key="ec_drill_satisfaction")
        else:
            selected_drill_satisfaction = None
            st.selectbox("Select Satisfaction", ["Select a Membership first"], disabled=True, key="ec_drill_satisfaction_disabled")
    
    # Determine drill level and prepare data
    if selected_drill_city == 'All Cities':
        chart_data = filtered_df.groupby('City').agg({
            'Total Spend': 'sum', 'Customer ID': 'count', 'Average Rating': 'mean'
        }).reset_index()
        chart_data.columns = ['City', 'Revenue', 'Customers', 'Avg Rating']
        chart_data['Revenue'] = chart_data['Revenue'].round(2)
        chart_data['Avg Rating'] = chart_data['Avg Rating'].round(2)
        chart_data = chart_data.sort_values('Revenue', ascending=False)
        x_col, y_col, title = 'City', 'Revenue', 'Revenue by City'
        color_col = 'Avg Rating'
    elif selected_drill_membership is None or selected_drill_membership == 'All Memberships':
        city_df = filtered_df[filtered_df['City'] == selected_drill_city]
        chart_data = city_df.groupby('Membership Type').agg({
            'Total Spend': 'sum', 'Customer ID': 'count', 'Average Rating': 'mean'
        }).reset_index()
        chart_data.columns = ['Membership', 'Revenue', 'Customers', 'Avg Rating']
        chart_data['Revenue'] = chart_data['Revenue'].round(2)
        chart_data['Avg Rating'] = chart_data['Avg Rating'].round(2)
        x_col, y_col, title = 'Membership', 'Revenue', f'Revenue by Membership ({selected_drill_city})'
        color_col = 'Avg Rating'
    elif selected_drill_satisfaction is None or selected_drill_satisfaction == 'All Levels':
        membership_df = filtered_df[(filtered_df['City'] == selected_drill_city) & 
                                    (filtered_df['Membership Type'] == selected_drill_membership)]
        chart_data = membership_df.groupby('Satisfaction Level').agg({
            'Total Spend': 'sum', 'Customer ID': 'count', 'Average Rating': 'mean'
        }).reset_index()
        chart_data.columns = ['Satisfaction', 'Revenue', 'Customers', 'Avg Rating']
        chart_data['Revenue'] = chart_data['Revenue'].round(2)
        chart_data['Avg Rating'] = chart_data['Avg Rating'].round(2)
        x_col, y_col, title = 'Satisfaction', 'Revenue', f'Revenue by Satisfaction ({selected_drill_membership} in {selected_drill_city})'
        color_col = 'Customers'
    else:
        satisfaction_df = filtered_df[(filtered_df['City'] == selected_drill_city) & 
                                      (filtered_df['Membership Type'] == selected_drill_membership) &
                                      (filtered_df['Satisfaction Level'] == selected_drill_satisfaction)]
        chart_data = satisfaction_df.groupby('Gender').agg({
            'Total Spend': 'sum', 'Customer ID': 'count', 'Average Rating': 'mean'
        }).reset_index()
        chart_data.columns = ['Gender', 'Revenue', 'Customers', 'Avg Rating']
        chart_data['Revenue'] = chart_data['Revenue'].round(2)
        chart_data['Avg Rating'] = chart_data['Avg Rating'].round(2)
        x_col, y_col, title = 'Gender', 'Revenue', f'Revenue by Gender ({selected_drill_satisfaction})'
        color_col = 'Customers'
    
    # Display chart
    fig = px.bar(chart_data, x=x_col, y=y_col, color=color_col,
                 color_continuous_scale='Viridis',
                 labels={color_col: 'Avg Rating' if color_col == 'Avg Rating' else 'Customers'})
    
    # Use appropriate format for hover based on color column type
    if color_col == 'Customers':
        hover_format = f'<b>%{{x}}</b><br>Revenue: $%{{y:,.2f}}<br>{color_col}: %{{marker.color:,.0f}}<extra></extra>'
    else:
        hover_format = f'<b>%{{x}}</b><br>Revenue: $%{{y:,.2f}}<br>{color_col}: %{{marker.color:.2f}}<extra></extra>'
    fig.update_traces(hovertemplate=hover_format)
    fig.update_layout(
        title=title,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_tickangle=-45 if len(chart_data) > 4 else 0,
        yaxis_title='Revenue ($)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Breadcrumb
    breadcrumb = "All Cities"
    if selected_drill_city != 'All Cities':
        breadcrumb += f" → {selected_drill_city}"
    if selected_drill_membership and selected_drill_membership != 'All Memberships':
        breadcrumb += f" → {selected_drill_membership}"
    if selected_drill_satisfaction and selected_drill_satisfaction != 'All Levels':
        breadcrumb += f" → {selected_drill_satisfaction}"
    st.caption(f"Current view: {breadcrumb}")



# MAIN ROUTING

if dashboard == "Retail Sales":
    render_superstore_dashboard()
elif dashboard == "HR Analytics":
    render_hr_dashboard()
elif dashboard == "Marketing Performance":
    render_marketing_dashboard()
elif dashboard == "E-commerce Customers":
    render_ecommerce_dashboard()