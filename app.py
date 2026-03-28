import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler

# Configuración de la página
st.set_page_config(page_title="AI Adoption Dashboard", layout="wide")

# Título principal
st.title("🤖 Adopción Global de IA e Impacto Laboral")

# Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("ai_company_adoption.csv")
    
    # Conversión de tipos de datos
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category')
    
    for col in df.select_dtypes(include='float64').columns:
        df[col] = df[col].astype('float32')
    
    for col in df.select_dtypes(include='int64').columns:
        df[col] = df[col].astype('int32')
    
    return df

df = load_data()

# KPIs principales
kpis = [
    'productivity_change_percent',
    'revenue_growth_percent',
    'cost_reduction_percent',
    'task_automation_rate',
    'time_saved_per_week',
    'ai_adoption_rate'
]

# Agregar datos a nivel de empresa (tomar el promedio de cada empresa)
@st.cache_data
def aggregate_by_company(df):
    """Agrupa los datos por company_id y toma el promedio de cada empresa"""
    numeric_cols = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns.tolist()
    
    df_agg = df.groupby('company_id', observed=False)[numeric_cols].mean().reset_index()
    
    # Mantener columnas categóricas (tomar la primera ocurrencia de cada empresa)
    categorical_cols = ['company_size', 'industry', 'country', 'region']
    for col in categorical_cols:
        if col in df.columns:
            df_agg[col] = df.groupby('company_id', observed=False)[col].first().values
    
    return df_agg

df = aggregate_by_company(df)

# Aplicar normalización a los KPIs
@st.cache_data
def normalize_kpis(df):
    df_copy = df.copy()
    
    # Identificar y tratar outliers usando el método IQR (capping)
    for col in kpis:
        Q1 = df_copy[col].quantile(0.25)
        Q3 = df_copy[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Normalización usando MinMaxScaler
    scaler = MinMaxScaler()
    df_copy[kpis] = scaler.fit_transform(df_copy[kpis])
    
    return df_copy

df_normalized = normalize_kpis(df)

# Calcular ROI
df['roi_ai'] = df['revenue_growth_percent'] + df['cost_reduction_percent']

# ==================== KPIs PRINCIPALES POR TAMAÑO DE EMPRESA ====================
st.markdown("---")
st.subheader("� Análisis financiero y operativo enfocado en el crecimiento por adopción de IA")

# Calcular métricas por tamaño de empresa
company_sizes = ['Enterprise', 'SME', 'Startup']
company_size_icons = {'Enterprise': '🏢', 'SME': '📊', 'Startup': '🚀'}

# Mostrar los tres tamaños de empresa en columnas simultáneas
col_ent, col_sme, col_startup = st.columns(3)

metrics_map = {
    'Enterprise': {
        'title': f"{company_size_icons['Enterprise']} Enterprise",
        'df': df[df['company_size'] == 'Enterprise']
    },
    'SME': {
        'title': f"{company_size_icons['SME']} SME",
        'df': df[df['company_size'] == 'SME']
    },
    'Startup': {
        'title': f"{company_size_icons['Startup']} Startup",
        'df': df[df['company_size'] == 'Startup']
    }
}

for col, company_size in zip([col_ent, col_sme, col_startup], company_sizes):
    df_size = metrics_map[company_size]['df']
    if df_size.empty:
        with col:
            st.markdown(f"### {metrics_map[company_size]['title']}")
            st.write("No hay datos disponibles")
        continue

    avg_productivity = df_size['productivity_change_percent'].mean()
    avg_revenue = df_size['revenue_growth_percent'].mean()
    avg_cost_reduction = df_size['cost_reduction_percent'].mean()
    total_cost_savings = (avg_cost_reduction / 100) * df_size['annual_revenue_usd_millions'].mean()
    total_revenue = (avg_revenue / 100) * df_size['annual_revenue_usd_millions'].mean()

    with col:
        st.markdown(f"### {metrics_map[company_size]['title']}")

        st.markdown(
            f"""
            <div style=\"background-color: #1a2332; padding: 15px; border-radius: 8px; border-left: 4px solid #00d9ff; margin-bottom: 10px;\">
                <p style=\"color: #999; margin: 0; font-size: 12px;\">📊 Prod. Promedio</p>
                <p style=\"color: #00d9ff; margin: 5px 0 0 0; font-size: 32px; font-weight: bold;\">{avg_productivity:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style=\"background-color: #1a2332; padding: 15px; border-radius: 8px; border-left: 4px solid #ffd700; margin-bottom: 10px;\">
                <p style=\"color: #999; margin: 0; font-size: 12px;\">💰 Crecimiento de Ingresos</p>
                <p style=\"color: #00d9ff; margin: 5px 0 0 0; font-size: 24px; font-weight: bold;\">{avg_revenue:.2f}%</p>
                <p style=\"color: #00cc88; margin: 5px 0 0 0; font-size: 12px;\">+ ${total_revenue:.1f}M Ingresos Totales</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style=\"background-color: #1a2332; padding: 15px; border-radius: 8px; border-left: 4px solid #ff6b6b;\">
                <p style=\"color: #999; margin: 0; font-size: 12px;\">💼 Ahorro en Costos</p>
                <p style=\"color: #ff6b6b; margin: 5px 0 0 0; font-size: 32px; font-weight: bold;\">{avg_cost_reduction:.1f}%</p>
                <p style=\"color: #00cc88; margin: 5px 0 0 0; font-size: 12px;\">+ ${total_cost_savings:.1f}M Ahorrados</p>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("")  # Espacio después del bloque principal

st.markdown("---")

# Crear pestañas
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Distribución de KPIs",
    "🔗 Análisis de Correlación",
    "💰 Análisis de Ingresos",
    "🎯 Relación Adopción-Performance",
    "📊 ROI y Tamaño de Empresa",
    "🌍 Mapa Geográfico"
])

# ==================== TAB 1: Distribución de KPIs ====================
with tab1:
    st.header("Distribución de Key Performance Indicators (KPIs)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("KPIs Originales")
        
        # Crear subgráficos con histogramas interactivos
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=kpis,
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "histogram"}]]
        )
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]
        for idx, (col, pos) in enumerate(zip(kpis, positions)):
            fig.add_trace(
                go.Histogram(x=df[col], nbinsx=30, name=col, marker_color='lightblue'),
                row=pos[0], col=pos[1]
            )
        
        fig.update_layout(height=900, showlegend=False, title_text="Distribución de KPIs Originales")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("KPIs Normalizados - Distribuciones")
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=kpis,
            specs=[[{"type": "box"}, {"type": "box"}],
                   [{"type": "box"}, {"type": "box"}],
                   [{"type": "box"}, {"type": "box"}]]
        )
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]
        for idx, (col, pos) in enumerate(zip(kpis, positions)):
            fig.add_trace(
                go.Box(y=df_normalized[col], name=col, marker_color='lightgreen'),
                row=pos[0], col=pos[1]
            )
        
        fig.update_layout(height=900, showlegend=False, title_text="Boxplots de KPIs Normalizados")
        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 2: Correlación ====================
with tab2:
    st.header("Matriz de Correlación de KPIs")
    
    corr = df[kpis].corr()
    
    # Crear heatmap interactivo con Plotly
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlación")
    ))
    
    fig.update_layout(
        title='Matriz de Correlación de KPIs',
        xaxis_title='KPIs',
        yaxis_title='KPIs',
        height=600,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Mostrar tabla de correlación
    st.subheader("Tabla de Correlaciones")
    st.dataframe(corr, use_container_width=True)

# ==================== TAB 3: Análisis de Ingresos ====================
with tab3:
    st.header("Análisis Univariable: Porcentaje de Crecimiento de Ingresos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        mean_growth = df['revenue_growth_percent'].mean()
        median_growth = df['revenue_growth_percent'].median()
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=df['revenue_growth_percent'],
            nbinsx=30,
            name='Frecuencia',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        fig.add_vline(x=mean_growth, line_dash="dash", line_color="red", 
                     annotation_text=f"Media: {mean_growth:.2f}", annotation_position="top right")
        fig.add_vline(x=median_growth, line_dash="dot", line_color="green",
                     annotation_text=f"Mediana: {median_growth:.2f}", annotation_position="top left")
        
        fig.update_layout(
            title='Distribución del Porcentaje de Crecimiento de Ingresos',
            xaxis_title='Porcentaje de Crecimiento de Ingresos',
            yaxis_title='Frecuencia',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Estadísticas Descriptivas")
        stats_data = df['revenue_growth_percent'].describe()
        st.dataframe(stats_data, use_container_width=True)
        
        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric("Media", f"{df['revenue_growth_percent'].mean():.2f}")
            st.metric("Desviación Estándar", f"{df['revenue_growth_percent'].std():.2f}")
        with col_metric2:
            st.metric("Mediana", f"{df['revenue_growth_percent'].median():.2f}")
            st.metric("Skewness", f"{df['revenue_growth_percent'].skew():.2f}")

# ==================== TAB 4: Relación Adopción-Performance ====================
with tab4:
    st.header("Relación entre Adopción de IA y Métricas de Performance")
    
    # Función para crear gráficos de densidad 2D interactivos con línea de regresión
    def create_interactive_density_plot(x_data, y_data, title, colorscale, x_label, y_label):
        """Crea un gráfico de densidad 2D interactivo con línea de regresión"""
        
        # Crear figura
        fig = go.Figure()
        
        # Agregar histograma 2D (densidad)
        fig.add_trace(go.Histogram2d(
            x=x_data,
            y=y_data,
            nbinsx=40,
            nbinsy=40,
            colorscale=colorscale,
            colorbar=dict(
                title="Densidad"
            ),
            hovertemplate='<b>%{xaxis.title.text}</b>: %{x:.2f}<br>' +
                         '<b>%{yaxis.title.text}</b>: %{y:.2f}<br>' +
                         '<b>Densidad</b>: %{z}<extra></extra>',
            name='Densidad'
        ))
        
        # Calcular línea de regresión
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_data.min(), x_data.max(), 100)
        y_line = p(x_line)
        
        # Agregar línea de regresión
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name='Línea de Regresión',
            line=dict(color='red', width=3),
            hovertemplate='Regresión<br>%{xaxis.title.text}: %{x:.2f}<br>%{yaxis.title.text}: %{y:.2f}<extra></extra>'
        ))
        
        # Actualizar layout
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            height=520,
            hovermode='closest',
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1
            ),
            plot_bgcolor='rgba(240,240,240,0.5)',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        )
        
        return fig
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Adopción de IA vs. Crecimiento de Ingresos")
        fig1 = create_interactive_density_plot(
            df['ai_adoption_rate'],
            df['revenue_growth_percent'],
            'Tasa de Adopción de IA vs. Crecimiento de Ingresos (%) con Densidad',
            'Blues',
            'Tasa de Adopción de IA (%)',
            'Crecimiento de Ingresos (%)'
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Adopción de IA vs. Reducción de Costos")
        fig2 = create_interactive_density_plot(
            df['ai_adoption_rate'],
            df['cost_reduction_percent'],
            'Tasa de Adopción de IA vs. Reducción de Costos (%) con Densidad',
            'Oranges',
            'Tasa de Adopción de IA (%)',
            'Reducción de Costos (%)'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Adopción de IA vs. Cambio de Productividad")
    fig3 = create_interactive_density_plot(
        df['ai_adoption_rate'],
        df['productivity_change_percent'],
        'Distribución de Puntos: Tasa de Adopción de IA vs. Cambio de Productividad',
        'Viridis',
        'Tasa de Adopción de IA (%)',
        'Cambio de Productividad (%)'
    )
    st.plotly_chart(fig3, use_container_width=True)

# ==================== TAB 5: ROI y Tamaño de Empresa ====================
with tab5:
    st.header("ROI de IA por Industria y Tamaño de Empresa")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ROI de IA Promedio por Industria")
        roi_industry = df.groupby('industry', observed=False)['roi_ai'].mean().sort_values(ascending=False)
        
        fig = go.Figure(data=[
            go.Bar(
                x=roi_industry.values,
                y=roi_industry.index,
                orientation='h',
                marker=dict(color=roi_industry.values, colorscale='Blues', showscale=True, colorbar=dict(title="ROI"))
            )
        ])
        
        fig.update_layout(
            title='ROI de IA Promedio por Industria',
            xaxis_title='ROI de IA (Crecimiento de Ingresos + Reducción de Costos)',
            yaxis_title='Industria',
            height=550,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("ROI de IA por Tamaño de Empresa")
        company_size_order = ['Startup', 'SME', 'Enterprise']
        
        fig = px.box(df,
                    x='company_size',
                    y='roi_ai',
                    category_orders={'company_size': company_size_order},
                    title='ROI de IA por Tamaño de Empresa',
                    labels={
                        'company_size': 'Tamaño de Empresa',
                        'roi_ai': 'ROI de IA'
                    },
                    color='company_size',
                    color_discrete_sequence=px.colors.qualitative.Set2)
        
        fig.update_layout(height=550, hovermode='closest', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de ROI por industria
    st.subheader("Tabla: ROI por Industria")
    roi_table = df.groupby('industry', observed=False)[['revenue_growth_percent', 'cost_reduction_percent', 'roi_ai']].mean().sort_values('roi_ai', ascending=False)
    roi_table.columns = ['Crecimiento Ingresos (%)', 'Reducción Costos (%)', 'ROI de IA']
    st.dataframe(roi_table, use_container_width=True)

# ==================== TAB 6: Mapa Geográfico ====================
with tab6:
    st.header("🌐 Impacto Global de Adopción de IA")
    
    # Crear agregación de datos por país
    df_map = df.groupby('country', observed=False).agg({
        'productivity_change_percent': 'mean',
        'ai_adoption_rate': 'mean',
        'revenue_growth_percent': 'mean',
        'cost_reduction_percent': 'mean'
    }).reset_index()
    
    # Selector de métrica a visualizar
    col1, col2 = st.columns([3, 1])
    with col2:
        metric = st.selectbox(
            "Selecciona la métrica:",
            ['Cambio de Productividad (%)', 'Tasa de Adopción de IA (%)', 'Crecimiento de Ingresos (%)', 'Reducción de Costos (%)'],
            index=0
        )
    
    # Mapear selección a columna
    metric_map = {
        'Cambio de Productividad (%)': 'productivity_change_percent',
        'Tasa de Adopción de IA (%)': 'ai_adoption_rate',
        'Crecimiento de Ingresos (%)': 'revenue_growth_percent',
        'Reducción de Costos (%)': 'cost_reduction_percent'
    }
    
    selected_metric = metric_map[metric]
    
    if not df_map.empty:
        # Crear mapa coroplético
        fig = px.choropleth(
            df_map,
            locations="country",
            locationmode='country names',
            color=selected_metric,
            hover_name="country",
            hover_data={
                'country': False,
                'ai_adoption_rate': ':.1f',
                'productivity_change_percent': ':.1f',
                'revenue_growth_percent': ':.1f',
                'cost_reduction_percent': ':.1f'
            },
            labels={
                'productivity_change_percent': 'Cambio Productividad (%)',
                'ai_adoption_rate': 'Adopción IA (%)',
                'revenue_growth_percent': 'Crecimiento Ingresos (%)',
                'cost_reduction_percent': 'Reducción Costos (%)'
            },
            color_continuous_scale="Viridis",
            template="plotly_dark"
        )
        
        fig.update_layout(
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            geo=dict(bgcolor='rgba(0,0,0,0)', lakecolor='#0e1117'),
            height=700,
            title_text=f"Mapa Global: {metric}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de datos por país
        st.subheader("📊 Datos por País")
        df_map_display = df_map.sort_values(selected_metric, ascending=False)
        df_map_display.columns = ['País', 'Cambio Productividad (%)', 'Adopción IA (%)', 'Crecimiento Ingresos (%)', 'Reducción Costos (%)']
        st.dataframe(df_map_display, use_container_width=True)
    else:
        st.warning("⚠️ No hay datos disponibles para mostrar el mapa.")

# ==================== Información General ====================
st.sidebar.title("📋 Información General")
st.sidebar.metric("Total de Registros", len(df))
st.sidebar.metric("Número de Empresas", df['company_id'].nunique())
st.sidebar.metric("Número de Industrias", df['industry'].nunique())
st.sidebar.metric("Rango de Años", f"{df['survey_year'].min()} - {df['survey_year'].max()}")

st.sidebar.divider()
st.sidebar.markdown("""
### 📊 Sobre este Dashboard
Este dashboard analiza el impacto de la adopción de IA en el desempeño empresarial 
mediante los siguientes KPIs:

- **Crecimiento de Ingresos**: Cambio porcentual en ingresos
- **Reducción de Costos**: Ahorro operacional logrado
- **Cambio de Productividad**: Mejora en la eficiencia
- **ROI de IA**: Suma de crecimiento de ingresos y reducción de costos

""")
