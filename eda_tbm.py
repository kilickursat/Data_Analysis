import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata

# Helper functions (copied from the original code)
def calculate_torque(row):
    arbeitsdruck, drehzahl = row['Arbeitsdruck'], row['Drehzahl']
    drehmomentkonst, n1 = 0.14376997, 25.7
    return round(arbeitsdruck * drehmomentkonst * (n1 / drehzahl if drehzahl > n1 else 1), 2)

def calculate_penetration_rate(row):
    return round(row['Advance_Rate'] / row['Drehzahl'] if row['Drehzahl'] != 0 else 0, 4)

# Streamlit app
def main():
    st.title("TBM Data Analysis App")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=';', decimal=',', na_values=['', 'NA', 'N/A', 'nan', 'NaN'], keep_default_na=True)
        df = df.replace([np.inf, -np.inf], np.nan)

        # Rename columns for clarity
        df = df.rename(columns={
            df.columns[13]: 'Arbeitsdruck',
            df.columns[7]: 'Drehzahl',
            df.columns[30]: 'Advance_Rate',
            df.columns[17]: 'Thrust_Force',
            df.columns[27]: 'Chainage',
            df.columns[2]: 'Relativzeit',
            df.columns[28]: 'Position_Grad'
        })

        df['Calculated_Torque'] = df.apply(calculate_torque, axis=1)
        df['Penetration_Rate'] = df.apply(calculate_penetration_rate, axis=1)

        st.write("Data loaded successfully. Shape:", df.shape)

        # Sidebar for navigation
        page = st.sidebar.selectbox("Choose a page", ["Data Overview", "Visualizations", "Statistical Analysis"])

        if page == "Data Overview":
            st.header("Data Overview")
            st.write(df.head())
            st.write(df.describe())
            st.write(df.info())

        elif page == "Visualizations":
            st.header("Visualizations")
            viz_type = st.selectbox("Choose a visualization", [
                "Machine Features",
                "Correlation Heatmap",
                "Animated Polar Plot",
                "3D Scatter Plot",
                "Animated Bubble Chart",
                "Density Heatmap",
                "3D Surface Plot"
            ])

            if viz_type == "Machine Features":
                visualize_machine_features(df)
            elif viz_type == "Correlation Heatmap":
                create_correlation_heatmap(df)
            elif viz_type == "Animated Polar Plot":
                create_animated_polar_plot(df)
            elif viz_type == "3D Scatter Plot":
                create_3d_scatter_plot(df)
            elif viz_type == "Animated Bubble Chart":
                create_animated_bubble_chart(df)
            elif viz_type == "Density Heatmap":
                create_density_heatmap(df)
            elif viz_type == "3D Surface Plot":
                create_3d_surface_plot(df)

        elif page == "Statistical Analysis":
            st.header("Statistical Analysis")
            analysis_type = st.selectbox("Choose an analysis", [
                "OLS Regression",
                "Statistical Summary"
            ])

            if analysis_type == "OLS Regression":
                x_col = st.selectbox("Select X variable", df.columns)
                y_col = st.selectbox("Select Y variable", df.columns)
                perform_ols_regression(df, x_col, y_col)
            elif analysis_type == "Statistical Summary":
                create_statistical_summary(df)

# Visualization functions (modified for Streamlit)
def visualize_machine_features(df):
    features = ['Arbeitsdruck', 'Penetration_Rate', 'Calculated_Torque', 'Advance_Rate', 'Drehzahl', 'Thrust_Force']
    fig = make_subplots(rows=len(features), cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=features)

    for i, feature in enumerate(features, start=1):
        fig.add_trace(go.Scatter(x=df['Relativzeit'], y=df[feature], name=feature), row=i, col=1)
        fig.update_yaxes(title_text=feature, row=i, col=1)

    fig.update_layout(height=1200, title_text="Features vs Time", showlegend=False)
    fig.update_xaxes(title_text="Time", row=len(features), col=1)
    st.plotly_chart(fig)

def create_correlation_heatmap(df):
    features = ['Arbeitsdruck', 'Drehzahl', 'Advance_Rate', 'Thrust_Force', 'Chainage', 'Position_Grad', 'Calculated_Torque', 'Penetration_Rate']
    corr_matrix = df[features].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax)
    plt.title('Correlation Heatmap of Selected Parameters')
    st.pyplot(fig)

def create_animated_polar_plot(df):
    pressure_column = 'Arbeitsdruck'
    rpm_column = 'Drehzahl'
    chainage_column = 'Chainage'

    # Convert to numeric and handle errors
    for col in [pressure_column, rpm_column, chainage_column]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove problematic rows
    df = df.dropna(subset=[pressure_column, rpm_column, chainage_column])

    if df.empty:
        st.write("No valid data points remaining after removing NaN values.")
        return

    # Create frames for each unique chainage
    frames = []
    for chainage in df[chainage_column].unique():
        chainage_data = df[df[chainage_column] == chainage]

        # Normalize time to 0-360 range for each chainage
        time_normalized = np.linspace(0, 360, len(chainage_data))

        # Calculate point density for pressure and RPM
        xy_pressure = np.vstack([time_normalized, chainage_data[pressure_column]])
        z_pressure = gaussian_kde(xy_pressure)(xy_pressure)

        xy_rpm = np.vstack([time_normalized, chainage_data[rpm_column]])
        z_rpm = gaussian_kde(xy_rpm)(xy_rpm)

        frame = go.Frame(
            data=[
                go.Scatterpolar(
                    r=chainage_data[pressure_column],
                    theta=time_normalized,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=z_pressure,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(
                            title='Pressure Density',
                            titleside='right',
                            x=1.1
                        )
                    ),
                    name='Pressure'
                ),
                go.Scatterpolar(
                    r=chainage_data[rpm_column],
                    theta=time_normalized,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=z_rpm,
                        colorscale='Plasma',
                        showscale=True,
                        colorbar=dict(
                            title='RPM Density',
                            titleside='right',
                            x=1.2
                        )
                    ),
                    name='RPM'
                )
            ],
            name=str(chainage)
        )
        frames.append(frame)

    # Create initial frame data
    initial_chainage = df[chainage_column].min()
    initial_data = df[df[chainage_column] == initial_chainage]
    initial_time = np.linspace(0, 360, len(initial_data))

    initial_xy_pressure = np.vstack([initial_time, initial_data[pressure_column]])
    initial_z_pressure = gaussian_kde(initial_xy_pressure)(initial_xy_pressure)

    initial_xy_rpm = np.vstack([initial_time, initial_data[rpm_column]])
    initial_z_rpm = gaussian_kde(initial_xy_rpm)(initial_xy_rpm)

    # Create the figure with frames
    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=initial_data[pressure_column],
                theta=initial_time,
                mode='markers',
                marker=dict(
                    size=5,
                    color=initial_z_pressure,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title='Pressure Density',
                        titleside='right',
                        x=1.1
                    )
                ),
                name='Pressure'
            ),
            go.Scatterpolar(
                r=initial_data[rpm_column],
                theta=initial_time,
                mode='markers',
                marker=dict(
                    size=5,
                    color=initial_z_rpm,
                    colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(
                        title='RPM Density',
                        titleside='right',
                        x=1.2
                    )
                ),
                name='RPM'
            )
        ],
        frames=frames
    )

    # Update layout
    fig.update_layout(
        title='Animated Polar Plot: Pressure and RPM Distribution Over Chainage',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(df[pressure_column].max(), df[rpm_column].max()) * 1.1],
                showline=False,
                showgrid=True,
                gridcolor='lightgrey',
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickmode='array',
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'],
                direction='clockwise',
                rotation=90,
                gridcolor='lightgrey'
            )
        ),
        showlegend=True,
        legend=dict(x=0.5, y=-0.1, orientation='h'),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[dict(label='Play',
                          method='animate',
                          args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]),
                     dict(label='Pause',
                          method='animate',
                          args=[[None], dict(frame=dict(duration=0, redraw=True), mode='immediate')])]
        )],
        sliders=[dict(
            steps=[
                dict(
                    method='animate',
                    args=[[str(chainage)], dict(mode='immediate', frame=dict(duration=100, redraw=True))],
                    label=str(chainage)
                )
                for chainage in df[chainage_column].unique()
            ],
            active=0,
            currentvalue=dict(prefix='Chainage: ', visible=True),
            pad=dict(t=50)
        )],
        height=800,
        margin=dict(r=100, t=100, b=100, l=100)
    )

    st.plotly_chart(fig)

def create_3d_scatter_plot(df):
    fig = px.scatter_3d(df, x='Arbeitsdruck', y='Penetration_Rate', z='Calculated_Torque',
                        color='Advance_Rate', size='Thrust_Force',
                        hover_data=['Drehzahl', 'Position_Grad'],
                        labels={'Arbeitsdruck': 'Working Pressure',
                                'Penetration_Rate': 'Penetration Rate',
                                'Calculated_Torque': 'Calculated Torque',
                                'Advance_Rate': 'Advance Rate',
                                'Thrust_Force': 'Thrust Force'},
                        title='3D Visualization of Key Parameters')
    fig.update_layout(scene=dict(xaxis_title='Working Pressure',
                                 yaxis_title='Penetration Rate',
                                 zaxis_title='Calculated Torque'))
    st.plotly_chart(fig)

def create_animated_bubble_chart(df):
    fig = px.scatter(df, x='Arbeitsdruck', y='Penetration_Rate',
                     animation_frame='Chainage', animation_group='Position_Grad',
                     size='Thrust_Force', color='Calculated_Torque', hover_name='Drehzahl',
                     log_x=True, size_max=55, range_x=[df['Arbeitsdruck'].min(), df['Arbeitsdruck'].max()],
                     range_y=[df['Penetration_Rate'].min(), df['Penetration_Rate'].max()],
                     title='Animated Bubble Chart: Parameter Evolution Along Chainage')
    st.plotly_chart(fig)

def create_density_heatmap(df):
    x = df['Arbeitsdruck']
    y = df['Penetration_Rate']

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    fig = go.Figure(data=go.Scatter(
        x=x, y=y, mode='markers',
        marker=dict(size=3, color=z, colorscale='Viridis', showscale=True)
    ))

    fig.update_layout(
        title='Density Heatmap: Working Pressure vs Penetration Rate',
        xaxis_title='Working Pressure',
        yaxis_title='Penetration Rate',
        coloraxis_colorbar=dict(title='Density')
    )
    st.plotly_chart(fig)

def create_3d_surface_plot(df):
    x = df['Arbeitsdruck']
    y = df['Penetration_Rate']
    z = df['Calculated_Torque']

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata((x, y), z, (xi, yi), method='linear')

    fig = go.Figure(data=[go.Surface(x=xi, y=yi, z=zi)])
    fig.update_layout(
        title='3D Surface Plot: Working Pressure, Penetration Rate, and Calculated Torque',
        scene=dict(
            xaxis_title='Working Pressure',
            yaxis_title='Penetration Rate',
            zaxis_title='Calculated Torque'
        ),
        autosize=False,
        width=800,
        height=800,
        margin=dict(l=65, r=50, b=65, t=90)
    )
    st.plotly_chart(fig)

def create_animated_bubble_chart(df):
    fig = px.scatter(df, x='Arbeitsdruck', y='Penetration_Rate',
                     animation_frame='Chainage', animation_group='Position_Grad',
                     size='Thrust_Force', color='Calculated_Torque', hover_name='Drehzahl',
                     log_x=True, size_max=55, range_x=[df['Arbeitsdruck'].min(), df['Arbeitsdruck'].max()],
                     range_y=[df['Penetration_Rate'].min(), df['Penetration_Rate'].max()],
                     title='Animated Bubble Chart: Parameter Evolution Along Chainage')
    st.plotly_chart(fig)

def create_density_heatmap(df):
    x = df['Arbeitsdruck']
    y = df['Penetration_Rate']

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    fig = go.Figure(data=go.Scatter(
        x=x, y=y, mode='markers',
        marker=dict(size=3, color=z, colorscale='Viridis', showscale=True)
    ))

    fig.update_layout(
        title='Density Heatmap: Working Pressure vs Penetration Rate',
        xaxis_title='Working Pressure',
        yaxis_title='Penetration Rate',
        coloraxis_colorbar=dict(title='Density')
    )
    st.plotly_chart(fig)

def create_3d_surface_plot(df):
    x = df['Arbeitsdruck']
    y = df['Penetration_Rate']
    z = df['Calculated_Torque']

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(xi, yi)

    Z = griddata((x, y), z, (X, Y), method='cubic')

    fig = go.Figure(data=[go.Surface(x=xi, y=yi, z=Z, colorscale='Viridis')])

    fig.update_layout(
        title='3D Surface Plot: Working Pressure, Penetration Rate, and Calculated Torque',
        scene=dict(
            xaxis_title='Working Pressure',
            yaxis_title='Penetration Rate',
            zaxis_title='Calculated Torque'
        ),
        width=800,
        height=800,
    )
    st.plotly_chart(fig)

def perform_ols_regression(df, x_col, y_col):
    X = sm.add_constant(df[x_col])
    model = sm.OLS(df[y_col], X).fit()
    st.write(f"OLS Regression Results ({y_col} vs {x_col}):")
    st.write(model.summary())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[x_col], df[y_col], alpha=0.5)
    ax.plot(df[x_col], model.predict(X), color='red', linewidth=2)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'{y_col} vs {x_col}')
    equation = f'y = {model.params[1]:.4f}x + {model.params[0]:.4f}'
    ax.text(0.05, 0.95, f'R² = {model.rsquared:.4f}\n{equation}',
             transform=ax.transAxes, verticalalignment='top')
    st.pyplot(fig)

def create_statistical_summary(df):
    features = ['Arbeitsdruck', 'Penetration_Rate', 'Calculated_Torque', 'Advance_Rate', 'Drehzahl', 'Thrust_Force']
    summary = df[features].describe()
    st.write(summary)

if __name__ == "__main__":
    main()












