import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from scipy.stats import percentileofscore

# Carga de datos
df = pd.read_csv('datos_reemplazados_no_estesi.csv')
df['PERCENTIL_PUNT_GLOBAL'] = df['PUNT_GLOBAL'].apply(lambda x: percentileofscore(df['PUNT_GLOBAL'], x))

def create_app():
    # Crear la aplicación Dash
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # Contenido del tablero
    app.layout = html.Div([
        dbc.Row([
            dbc.Col(html.H2("Análisis de Factores que afectan el Puntaje en Prueba Saber 11"), width=12),
        ]),
        
        dbc.Row([
            # Selector para características a analizar
            dbc.Col([
                html.H4("Seleccione la característica para comparar con el puntaje global"),
                dcc.Dropdown(
                    id='feature-dropdown',
                    options=[
                        {'label': col, 'value': col} for col in df.columns if col != 'PUNT_GLOBAL'
                    ],
                     
                ),
                dcc.Graph(id='comparison-graph')
            ], width=6),

            # Gráfico de distribución
            dbc.Col([
                html.H4("Distribución del puntaje global por categoría"),
                dcc.Dropdown(
                    id='categorical-dropdown',
                    options=[
                        {'label': 'Educación de la Madre', 'value': 'FAMI_EDUCACIONMADRE'},
                        {'label': 'Educación del Padre', 'value': 'FAMI_EDUCACIONPADRE'},
                        {'label': 'Tiene Internet', 'value': 'FAMI_TIENEINTERNET'},
                        {'label': 'Colegio Bilingue', 'value': 'COLE_BILINGUE'},
                        {'label': 'Colegio Calendario', 'value': 'COLE_CALENDARIO'},
                        {'label': 'Caracter del Colegio', 'value': 'COLE_CARACTER'},
                        {'label': 'Tipo de Jornada', 'value': 'COLE_JORNADA'},
                        {'label': 'Género', 'value': 'ESTU_GENERO'},
                        {'label': 'Estrato', 'value': 'FAMI_ESTRATOVIVIENDA'},
                        {'label': 'Tiene Automóvil', 'value': 'FAMI_TIENEAUTOMOVIL'},
                        {'label': 'Tiene Computador', 'value': 'FAMI_TIENECOMPUTADOR'},
                        {'label': 'Tiene Internet', 'value': 'FAMI_TIENEINTERNET'},
                    ],
                    placeholder="Seleccione una categoría"
                ),
                dcc.Graph(id='distribution-graph')
            ], width=6),
        ]),

        dbc.Row([
            # Segmentación
            dbc.Col([
                html.H4("Segmentación por factor"),
                dcc.Dropdown(
                    id='segment-dropdown',
                    options=[
                        {'label': 'Educación de la Madre', 'value': 'FAMI_EDUCACIONMADRE'},
                        {'label': 'Educación del Padre', 'value': 'FAMI_EDUCACIONPADRE'},
                        {'label': 'Tiene Internet', 'value': 'FAMI_TIENEINTERNET'},
                        {'label': 'Colegio Bilingue', 'value': 'COLE_BILINGUE'},
                        {'label': 'Colegio Calendario', 'value': 'COLE_CALENDARIO'},
                        {'label': 'Caracter del Colegio', 'value': 'COLE_CARACTER'},
                        {'label': 'Tipo de Jornada', 'value': 'COLE_JORNADA'},
                        {'label': 'Género', 'value': 'ESTU_GENERO'},
                        {'label': 'Estrato', 'value': 'FAMI_ESTRATOVIVIENDA'},
                        {'label': 'Tiene Automóvil', 'value': 'FAMI_TIENEAUTOMOVIL'},
                        {'label': 'Tiene Computador', 'value': 'FAMI_TIENECOMPUTADOR'},
                        {'label': 'Tiene Internet', 'value': 'FAMI_TIENEINTERNET'},
                    ],
                    placeholder="Seleccione un segmento"
                ),
                dcc.Graph(id='segment-graph')
            ], width=12),
        ])
    ])

    # Callback para actualizar gráficos
    @app.callback(
        [Output('comparison-graph', 'figure'),
         Output('distribution-graph', 'figure'),
         Output('segment-graph', 'figure')],
        [Input('feature-dropdown', 'value'),
         Input('categorical-dropdown', 'value'),
         Input('segment-dropdown', 'value')]
    )
    def update_graphs(selected_feature, selected_category, selected_segment):
        # Comparación entre puntaje global y la característica seleccionada
        fig_comparison = px.scatter(
            df, x=selected_feature, y='PUNT_GLOBAL',
            title=f"Puntaje Global vs {selected_feature}",
            labels={selected_feature: selected_feature, 'PUNT_GLOBAL': 'Puntaje Global'},
            color='PUNT_GLOBAL',
            size='PUNT_GLOBAL'
        )

        # Distribución del puntaje global por categoría
        if selected_category:
            fig_distribution = px.box(
                df, x=selected_category, y='PUNT_GLOBAL',
                title=f"Distribución del Puntaje Global por {selected_category}",
                labels={selected_category: selected_category, 'PUNT_GLOBAL': 'Puntaje Global'},
                color=selected_category
            )
        else:
            fig_distribution = px.box(pd.DataFrame(), title="Seleccione una categoría")

        # Segmentación por factor
        if selected_segment:
            fig_segment = px.histogram(
                df, x=selected_segment, color='PUNT_GLOBAL',
                title=f"Segmentación del Puntaje Global por {selected_segment}",
                labels={selected_segment: selected_segment, 'PUNT_GLOBAL': 'Puntaje Global'}
            )
        else:
            fig_segment = px.histogram(pd.DataFrame(), title="Seleccione un segmento")

        return fig_comparison, fig_distribution, fig_segment

    return app

# Ejecutar la aplicación
if __name__ == '__main__':
    app = create_app()
    app.run_server(debug=True, host="127.0.0.1", port=8050)