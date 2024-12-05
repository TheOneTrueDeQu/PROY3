import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from scipy.stats import percentileofscore
import tensorflow as tf
import numpy as np

# Carga de datos
df = pd.read_csv('datos_reemplazados_no_estesi.csv')
df['PERCENTIL_PUNT_GLOBAL'] = df['PUNT_GLOBAL'].apply(lambda x: percentileofscore(df['PUNT_GLOBAL'], x))

# Carga del modelo predictivo
modelo = tf.keras.models.load_model('modelo_saber11.h5', compile=False)

def create_app():
    # Crear la aplicación Dash
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # Contenido del tablero
    app.layout = html.Div([
        dbc.Row([
            dbc.Col(html.H2("Análisis de Factores que afectan el Puntaje en Prueba Saber 11"), width=12),
        ]),
        
        # Gráficos principales
        dbc.Row([
            dbc.Col([
                html.H4("Seleccione la característica para comparar con el puntaje global"),
                dcc.Dropdown(
                    id='feature-dropdown',
                    options=[
                        {'label': col, 'value': col} for col in df.columns if col != 'PUNT_GLOBAL'
                    ],
                    value='PUNT_INGLES'  # Valor por defecto
                ),
                dcc.Graph(id='comparison-graph')
            ], width=6),

            dbc.Col([
                html.H4("Distribución del puntaje global por categoría"),
                dcc.Dropdown(
                    id='categorical-dropdown',
                    options=[
                        {'label': 'Colegio Bilingüe', 'value': 'COLE_BILINGUE'},
                        {'label': 'Educación de la Madre', 'value': 'FAMI_EDUCACIONMADRE'},
                        {'label': 'Estrato de Vivienda', 'value': 'FAMI_ESTRATOVIVIENDA'},
                    ],
                    value='COLE_BILINGUE'  # Valor por defecto
                ),
                dcc.Graph(id='distribution-graph')
            ], width=6),
        ]),

        # Segmentación
        dbc.Row([
            dbc.Col([
                html.H4("Segmentación por factor"),
                dcc.Dropdown(
                    id='segment-dropdown',
                    options=[
                        {'label': 'Estrato de Vivienda', 'value': 'FAMI_ESTRATOVIVIENDA'},
                        {'label': 'Colegio Bilingüe', 'value': 'COLE_BILINGUE'},
                        {'label': 'Género', 'value': 'ESTU_GENERO'},
                    ],
                    value='FAMI_ESTRATOVIVIENDA'  # Valor por defecto
                ),
                dcc.Graph(id='segment-graph')
            ], width=12),
        ]),

        # Sección Modelo Predictivo
        dbc.Row([
            dbc.Col(html.H4("Modelo Predictivo"), width=12),
        ]),

        # Dropdowns para predicción
        dbc.Row([
            dbc.Col([
                html.Label("Colegio Bilingüe:"),
                dcc.Dropdown(
                    id='input-COLE_BILINGUE',
                    options=[{'label': "No", 'value': 1}, {'label': "Sí", 'value': 0}],
                    placeholder="Seleccione una opción"
                ),
                html.Label("Calendario del Colegio:"),
                dcc.Dropdown(
                    id='input-COLE_CALENDARIO',
                    options=[{'label': "A", 'value': 0}, {'label': "B", 'value': 1}, {'label': "OTRO", 'value': 2}],
                    placeholder="Seleccione una opción"
                ),
                html.Label("Caracter del Colegio:"),
                dcc.Dropdown(
                    id='input-COLE_CARACTER',
                    options=[
                        {'label': "Académico", 'value': 0}, {'label': "No Aplica", 'value': 1},
                        {'label': "Técnico", 'value': 2}, {'label': "Otro", 'value': 3}
                    ],
                    placeholder="Seleccione una opción"
                ),
                html.Label("Tipo de Jornada:"),
                dcc.Dropdown(
                    id='input-COLE_JORNADA',
                    options=[
                        {'label': "Completa", 'value': 5}, {'label': "Mañana", 'value': 4}, 
                        {'label': "Noche", 'value': 3}, {'label': "Sabatina", 'value': 2},
                        {'label': "Tarde", 'value': 1}, {'label': "Única", 'value': 0}
                    ],
                    placeholder="Seleccione una opción"
                ),
                html.Label("Naturaleza del Colegio:"),
                dcc.Dropdown(
                    id='input-COLE_NATURALEZA',
                    options=[{'label': "No Oficial", 'value': 0}, {'label': "Oficial", 'value': 1}],
                    placeholder="Seleccione una opción"
                ),
            ], width=6),

            dbc.Col([
                html.Label("Género del Estudiante:"),
                dcc.Dropdown(
                    id='input-ESTU_GENERO',
                    options=[
                        {'label': "Masculino", 'value': 0}, {'label': "Femenino", 'value': 1},
                        {'label': "No Especifica", 'value': 2}
                    ],
                    placeholder="Seleccione una opción"
                ),
                html.Label("Educación de la Madre:"),
                dcc.Dropdown(
                    id='input-FAMI_EDUCACIONMADRE',
                    options=[
                        {'label': "Profesional Completa", 'value': 11}, {'label': "Profesional Incompleta", 'value': 10},
                        {'label': "Ninguno", 'value': 9}, {'label': "No Aplica", 'value': 8}, {'label': "No Sabe", 'value': 7},
                        {'label': "Postgrado", 'value': 6}, {'label': "Primaria Completa", 'value': 5},
                        {'label': "Primaria Incompleta", 'value': 4}, {'label': "Bachillerato Completo", 'value': 3},
                        {'label': "Bachillerato Incompleto", 'value': 2}, {'label': "Técnico Completo", 'value': 1},
                        {'label': "Técnico Incompleto", 'value': 0}
                    ],
                    placeholder="Seleccione una opción"
                ),
                html.Label("Educación deL padre:"),
                dcc.Dropdown(
                    id='input-FAMI_EDUCACIONPADRE',
                    options=[
                        {'label': "Profesional Completa", 'value': 11}, {'label': "Profesional Incompleta", 'value': 10},
                        {'label': "Ninguno", 'value': 9}, {'label': "No Aplica", 'value': 8}, {'label': "No Sabe", 'value': 7},
                        {'label': "Postgrado", 'value': 6}, {'label': "Primaria Completa", 'value': 5},
                        {'label': "Primaria Incompleta", 'value': 4}, {'label': "Bachillerato Completo", 'value': 3},
                        {'label': "Bachillerato Incompleto", 'value': 2}, {'label': "Técnico Completo", 'value': 1},
                        {'label': "Técnico Incompleto", 'value': 0}
                    ],
                    placeholder="Seleccione una opción"
                ),
                html.Label("Estrato de Vivienda:"),
                dcc.Dropdown(
                    id='input-FAMI_ESTRATOVIVIENDA',
                    options=[
                        {'label': "Estrato 1", 'value': 6}, {'label': "Estrato 2", 'value': 5}, 
                        {'label': "Estrato 3", 'value': 4}, {'label': "Estrato 4", 'value': 3},
                        {'label': "Estrato 5", 'value': 2}, {'label': "Estrato 6", 'value': 1},
                        {'label': "Sin Estrato", 'value': 0}
                    ],
                    placeholder="Seleccione una opción"
                ),
                html.Label("Tiene Carro?:"),
                dcc.Dropdown(
                    id='input-FAMI_TIENEAUTOMOVIL',
                    options=[{'label': "No", 'value': 1}, {'label': "Sí", 'value': 0}],
                    placeholder="Seleccione una opción"
                ),
                html.Label("Tiene Computador?:"),
                dcc.Dropdown(
                    id='input-FAMI_TIENECOMPUTADOR',
                    options=[{'label': "No", 'value': 1}, {'label': "Sí", 'value': 0}],
                    placeholder="Seleccione una opción"
                ),
                html.Label("tiene internet?:"),
                dcc.Dropdown(
                    id='input-FAMI_TIENEINTERNET',
                    options=[{'label': "No", 'value': 1}, {'label': "Sí", 'value': 0}],
                    placeholder="Seleccione una opción"
                ),
                html.Label("tiene lavadora?:"),
                dcc.Dropdown(
                    id='input-FAMI_TIENELAVADORA',
                    options=[{'label': "No", 'value': 1}, {'label': "Sí", 'value': 0}],
                    placeholder="Seleccione una opción"
                ),
            ], width=6),
        ]),

        dbc.Row([
            dbc.Col([
                html.Button('Predecir', id='boton-prediccion', n_clicks=0),
                html.Div(id='output-prediccion', style={'marginTop': 20})
            ], width=12),
        ])
    ])

    # Callback para gráficos
    @app.callback(
        [Output('comparison-graph', 'figure'),
         Output('distribution-graph', 'figure'),
         Output('segment-graph', 'figure')],
        [Input('feature-dropdown', 'value'),
         Input('categorical-dropdown', 'value'),
         Input('segment-dropdown', 'value')]
    )
    def update_graphs(selected_feature, selected_category, selected_segment):
        fig_comparison = px.scatter(df, x=selected_feature, y='PUNT_GLOBAL',color='PUNT_GLOBAL',size='PUNT_GLOBAL', title=f"Puntaje Global vs {selected_feature}")
        fig_distribution = px.box(df, x=selected_category, y='PUNT_GLOBAL', color=selected_category, title=f"Distribución por {selected_category}")
        fig_segment = px.histogram(df, x=selected_segment, title=f"Segmentación por {selected_segment}")
        return fig_comparison, fig_distribution, fig_segment

    # Callback para predicción
    @app.callback(
        Output('output-prediccion', 'children'),
        [Input('boton-prediccion', 'n_clicks')],
        [State('input-COLE_BILINGUE', 'value'),
         State('input-COLE_CALENDARIO', 'value'),
         State('input-COLE_CARACTER', 'value'),
         State('input-COLE_JORNADA', 'value'),
         State('input-COLE_NATURALEZA', 'value'),
         State('input-ESTU_GENERO', 'value'),
         State('input-FAMI_EDUCACIONMADRE', 'value'),
         State('input-FAMI_EDUCACIONPADRE', 'value'),
         State('input-FAMI_ESTRATOVIVIENDA', 'value'),
         State('input-FAMI_TIENEAUTOMOVIL', 'value'),
         State('input-FAMI_TIENECOMPUTADOR', 'value'),
         State('input-FAMI_TIENEINTERNET', 'value'),
         State('input-FAMI_TIENELAVADORA', 'value'),]
    )
    def hacer_prediccion(n_clicks, *values):
        if n_clicks > 0:
            try:
                datos = np.array(values).reshape(1, -1)
                prediccion = modelo.predict(datos)
                return html.Div([
                                    html.P(f'El puntaje general esperado es: {prediccion[0][5]*0.9:.2f}'),
                                    html.P(f'El puntaje de Inglés esperado es: {prediccion[0][0]*0.9:.2f}'),
                                    html.P(f'El puntaje de Matemáticas esperado es: {prediccion[0][1]*0.9:.2f}'),
                                    html.P(f'El puntaje de Sociales y Ciudadanas esperado es: {prediccion[0][2]*0.9:.2f}'),
                                    html.P(f'El puntaje de Ciencias Naturales esperado es: {prediccion[0][3]*0.9:.2f}'),
                                    html.P(f'El puntaje de Lectura Crítica esperado es: {prediccion[0][4]*0.9:.2f}')
                                ])
            except Exception as e:
                return f'Error al hacer la predicción: {e}'
        return 'Ingresa valores y presiona predecir.'

    return app

# Ejecutar la aplicación
if __name__ == '__main__':
    app = create_app()
    app.run_server(debug=True, host="0.0.0.0", port=8050)

