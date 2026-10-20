import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output

# -----------------------------
# 0. FILE PATHS — adjust these if your filenames differ
# -----------------------------
PATHS = {
    "mpnet": {
        "umap": "data/processed/cams_long_umap_coded.csv",
        "sim": "data/processed/participant_similarity_results.csv",
    },
    "mentalbert": {
        "umap": "data/processed/cams_long_umap_mentalbert_coded.csv",
        "sim": "data/processed/participant_similarity_results_mentalbert.csv",
    },
}

MODEL_LABELS = {
    "mpnet": "MPNet (general)",
    "mentalbert": "MentalBERT (domain-specific)",
}

# "Color by" options: value -> (df column, legend title)
COLOR_OPTIONS = {
    "type": ("type", "Type"),
    "drivers_theme": ("drivers_theme", "Drivers Theme"),
    "rfd_code": ("rfd_code", "RFD Code"),
}

display_map = {"drivers": "Drivers", "rfd": "RFD"}

# Qualitative palette with enough distinct colors for the largest coding scheme (~24 categories)
QUAL_PALETTE = px.colors.qualitative.Dark24


def _is_uncoded_label(label):
    low = str(label).lower()
    return (
        "uncoded" in low
        or "uncodeable" in low
        or "uncertain" in low
        or low.startswith("999")
        or low.startswith("998")
    )


def sorted_categories(values):
    """Put Uncoded / 998 / 999 'other' buckets at the end of the legend."""
    uniq = sorted(set(values))
    return sorted(uniq, key=lambda v: (_is_uncoded_label(v), str(v)))


def build_color_map(categories):
    color_map = {}
    palette_idx = 0
    for cat in categories:
        if _is_uncoded_label(cat):
            color_map[cat] = "#B0B0B0"  # grey for uncoded / other-uncertain buckets
        else:
            color_map[cat] = QUAL_PALETTE[palette_idx % len(QUAL_PALETTE)]
            palette_idx += 1
    return color_map


# -----------------------------
# 1. LOAD + CLEAN A SINGLE MODEL'S DATA
# -----------------------------
def load_model_data(umap_path, sim_path):
    df = pd.read_csv(umap_path)
    sim_df = pd.read_csv(sim_path)

    df["text"] = df["text"].fillna("").astype(str)
    df["type"] = df["type"].fillna("unknown").astype(str).str.strip().str.lower()
    df["row_id"] = df["row_id"].astype(int)
    df["id"] = df["id"].astype(str)

    sim_df["id"] = sim_df["id"].astype(str)

    type_map = {
        "driver": "drivers",
        "drivers": "drivers",
        "rfd": "rfd",
        "reason for dying": "rfd",
        "reasons for dying": "rfd",
    }
    df["type"] = df["type"].replace(type_map)
    df["type_label"] = df["type"].map(display_map).fillna(df["type"])

    # Coding columns — fall back gracefully if this CSV predates the coding merge
    for col in ["drivers_theme", "drivers_code", "rfd_code"]:
        if col not in df.columns:
            df[col] = "Uncoded"
        df[col] = df[col].fillna("Uncoded").astype(str).str.strip()
        df[col] = df[col].replace({"": "Uncoded", "nan": "Uncoded"})

    df["text_preview"] = df["text"].str.slice(0, 80)
    df.loc[df["text"].str.len() > 80, "text_preview"] += "..."

    df = df.set_index("row_id", drop=False)

    # Precompute a stable color map per "color by" option, based on the FULL
    # dataset (not a filtered subset), so colors/legend don't shift when the
    # user filters by type or participant.
    color_maps = {}
    for value, (col, _title) in COLOR_OPTIONS.items():
        if col == "type":
            color_maps[value] = {"drivers": "#1f77b4", "rfd": "#d62728"}
        else:
            cats = sorted_categories(df[col].unique())
            color_maps[value] = build_color_map(cats)

    return df, sim_df, color_maps


# Load both datasets once at startup
datasets = {}
for model_key, paths in PATHS.items():
    df_m, sim_m, color_maps_m = load_model_data(paths["umap"], paths["sim"])
    datasets[model_key] = {"df": df_m, "sim": sim_m, "color_maps": color_maps_m}


# -----------------------------
# 2. HELPER FUNCTIONS
# -----------------------------
def make_empty_umap_figure(message="No data available for the selected filters."):
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=16)
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
    )
    return fig


def make_umap_figure(filtered_df, full_df, color_by):
    if filtered_df.empty:
        return make_empty_umap_figure()

    color_col, legend_title = COLOR_OPTIONS[color_by]
    color_map = full_df.attrs["color_maps"][color_by]
    category_order = list(color_map.keys())

    x_range = [full_df["x"].min(), full_df["x"].max()]
    y_range = [full_df["y"].min(), full_df["y"].max()]
    z_range = [full_df["z"].min(), full_df["z"].max()]

    fig = px.scatter_3d(
        filtered_df,
        x="x", y="y", z="z",
        color=color_col,
        color_discrete_map=color_map,
        category_orders={color_col: category_order},
        hover_name="type_label",
        hover_data={
            "id": True,
            "text_preview": True,
            "drivers_theme": True,
            "drivers_code": True,
            "rfd_code": True,
            "x": False, "y": False, "z": False,
            "row_id": False, "type": False, "type_label": False,
        },
        custom_data=["row_id"],
    )

    marker_size = 5 if len(filtered_df) > 20 else 8
    fig.update_traces(marker=dict(size=marker_size))

    if color_col == "type":
        fig.for_each_trace(
            lambda trace: trace.update(name=display_map.get(trace.name, trace.name))
        )

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis=dict(range=x_range, title="x", autorange=False),
            yaxis=dict(range=y_range, title="y", autorange=False),
            zaxis=dict(range=z_range, title="z", autorange=False),
            aspectmode="cube",
        ),
        legend=dict(title=legend_title, font=dict(size=12), title_font=dict(size=14), itemsizing="constant"),
        uirevision="fixed",
    )

    return fig


def make_overall_similarity_figure(similarity_df):
    mean_dd = similarity_df["driver_driver_similarity"].dropna().mean()
    mean_rr = similarity_df["rfd_rfd_similarity"].dropna().mean()
    mean_dr = similarity_df["driver_rfd_similarity"].dropna().mean()

    plot_df = pd.DataFrame({
        "comparison": ["Driver–Driver", "RFD–RFD", "Driver–RFD"],
        "similarity": [mean_dd, mean_rr, mean_dr],
    })

    fig = px.bar(
        plot_df, x="comparison", y="similarity", color="comparison",
        color_discrete_map={
            "Driver–Driver": "#1f77b4",
            "RFD–RFD": "#ff7f0e",
            "Driver–RFD": "#2ca02c",
        },
        text=plot_df["similarity"].round(2),
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        title="Overall Sample Mean Semantic Similarity",
        showlegend=False,
        height=240,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis_title="Mean Cosine Similarity",
        xaxis_title="",
        yaxis=dict(range=[0, 0.9]),
    )

    return fig


def make_participant_similarity_figure(selected_id, sim_df):
    if selected_id == "All":
        fig = go.Figure()
        fig.add_annotation(
            text="Select a participant to view.", x=0.5, y=0.5,
            xref="paper", yref="paper", showarrow=False, font=dict(size=14),
        )
        fig.update_layout(
            title="Participant-Level Semantic Similarity",
            height=400, margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        return fig

    row = sim_df[sim_df["id"] == selected_id]

    if row.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No similarity summary available for this participant.",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14),
        )
        fig.update_layout(
            title="Participant-Level Semantic Similarity",
            height=300, margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        return fig

    row = row.iloc[0]

    plot_df = pd.DataFrame({
        "comparison": ["Driver–Driver", "RFD–RFD", "Driver–RFD"],
        "similarity": [
            row["driver_driver_similarity"],
            row["rfd_rfd_similarity"],
            row["driver_rfd_similarity"],
        ],
    })

    fig = px.bar(
        plot_df, x="comparison", y="similarity", color="comparison",
        color_discrete_map={
            "Driver–Driver": "#1f77b4",
            "RFD–RFD": "#ff7f0e",
            "Driver–RFD": "#2ca02c",
        },
        text=plot_df["similarity"].round(2),
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        title=f"Participant {selected_id}: Semantic Similarity",
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis_title="Cosine Similarity",
        xaxis_title="",
    )

    return fig


# Attach each model's color maps onto its df via .attrs so make_umap_figure
# can look them up from whichever df (filtered or not) gets passed around
for model_key in datasets:
    datasets[model_key]["df"].attrs["color_maps"] = datasets[model_key]["color_maps"]

# Initial figures (default to mpnet)
initial_df = datasets["mpnet"]["df"]
initial_sim = datasets["mpnet"]["sim"]
initial_umap_fig = make_umap_figure(initial_df, initial_df, "type")
initial_participant_fig = make_participant_similarity_figure("All", initial_sim)

# -----------------------------
# 3. DASH APP
# -----------------------------
app = Dash(__name__)
server = app.server  # important for gunicorn

model_options = [{"label": MODEL_LABELS[k], "value": k} for k in PATHS.keys()]

color_by_options = [
    {"label": "Type (Driver vs RFD)", "value": "type"},
    {"label": "Drivers Theme", "value": "drivers_theme"},
    {"label": "RFD Code", "value": "rfd_code"},
]

type_options = [{"label": "All", "value": "All"}] + [
    {"label": display_map.get(t, t), "value": t} for t in sorted(initial_df["type"].unique())
]

id_options = [{"label": "All participants", "value": "All"}] + [
    {"label": pid, "value": pid} for pid in sorted(initial_df["id"].unique())
]

app.layout = html.Div(
    style={
        "fontFamily": "Arial, sans-serif", "padding": "16px",
        "backgroundColor": "#fafafa", "maxWidth": "1400px", "margin": "0 auto",
    },
    children=[
        html.H1("Exploring the Semantic Similarity of CAMS Constructs", style={"marginBottom": "6px"}),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "260px minmax(0, 1fr)",
                "gap": "16px", "alignItems": "start",
            },
            children=[
                # LEFT PANEL
                html.Div(
                    style={
                        "backgroundColor": "white", "padding": "14px",
                        "borderRadius": "10px", "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
                    },
                    children=[
                        html.H3("Controls", style={"marginTop": "0"}),

                        html.Label("Embedding model"),
                        dcc.Dropdown(
                            id="model-filter",
                            options=model_options,
                            value="mpnet",
                            clearable=False,
                        ),

                        html.Br(),

                        html.Label("Color 3D points by"),
                        dcc.Dropdown(
                            id="color-by-filter",
                            options=color_by_options,
                            value="type",
                            clearable=False,
                        ),

                        html.Br(),

                        html.Label("Filter by type"),
                        dcc.Dropdown(id="type-filter", options=type_options, value="All", clearable=False),

                        html.Br(),

                        html.Label("Filter by participant"),
                        dcc.Dropdown(id="id-filter", options=id_options, value="All", clearable=False),

                        html.Br(),

                        html.Div(
                            children=[
                                html.P(html.Strong("About This Page"), style={"marginBottom": "8px"}),
                                html.P(
                                    "This dashboard explores semantic similarity among CAMS Drivers and Reasons for Dying (RFD) using natural language processing (NLP). "
                                    "The 3D visualization shows individual RFD and driver responses represented in 3D space. The bar chart shows the average within-person relationships between constructs. "
                                    "Use the model dropdown to compare a general-purpose embedding model (MPNet) against a mental-health-domain-specific model (MentalBERT), and the 'Color 3D points by' dropdown to see how the qualitative coding frameworks (applied to both Drivers and RFD text) align with the embedding space.",
                                    style={"marginBottom": "8px"},
                                ),
                                html.P(html.Strong("How to interpret it"), style={"marginBottom": "6px"}),
                                html.Ul([
                                    html.Li("Each point in the 3D plot represents one response (Driver or RFD)."),
                                    html.Li("Points that appear closer together represent responses that are more similar in meaning."),
                                    html.Li("The 3D axes are not directly interpretable units. They are a visualization of high-dimensional semantic relationships."),
                                    html.Li("Every response was coded with BOTH the Drivers framework and the RFD framework, so you can color by either scheme regardless of response type."),
                                    html.Li("Grey points are 'Uncoded' (no matching qualitative code)"),
                                    html.Li("The participant-level bar chart shows average cosine similarity, where higher values indicate greater semantic overlap."),
                                ], style={"paddingLeft": "20px", "marginTop": "0", "marginBottom": "0"}),
                            ],
                            style={
                                "backgroundColor": "#f8f9fb", "padding": "12px", "borderRadius": "8px",
                                "fontSize": "13px", "lineHeight": "1.45", "border": "1px solid #e3e6eb",
                            },
                        ),
                    ],
                ),

                # RIGHT SIDE: 3D plot on top, detail panels underneath
                html.Div(
                    style={"display": "flex", "flexDirection": "column", "gap": "16px", "minWidth": "0"},
                    children=[
                        html.Div(
                            style={
                                "backgroundColor": "white", "padding": "10px",
                                "borderRadius": "10px", "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
                            },
                            children=[
                                html.H3("3D UMAP Visualization", style={"marginTop": "0"}),
                                dcc.Graph(id="umap-graph", figure=initial_umap_fig, style={"height": "100vh"}),
                            ],
                        ),
                        html.Div(
                            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "alignItems": "start"},
                            children=[
                                html.Div(
                                    id="text-panel",
                                    style={
                                        "backgroundColor": "white", "padding": "14px",
                                        "borderRadius": "10px", "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
                                        "minHeight": "300px",
                                    },
                                    children=[
                                        html.H3("Selected Point", style={"marginTop": "0"}),
                                        html.P("Click a point in the plot to view its full text and metadata."),
                                    ],
                                ),
                                html.Div(
                                    style={
                                        "backgroundColor": "white", "padding": "10px",
                                        "borderRadius": "10px", "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
                                    },
                                    children=[dcc.Graph(id="participant-similarity-graph", figure=initial_participant_fig)],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)

# -----------------------------
# 4. CALLBACK: UPDATE TYPE/ID DROPDOWN OPTIONS WHEN MODEL CHANGES
# -----------------------------
@app.callback(
    Output("type-filter", "options"),
    Output("type-filter", "value"),
    Output("id-filter", "options"),
    Output("id-filter", "value"),
    Input("model-filter", "value"),
)
def update_filter_options(selected_model):
    df = datasets[selected_model]["df"]

    type_opts = [{"label": "All", "value": "All"}] + [
        {"label": display_map.get(t, t), "value": t} for t in sorted(df["type"].unique())
    ]
    id_opts = [{"label": "All participants", "value": "All"}] + [
        {"label": pid, "value": pid} for pid in sorted(df["id"].unique())
    ]

    # Reset both filters to "All" whenever the model changes
    return type_opts, "All", id_opts, "All"


# -----------------------------
# 5. CALLBACK: UPDATE UMAP
# -----------------------------
@app.callback(
    Output("umap-graph", "figure"),
    Input("type-filter", "value"),
    Input("id-filter", "value"),
    Input("model-filter", "value"),
    Input("color-by-filter", "value"),
)
def update_graph(selected_type, selected_id, selected_model, color_by):
    df = datasets[selected_model]["df"]
    filtered_df = df.copy()

    if selected_type != "All":
        filtered_df = filtered_df[filtered_df["type"] == selected_type]

    if selected_id != "All":
        filtered_df = filtered_df[filtered_df["id"] == selected_id]

    return make_umap_figure(filtered_df, df, color_by)


# -----------------------------
# 6. CALLBACK: UPDATE PARTICIPANT SIMILARITY CHART
# -----------------------------
@app.callback(
    Output("participant-similarity-graph", "figure"),
    Input("id-filter", "value"),
    Input("model-filter", "value"),
)
def update_participant_similarity(selected_id, selected_model):
    sim_df = datasets[selected_model]["sim"]
    return make_participant_similarity_figure(selected_id, sim_df)


# -----------------------------
# 7. CALLBACK: SHOW CLICKED TEXT
# -----------------------------
@app.callback(
    Output("text-panel", "children"),
    Input("umap-graph", "clickData"),
    Input("model-filter", "value"),
)
def display_click_data(clickData, selected_model):
    if clickData is None:
        return [
            html.H3("Selected Point", style={"marginTop": "0"}),
            html.P("Click a point in the plot to view its full text and metadata."),
        ]

    df = datasets[selected_model]["df"]
    row_id = clickData["points"][0]["customdata"][0]

    if row_id not in df.index:
        return [
            html.H3("Selected Point", style={"marginTop": "0"}),
            html.P("Click a point in the plot to view its full text and metadata."),
        ]

    row = df.loc[row_id]

    panel_children = [
        html.H4("Selected Point", style={"marginTop": "0", "marginBottom": "8px"}),
        html.Div(
            row["text"],
            style={
                "whiteSpace": "pre-wrap", "lineHeight": "1.4", "backgroundColor": "#f7f7f7",
                "padding": "10px", "borderRadius": "8px", "border": "1px solid #e0e0e0",
                "maxHeight": "180px", "overflowY": "auto", "marginBottom": "10px",
            },
        ),
        html.Div(
            [
                html.Span([html.Strong("Participant: "), str(row["id"])]),
                html.Span("  ·  "),
                html.Span([html.Strong("Type: "), row["type_label"]]),
            ],
            style={"fontSize": "13px", "marginBottom": "6px"},
        ),
        html.Div(
            [
                html.Span([html.Strong("Drivers theme: "), row["drivers_theme"]]),
                html.Br(),
                html.Span([html.Strong("Drivers code: "), row["drivers_code"]]),
                html.Br(),
                html.Span([html.Strong("RFD code: "), row["rfd_code"]]),
            ],
            style={"fontSize": "13px", "lineHeight": "1.5"},
        ),
    ]

    return panel_children


# -----------------------------
# 9. RUN APP
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
