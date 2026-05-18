"""Reusable Plotly palette chart components for Streamlit."""
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict
import streamlit as st


def _hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """Convert a 6-digit hex color string to an rgba() string Plotly accepts."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def render_dominant_pie_chart(dominant_colors: List[Dict]) -> None:
    """Render a donut chart of dominant color percentages."""
    labels = [f"{c['color_name']} ({c['percentage']}%)" for c in dominant_colors]
    values = [c["percentage"] for c in dominant_colors]
    colors = [c["hex"] for c in dominant_colors]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        marker=dict(colors=colors, line=dict(color="#1a1a2e", width=2)),
        textfont=dict(size=13),
        hovertemplate="<b>%{label}</b><br>Coverage: %{value}%<extra></extra>",
    )])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        margin=dict(t=20, b=20, l=20, r=20),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_palette_swatches(colors: List[Dict]) -> None:
    """Render horizontal color swatches for a palette."""
    cols = st.columns(len(colors))
    for col, c in zip(cols, colors):
        r = c["rgb"]["R"]; g = c["rgb"]["G"]; b = c["rgb"]["B"]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        txt = "#000" if lum > 128 else "#fff"
        with col:
            st.markdown(
                f"""
                <div style="
                    background:{c['hex']};
                    border-radius:10px;
                    padding:24px 6px 10px;
                    text-align:center;
                    box-shadow:0 2px 12px rgba(0,0,0,0.4);
                    margin-bottom:4px;
                ">
                    <div style="color:{txt};font-size:0.7rem;font-weight:600;">
                        {c['hex']}
                    </div>
                    <div style="color:{txt};font-size:0.65rem;opacity:0.8;margin-top:2px;">
                        {c.get('color_name','')[:14]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_color_bar(dominant_colors: List[Dict]) -> None:
    """Render a proportional horizontal color bar."""
    total = sum(c["percentage"] for c in dominant_colors) or 1
    segments = "".join(
        f'<div style="flex:{c["percentage"]/total:.4f};background:{c["hex"]};'
        f'height:100%;border-radius:{4 if i==0 else 0}px {4 if i==len(dominant_colors)-1 else 0}px '
        f'{4 if i==len(dominant_colors)-1 else 0}px {4 if i==0 else 0}px;" '
        f'title="{c["color_name"]} — {c["percentage"]}%"></div>'
        for i, c in enumerate(dominant_colors)
    )
    st.markdown(
        f'<div style="display:flex;width:100%;height:42px;border-radius:4px;'
        f'overflow:hidden;box-shadow:0 2px 10px rgba(0,0,0,0.4);margin-bottom:12px;">'
        f'{segments}</div>',
        unsafe_allow_html=True,
    )


def render_color_timeline(frame_results: List[Dict]) -> None:
    """Render a Plotly line chart of color presence over video time."""
    if not frame_results:
        st.info("No frame data to plot.")
        return

    # Collect all color names
    all_colors = set()
    for fr in frame_results:
        all_colors.update(fr.get("detections", {}).keys())

    fig = go.Figure()
    COLOR_MAP = {
        "red": "#FF4444", "blue": "#4488FF", "green": "#44BB44",
        "yellow": "#FFCC00", "orange": "#FF8800", "purple": "#9933CC",
        "pink": "#FF66AA", "cyan": "#00CCCC", "white": "#CCCCCC", "black": "#888888",
    }

    timestamps = [fr["timestamp_sec"] for fr in frame_results]
    for color in sorted(all_colors):
        counts = [fr.get("detections", {}).get(color, 0) for fr in frame_results]
        fig.add_trace(go.Scatter(
            x=timestamps, y=counts,
            name=color.capitalize(),
            mode="lines",
            line=dict(color=COLOR_MAP.get(color, "#AAAAAA"), width=2),
            fill="tozeroy",
            fillcolor=COLOR_MAP.get(color, "#AAAAAA") + "33",
            hovertemplate=f"<b>{color}</b><br>Time: %{{x:.1f}}s<br>Regions: %{{y}}<extra></extra>",
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,13,30,0.8)",
        font=dict(color="white"),
        xaxis=dict(title="Time (seconds)", gridcolor="#333355"),
        yaxis=dict(title="Detected Regions", gridcolor="#333355"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.4),
        height=400,
        margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)
