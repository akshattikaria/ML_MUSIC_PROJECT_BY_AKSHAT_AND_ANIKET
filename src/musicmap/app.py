"""Interactive MusicMap demo (optional ``app`` extra).

Launch with ``musicmap app`` or ``streamlit run src/musicmap/app.py``.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from musicmap.data import load_dataset
from musicmap.discovery import nearest_to_features

ROOT = Path(os.environ.get("MUSICMAP_ROOT", Path(__file__).resolve().parents[2]))
REPORTS, MODELS = ROOT / "reports", ROOT / "models"
GENRE_COLORS = dict(zip(
    ["Electronic", "Experimental", "Folk", "Hip-Hop", "Instrumental", "International", "Pop", "Rock"],
    ["#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc", "#ca9161", "#fbafe4", "#949494"],
))

st.set_page_config(page_title="MusicMap", page_icon="🎵", layout="wide")


@st.cache_resource
def resources():
    dataset = load_dataset(ROOT)
    model = joblib.load(MODELS / "best_model.joblib")
    selection = json.loads((REPORTS / "selection.json").read_text())
    comparison = pd.read_csv(REPORTS / "model_comparison.csv")
    song_map = pd.read_csv(REPORTS / "song_map.csv")
    return dataset, model, selection, comparison, song_map


def probability_chart(probabilities: np.ndarray, classes, actual: str | None = None):
    frame = pd.DataFrame({"genre": classes, "probability": probabilities}).sort_values("probability")
    frame["kind"] = np.where(frame["genre"].eq(actual), "actual genre", "other")
    chart = px.bar(
        frame, x="probability", y="genre", orientation="h", color="kind",
        color_discrete_map={"actual genre": "#50a995", "other": "#255f85"},
        range_x=[0, 1], text=frame["probability"].map("{:.0%}".format), height=330,
    )
    chart.update_layout(margin=dict(l=0, r=10, t=10, b=0), showlegend=actual is not None,
                        legend_title_text="", xaxis_tickformat=".0%", yaxis_title="")
    return chart


def show_prediction(model, features: pd.DataFrame, actual: str | None = None):
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)
    best = int(np.argmax(probabilities))
    left, right = st.columns([1, 2])
    with left:
        st.metric("Predicted genre", classes[best], f"{probabilities[best]:.0%} confidence", delta_color="off")
        if actual is not None:
            if classes[best] == actual:
                st.success(f"Correct — dataset label: **{actual}**")
            else:
                rank = int(np.sum(probabilities > probabilities[classes.index(actual)])) + 1
                st.warning(f"Dataset label: **{actual}** (ranked #{rank} by the model)")
        if probabilities[best] < 0.45:
            st.caption("Low confidence: this track sits between several genres.")
    with right:
        st.plotly_chart(probability_chart(probabilities, classes, actual), width="stretch")


def main():
    try:
        dataset, model, selection, comparison, song_map = resources()
    except FileNotFoundError as error:
        st.error(f"Missing project file: {error}. Run `musicmap improve` first.")
        return
    champion = selection["champion"]
    chosen = comparison.loc[comparison["model"].eq(champion)].iloc[0]

    st.title("🎵 MusicMap")
    st.caption("Music genre classification and similarity on FMA-small · artist-disjoint evaluation")
    a, b, c, d = st.columns(4)
    a.metric("Champion model", champion)
    b.metric("Test accuracy", f"{chosen.test_accuracy:.1%}", "chance 12.5%", delta_color="off")
    c.metric("Test macro F1", f"{chosen.test_f1_macro:.3f}")
    if "test_top3_accuracy" in chosen and pd.notna(chosen.test_top3_accuracy):
        d.metric("Top-3 accuracy", f"{chosen.test_top3_accuracy:.1%}")

    tab_track, tab_audio, tab_map, tab_results = st.tabs(
        ["🔎 Explore a track", "⬆️ Classify your audio", "🗺️ Music map", "📊 Results"]
    )

    with tab_track:
        meta = dataset.metadata
        split = st.radio("Partition", ["test", "validation", "training"], horizontal=True,
                         help="Test tracks were never used to fit or choose the model's settings; "
                              "the saved model was refitted on training + validation.")
        pool = meta.loc[meta["split"].eq(split)]
        genre = st.selectbox("Filter by genre", ["All genres", *sorted(pool["genre"].unique())])
        if genre != "All genres":
            pool = pool.loc[pool["genre"].eq(genre)]
        labels = {tid: f"{row.title} — {row.artist}  (#{tid})" for tid, row in pool.iterrows()}
        track_id = st.selectbox("Track", list(labels), format_func=labels.get)
        if track_id is not None:
            row = meta.loc[track_id]
            st.subheader(f"{row.title}")
            st.caption(f"{row.artist} · track {track_id} · {split} partition")
            if split != "test":
                st.info("This track was used to fit the final model, so the prediction is in-sample.")
            show_prediction(model, dataset.X.loc[[track_id]], actual=row.genre)
            st.markdown("**Most similar songs by other artists** (training catalog, PCA audio space)")
            neighbors = nearest_to_features(
                dataset, MODELS, dataset.X.loc[[track_id]], k=8,
                exclude_ids=(track_id,), exclude_artist=row.artist_id,
            )
            st.dataframe(neighbors.reset_index().round({"distance": 2}), hide_index=True, width="stretch")

    with tab_audio:
        st.write("Upload a music file (MP3, WAV, OGG, FLAC or M4A). MusicMap extracts the same 518 "
                 "descriptors as FMA from a 30-second excerpt in the middle of the file.")
        st.caption("Experimental: modern audio libraries compute slightly different values from the 2017 "
                   "FMA extraction, so treat the result as a demonstration rather than a benchmark. "
                   "M4A/AAC requires FFmpeg on the machine running MusicMap.")
        upload = st.file_uploader("Audio file", type=["mp3", "wav", "ogg", "flac", "m4a"])
        if upload is not None:
            try:
                from musicmap.audio import extract_file
            except ImportError:
                st.error('Install the audio extra first: python -m pip install -e ".[audio]"')
                return
            suffix = Path(upload.name).suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(upload.getbuffer())
                path = Path(handle.name)
            try:
                with st.spinner("Extracting audio descriptors…"):
                    features = extract_file(path).to_frame().T[list(model.feature_names_in_)]
                audio_formats = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg",
                                 ".flac": "audio/flac", ".m4a": "audio/mp4"}
                st.audio(upload.getvalue(), format=audio_formats.get(suffix.lower(), upload.type))
                show_prediction(model, features)
                st.markdown("**Most similar songs in the training catalog**")
                st.dataframe(nearest_to_features(dataset, MODELS, features, k=8).reset_index().round({"distance": 2}),
                             hide_index=True, width="stretch")
            except Exception as error:  # noqa: BLE001 - show decoding problems to the user
                st.error(f"Could not process this file: {error}")
            finally:
                path.unlink(missing_ok=True)

    with tab_map:
        st.write("Every point is a song, placed by the first two principal components of its audio "
                 "descriptors (fitted on training tracks only). Colours show genres; labels are not "
                 "used to place points.")
        genres = st.multiselect("Genres", sorted(song_map["genre"].unique()), default=sorted(song_map["genre"].unique()))
        shown = song_map.loc[song_map["genre"].isin(genres)]
        figure = px.scatter(shown, x="pc1", y="pc2", color="genre", color_discrete_map=GENRE_COLORS,
                            hover_name="title", hover_data=["artist", "track_id", "split"],
                            opacity=0.6, height=640)
        figure.update_traces(marker_size=5)
        st.plotly_chart(figure, width="stretch")

    with tab_results:
        columns = [c for c in ("model", "cv_mean_f1", "validation_f1_macro", "test_accuracy",
                               "test_f1_macro", "test_top3_accuracy") if c in comparison]
        st.dataframe(comparison[columns].round(4), hide_index=True, width="stretch")
        for name, caption in (("model_comparison.png", "Validation and test scores"),
                              ("improvement_comparison.png", "Progress across runs"),
                              ("confusion_matrices.png", "Confusion matrices"),
                              ("feature_importance.png", "Feature-family importance"),
                              ("confidence.png", "Accuracy by confidence")):
            path = REPORTS / "figures" / name
            if path.exists():
                st.image(str(path), caption=caption)


main()
