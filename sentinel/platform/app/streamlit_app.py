# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL Console web app (entry point).

    streamlit run sentinel/platform/app/streamlit_app.py --server.headless true \
        --browser.gatherUsageStats false

The API (``python -m sentinel.platform.api``) is needed only for upload-and-predict and live
USGS values (``SENTINEL_API_URL``, default http://127.0.0.1:8000); every other page reads the
local data products, registry and documents directly through the same library as the API.
"""

from __future__ import annotations

import streamlit as st

from sentinel.platform.app import ui
from sentinel.platform.app.views import (
    PAGES,
    early_warning,
    home,
    integrity,
    map_view,
    models_page,
    site_page,
    upload,
)

st.set_page_config(page_title="SENTINEL Console", layout="wide")

PAGES.update(
    {
        "home": st.Page(home.render, title="Home", url_path="home", default=True),
        "map": st.Page(map_view.render, title="Map", url_path="map"),
        "site": st.Page(site_page.render, title="Site", url_path="site"),
        "models": st.Page(models_page.render, title="Models", url_path="models"),
        "upload": st.Page(upload.render, title="Data upload", url_path="upload"),
        "early_warning": st.Page(early_warning.render, title="Early warning", url_path="early-warning"),
        "integrity": st.Page(integrity.render, title="Integrity", url_path="integrity"),
    }
)
nav = st.navigation(list(PAGES.values()))
ui.validated_banner(ui.settings())
nav.run()
