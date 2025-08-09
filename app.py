import os
import re
import json
import itertools
from io import StringIO
from datetime import datetime

import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Health Assist (MVP)", page_icon="🩺", layout="centered")

# -----------------------
# Safety banner
# -----------------------
st.markdown(
    """
**Disclaimer:** This tool is for educational purposes only and **not** medical advice.  
If you have emergency symptoms (e.g., chest pain, trouble breathing, one-sided weakness), call emergency services.
"""
)
st.title("🩺 Patient Support & Assistance")

# =========================
# CONFIG & SECRETS
# =========================
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))  # optional
USE_LIVE_RXNORM = True   # set False to rely only on local synonyms

# =========================
# CACHING HELPERS
# =========================
@st.cache_data(show_spinner=False, ttl=60*60)
def cached_get(url, params=None, headers=None, timeout=25):
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return {"url": r.url, "json": r.json()}

@st.cache_data(show_spinner=False, ttl=60*60)
def cached_post(url, payload=None, headers=None, timeout=25):
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    return {"url": r.url, "json": r.json()}

# =========================
# A1) SYMPTOM TRIAGE SETUP
# =========================

# --- MedlinePlus links with fallback (so app never breaks) ---
DEFAULT_LINKS_CSV = """symptom,url
headache,https://medlineplus.gov/headache.html
fever,https://medlineplus.gov/fever.html
chest pain,https://medlineplus.gov/chestpain.html
shortness of breath,https://medlineplus.gov/dyspnea.html
cough,https://medlineplus.gov/cough.html
sore throat,https://medlineplus.gov/sorethroat.html
vomiting,https://medlineplus.gov/nauseaandvomiting.html
diarrhea,https://medlineplus.gov/diarrhea.html
abdominal pain,https://medlineplus.gov/abdominalpain.html
dizziness,https://medlineplus.gov/dizzinessandvertigo.html
fatigue,https://medlineplus.gov/fatigue.html
back pain,https://medlineplus.gov/backpain.html
rash,https://medlineplus.gov/rash.html
ear pain,https://medlineplus.gov/earinfections.html
runny nose,https://medlineplus.gov/commoncold.html
loss of smell,https://medlineplus.gov/coronavirusdisease2019covid19.html
loss of taste,https://medlineplus.gov/coronavirusdisease2019covid19.html
"""

def load_symptom_links():
    try:
        df = pd.read_csv("symptom_links.csv")
        if df.empty or not {"symptom", "url"}.issubset(df.columns):
            raise ValueError("symptom_links.csv is empty or missing required columns.")
        return dict(zip(df["symptom"].str.lower(), df["url"]))
    except Exception as e:
        st.warning(f"Using built-in default links because symptom_links.csv could not be loaded ({e}).")
        df = pd.read_csv(StringIO(DEFAULT_LINKS_CSV))
        return dict(zip(df["symptom"].str.lower(), df["url"]))

SYMPTOM_LINKS = load_symptom_links()

# Core vocab (expand later)
SYMPTOMS = [
    "headache","fever","chest pain","shortness of breath","cough","sore throat",
    "vomiting","diarrhea","abdominal pain","dizziness","fatigue","back pain",
    "rash","ear pain","runny nose","loss of smell","loss of taste"
]
RED_FLAGS = [
    "chest pain","shortness of breath","one-sided weakness","confusion","fainting",
    "uncontrolled bleeding","severe allergic reaction","blue lips","worst headache",
    "stiff neck with fever","vision loss","severe abdominal pain","blood in stool","black tarry stool"
]
SEVERITY_WORDS = {"mild":1,"moderate":2,"severe":3,"worst":3}
DURATION_RE = re.compile(r"(\b\d+\b)\s*(hour|hours|day|days|week|weeks)", re.I)

def detect_symptoms_rules(text: str):
    t = text.lower()
    seen, out = set(), []
    for s in SYMPTOMS:
        if s in t and s not in seen:
            out.append(s); seen.add(s)
    return out

# --- AI symptom boost (optional) ---
HF_ZS_MODEL = "facebook/bart-large-mnli"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_ZS_MODEL}"

def ai_detect_symptoms(text: str, candidates: list[str], top_k: int = 6):
    if not HF_TOKEN:
        return []
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": text, "parameters": {"candidate_labels": candidates, "multi_label": True}}
    try:
        resp = cached_post(HF_API_URL, payload=payload, headers=headers)
        data = resp["json"]
        labels, scores = data.get("labels", []), data.get("scores", [])
        ranked = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
    except Exception:
        return []

def extract_context(text: str):
    t = text.lower()
    # severity
    severity_score, severity_label = 0, "unknown"
    for w, s in SEVERITY_WORDS.items():
        if w in t and s > severity_score:
            severity_score, severity_label = s, w
    # duration → approx days
    duration_days = 0
    m = DURATION_RE.search(text)
    if m:
        n = int(m.group(1)); unit = m.group(2).lower()
        if unit.startswith("hour"):
            duration_days = 1 if n >= 24 else 0
        elif unit.startswith("day"):
            duration_days = n
        elif unit.startswith("week"):
            duration_days = n * 7
    # red flags
    red_hits = [rf for rf in RED_FLAGS if rf in t]
    return {"severity_label":severity_label,"severity_score":severity_score,
            "duration_days":duration_days,"red_hits":red_hits}

def triage(text: str):
    symptoms = detect_symptoms_rules(text)
    # AI assist (if token provided)
    ai_hits = ai_detect_symptoms(text, SYMPTOMS, top_k=6)
    for label, score in ai_hits:
        if score >= 0.35 and label not in symptoms:
            symptoms.append(label)

    ctx = extract_context(text)
    if ctx["red_hits"]:
        return {"level":"URGENT",
                "reason":"Detected red-flag symptom(s): " + ", ".join(ctx["red_hits"]),
                "symptoms":symptoms,"context":ctx}
    if ctx["severity_score"] >= 2 and ctx["duration_days"] >= 2:
        return {"level":"SOON",
                "reason":f"Moderate/severe symptoms persisting for {ctx['duration_days']} day(s).",
                "symptoms":symptoms,"context":ctx}
    return {"level":"ROUTINE",
            "reason":"No red flags detected; mild or short-duration symptoms.",
            "symptoms":symptoms,"context":ctx}

# ==============================
# A2) MEDS + REMINDERS SETUP
# ==============================

# --- Local synonyms fallback (works offline) ---
def load_synonyms_map():
    try:
        df = pd.read_csv("meds_synonyms.csv")
        df["alias"] = df["alias"].str.lower().str.strip()
        df["canonical"] = df["canonical"].str.lower().str.strip()
        m = dict(zip(df["alias"], df["canonical"]))
        for c in df["canonical"].unique():
            m.setdefault(c, c)
        return m
    except Exception:
        return {
            "tylenol":"acetaminophen","paracetamol":"acetaminophen",
            "advil":"ibuprofen","motrin":"ibuprofen","aleve":"naproxen",
            "zoloft":"sertraline","prozac":"fluoxetine","celexa":"citalopram","lexapro":"escitalopram",
            "warfarin":"warfarin","coumadin":"warfarin","aspirin":"aspirin","ibuprofen":"ibuprofen",
            "naproxen":"naproxen","sertraline":"sertraline","fluoxetine":"fluoxetine",
            "citalopram":"citalopram","escitalopram":"escitalopram","acetaminophen":"acetaminophen","linezolid":"linezolid",
        }

SYN_MAP = load_synonyms_map()

# --- RxNorm live normalization (preferred when available) ---
RXNAV = "https://rxnav.nlm.nih.gov/REST"

@st.cache_data(show_spinner=False, ttl=60*60)
def rxnorm_lookup(name: str):
    """Return (rxcui, canonical_name) or (None, None)"""
    try:
        # approximate term → get top candidate
        approx = cached_get(f"{RXNAV}/approximateTerm.json", params={"term": name, "maxEntries": 1})
        cand = approx["json"]["approximateGroup"]["candidate"][0]
        rxcui = cand["rxcui"]
        props = cached_get(f"{RXNAV}/rxcui/{rxcui}/properties.json")
        cname = props["json"].get("properties", {}).get("name", name)
        return str(rxcui), cname
    except Exception:
        return None, None

def normalize_meds(user_text: str):
    """Normalize user input to canonical names and (optionally) RXCUIs."""
    raw = [x.strip() for x in user_text.split(",") if x.strip()]
    meds = []
    for r in raw:
        alias = r.lower()
        # Try live RxNorm first
        rxcui, cname = (None, None)
        if USE_LIVE_RXNORM:
            rxcui, cname = rxnorm_lookup(alias)
        if not cname:
            cname = SYN_MAP.get(alias, alias)
        meds.append({"input": r, "canonical": cname.lower(), "rxcui": rxcui})
    # de-dupe by canonical
    seen, uniq = set(), []
    for m in meds:
        key = m["canonical"]
        if key not in seen:
            uniq.append(m); seen.add(key)
    return uniq

# --- High-priority DDI table (rule layer so we always catch big risks) ---
def load_ddi_table():
    try:
        df = pd.read_csv("onc_high_priority_ddi.csv")
        for c in ["drug_a","drug_b"]:
            df[c] = df[c].str.lower().str.strip()
        return df
    except Exception:
        data = StringIO("""drug_a,drug_b,severity,note,source
sertraline,linezolid,major,Risk of serotonin syndrome; avoid or monitor closely,https://www.accessdata.fda.gov/
warfarin,ibuprofen,major,Bleeding risk increases; avoid or monitor INR closely,https://www.accessdata.fda.gov/
warfarin,aspirin,major,Bleeding risk increases; avoid or monitor INR closely,https://www.accessdata.fda.gov/
ibuprofen,naproxen,moderate,Avoid duplicate NSAIDs; higher GI/renal risk,https://www.accessdata.fda.gov/
""")
        df = pd.read_csv(data)
        for c in ["drug_a","drug_b"]:
            df[c] = df[c].str.lower().str.strip()
        return df

DDI = load_ddi_table()

def check_interactions(meds_norm: list[dict]):
    """Find rule-based high-priority DDIs among normalized meds."""
    names = [m["canonical"] for m in meds_norm]
    results = []
    if len(names) < 2 or DDI.empty:
        return results
    for a, b in itertools.combinations(sorted(names), 2):
        hit = DDI[
            ((DDI["drug_a"] == a) & (DDI["drug_b"] == b)) |
            ((DDI["drug_a"] == b) & (DDI["drug_b"] == a))
        ]
        if not hit.empty:
            row = hit.iloc[0]
            results.append({
                "a": a, "b": b,
                "severity": str(row["severity"]).lower(),
                "note": row.get("note",""),
                "source": row.get("source",""),
            })
    return results

# --- openFDA Drug Label evidence (interactions/warnings/contraindications) ---
OPENFDA_URL = "https://api.fda.gov/drug/label.json"

@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_drug_label_text(drug_query: str, limit: int = 1):
    try:
        params = {"search": f'openfda.brand_name:"{drug_query}" OR openfda.generic_name:"{drug_query}"', "limit": limit}
        resp = cached_get(OPENFDA_URL, params=params)
        results = resp["json"].get("results", [])
        if not results:
            return {}
        d = results[0]
        return {
            "id": d.get("id", ""),
            "brand": (d.get("openfda", {}) or {}).get("brand_name", [""])[0] if d.get("openfda") else "",
            "generic": (d.get("openfda", {}) or {}).get("generic_name", [""])[0] if d.get("openfda") else "",
            "interactions": " ".join(d.get("drug_interactions", []) or [])[:4000],
            "warnings": " ".join(d.get("warnings", []) or [])[:4000],
            "contraindications": " ".join(d.get("contraindications", []) or [])[:4000],
        }
    except Exception:
        return {}

def make_ics(title, description, time_hhmm, days):
    hh, mm = map(int, time_hhmm.split(":"))
    dow_map = {"Mon":"MO","Tue":"TU","Wed":"WE","Thu":"TH","Fri":"FR","Sat":"SA","Sun":"SU"}
    byday = ",".join(dow_map[d] for d in days) if days else "MO,TU,WE,TH,FR"
    dtstart = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    ics = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//AI Health Assist//Reminders//EN
BEGIN:VEVENT
DTSTART:{dtstart}
RRULE:FREQ=WEEKLY;BYDAY={byday};BYHOUR={hh};BYMINUTE={mm};BYSECOND=0
SUMMARY:{title}
DESCRIPTION:{description}
END:VEVENT
END:VCALENDAR
"""
    return ics

# =========
#   UI
# =========

tab1, tab2 = st.tabs(["🩺 Symptoms", "💊 Medications"])

with tab1:
    st.subheader("Describe your symptoms")
    user_text = st.text_area(
        "Type in plain English (e.g., 'Severe headache and fever for 3 days')",
        height=140,
        key="symptoms_input",
    )

    if st.button("Check triage", type="primary", key="btn_triage"):
        if not user_text.strip():
            st.error("Please enter a description.")
        else:
            result = triage(user_text)
            level, reason = result["level"], result["reason"]

            if level == "URGENT":
                st.error(f"🔴 URGENT — {reason}")
            elif level == "SOON":
                st.warning(f"🟠 SOON — {reason}")
            else:
                st.success(f"🟢 ROUTINE — {reason}")

            if result["symptoms"]:
                st.subheader("Learn more about your symptoms:")
                for s in result["symptoms"]:
                    url = SYMPTOM_LINKS.get(s.lower())
                    if url:
                        st.markdown(f"- [{s.title()}]({url})")

            with st.expander("Why this result? (details)"):
                st.json(result)

with tab2:
    st.subheader("Medication Interaction Checker")
    meds_raw = st.text_input(
        "Enter medication names (comma-separated)",
        placeholder="e.g., sertraline, ibuprofen",
        key="meds_input",
    )
    st.caption("Tip: brand or generic names are okay. We normalize names (RxNorm when possible).")

    if st.button("Normalize & Check Interactions", type="primary", key="btn_meds"):
        meds = normalize_meds(meds_raw)
        if not meds:
            st.error("Please enter at least one medication name.")
        else:
            st.write("**Normalized medications:**")
            st.dataframe(pd.DataFrame(meds), use_container_width=True)

            hits = check_interactions(meds)
            if hits:
                st.subheader("Potential Interactions (high-priority rules)")
                for h in hits:
                    sev = h["severity"].capitalize()
                    st.write(f"- **{h['a']} × {h['b']}** — *{sev}*")
                    if h.get("note"):
                        st.caption(h["note"])
                    if h.get("source"):
                        st.caption(f"[Source]({h['source']})")
            else:
                st.success("No interactions found in the current high-priority list.")

            # Official label evidence (openFDA)
            st.write("---")
            st.subheader("Official label info (openFDA)")
            for m in meds:
                q = m["canonical"]
                info = fetch_drug_label_text(q, limit=1)
                if info:
                    brand = info.get("brand") or q
                    st.markdown(f"**{brand.title()}**")
                    if info.get("interactions"):
                        st.caption("Interactions (from FDA label):")
                        text = info["interactions"]
                        st.write(text[:700] + ("..." if len(text) > 700 else ""))
                    if info.get("warnings"):
                        with st.expander("Warnings"):
                            st.write(info["warnings"])
                    if info.get("contraindications"):
                        with st.expander("Contraindications"):
                            st.write(info["contraindications"])
                else:
                    st.caption(f"No FDA label text found for **{q}**.")

    st.write("---")
    st.subheader("Create a medication reminder")
    with st.form("reminder_form"):
        med_choice = st.text_input("Medication name", placeholder="e.g., ibuprofen")
        dose = st.text_input("Dose", placeholder="e.g., 400 mg")
        time_str = st.time_input("Reminder time").strftime("%H:%M")
        days = st.multiselect("Days of week", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                              default=["Mon","Tue","Wed","Thu","Fri"])
        submit = st.form_submit_button("Generate calendar (.ics)")

    if submit:
        if not med_choice.strip():
            st.error("Enter a medication name.")
        else:
            title = f"Take {med_choice} {f'({dose})' if dose else ''}".strip()
            desc = "Medication reminder generated by AI Health Assist (educational use)."
            ics = make_ics(title, desc, time_str, days)
            st.success("Calendar file generated. Download and add it to your calendar.")
            st.download_button(
                "Download .ics",
                data=ics.encode("utf-8"),
                file_name=f"reminder_{med_choice.replace(' ','_')}.ics",
                mime="text/calendar",
            )
