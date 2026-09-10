import streamlit as st
import pandas as pd
import numpy as np
from groq import Groq

st.set_page_config(page_title="Renewable Grid Operations Agent", page_icon="⚡", layout="wide")

st.title("⚡ Renewable Grid Operations Agent")
st.caption("AI-assisted decision support for a simplified renewable-integrated power system")

st.info(
    "Upload the supplied 24-hour CSV input file. Python performs the engineering calculations; "
    "the Groq AI agent interprets the results and provides decision support."
)

# ---------- Power-system calculations ----------

def calculate_results(data):
    data = data.copy()

    # Net load: positive = deficit, negative = renewable surplus
    data["renewable_mw"] = data["solar_mw"] + data["wind_mw"]
    data["net_load_mw"] = data["load_mw"] - data["renewable_mw"]

    capacity = float(data["battery_capacity_mwh"].iloc[0])
    max_charge = float(data["max_charge_mw"].iloc[0])
    max_discharge = float(data["max_discharge_mw"].iloc[0])
    soc = float(data["initial_soc_percent"].iloc[0]) / 100.0
    min_soc = float(data["min_soc_percent"].iloc[0]) / 100.0
    max_soc = float(data["max_soc_percent"].iloc[0]) / 100.0
    charge_eff = float(data["charge_efficiency"].iloc[0])
    discharge_eff = float(data["discharge_efficiency"].iloc[0])
    grid_limit = float(data["grid_import_limit_mw"].iloc[0])

    soc_values = []
    battery_charge = []
    battery_discharge = []
    grid_import = []
    curtailment = []
    deficit = []

    for net_load in data["net_load_mw"]:
        charge = 0.0
        discharge = 0.0
        curtailed = 0.0
        imported = 0.0

        energy = soc * capacity

        if net_load < 0:
            surplus = -net_load
            available_room = max(0.0, (max_soc * capacity) - energy)
            charge = min(surplus, max_charge, available_room / charge_eff)
            energy += charge * charge_eff
            curtailed = max(0.0, surplus - charge)

        else:
            available_energy = max(0.0, energy - min_soc * capacity)
            discharge = min(net_load, max_discharge, available_energy * discharge_eff)
            energy -= discharge / discharge_eff
            imported = max(0.0, net_load - discharge)

        soc = energy / capacity
        soc_values.append(soc * 100)
        battery_charge.append(charge)
        battery_discharge.append(discharge)
        grid_import.append(imported)
        curtailment.append(curtailed)
        deficit.append(max(0.0, imported - grid_limit))

    data["battery_charge_mw"] = battery_charge
    data["battery_discharge_mw"] = battery_discharge
    data["battery_soc_percent"] = soc_values
    data["grid_import_mw"] = grid_import
    data["curtailment_mw"] = curtailment
    data["unserved_or_excess_deficit_mw"] = deficit
    data["renewable_penetration_percent"] = (
        data["renewable_mw"] / data["load_mw"] * 100
    )

    return data


def summarize(data):
    critical_deficit = data["unserved_or_excess_deficit_mw"].max()
    return {
        "peak_load_mw": round(data["load_mw"].max(), 1),
        "peak_load_hour": int(data.loc[data["load_mw"].idxmax(), "hour"]),
        "minimum_net_load_mw": round(data["net_load_mw"].min(), 1),
        "minimum_net_load_hour": int(data.loc[data["net_load_mw"].idxmin(), "hour"]),
        "maximum_net_load_mw": round(data["net_load_mw"].max(), 1),
        "maximum_net_load_hour": int(data.loc[data["net_load_mw"].idxmax(), "hour"]),
        "minimum_battery_soc_percent": round(data["battery_soc_percent"].min(), 1),
        "minimum_battery_soc_hour": int(data.loc[data["battery_soc_percent"].idxmin(), "hour"]),
        "maximum_grid_import_mw": round(data["grid_import_mw"].max(), 1),
        "total_grid_import_mwh": round(data["grid_import_mw"].sum(), 1),
        "total_curtailment_mwh": round(data["curtailment_mw"].sum(), 1),
        "maximum_excess_deficit_mw": round(critical_deficit, 1),
        "average_renewable_penetration_percent": round(data["renewable_penetration_percent"].mean(), 1),
    }


# ---------- Agent tools ----------
# These are deliberately simple. The LLM can call them when it needs an analysis.

def tool_grid_analysis(results):
    s = summarize(results)
    return (
        f"Peak load: {s['peak_load_mw']} MW at hour {s['peak_load_hour']}. "
        f"Maximum net load: {s['maximum_net_load_mw']} MW at hour {s['maximum_net_load_hour']}. "
        f"Minimum net load: {s['minimum_net_load_mw']} MW at hour {s['minimum_net_load_hour']}. "
        f"Average hourly renewable penetration: {s['average_renewable_penetration_percent']}%."
    )

def tool_battery_analysis(results):
    s = summarize(results)
    return (
        f"Minimum battery SOC: {s['minimum_battery_soc_percent']}% at hour "
        f"{s['minimum_battery_soc_hour']}. Maximum grid import: {s['maximum_grid_import_mw']} MW. "
        f"Total grid import: {s['total_grid_import_mwh']} MWh."
    )

def tool_problem_analysis(results):
    s = summarize(results)
    return (
        f"Total renewable curtailment: {s['total_curtailment_mwh']} MWh. "
        f"Maximum grid import: {s['maximum_grid_import_mw']} MW. "
        f"Maximum deficit beyond the configured grid-import limit: "
        f"{s['maximum_excess_deficit_mw']} MW. "
        f"Critical evening/deficit hour is around hour {s['maximum_net_load_hour']}."
    )

def tool_recommendation(results):
    s = summarize(results)
    recommendations = []
    if s["total_curtailment_mwh"] > 0:
        recommendations.append("shift flexible demand or increase storage charging during renewable-surplus hours")
    if s["maximum_grid_import_mw"] > 0:
        recommendations.append("reserve battery energy for high net-load hours and consider flexible-load shifting")
    if s["minimum_battery_soc_percent"] <= 25:
        recommendations.append("avoid unnecessary early battery discharge so energy is available for critical hours")
    if not recommendations:
        recommendations.append("maintain the current operating strategy while monitoring renewable variability")
    return "; ".join(recommendations) + "."

# ---------- File input ----------
uploaded = st.file_uploader("Upload 24-hour grid input CSV", type=["csv"])

if uploaded is None:
    st.warning("Upload the provided grid_input_24h.csv file to start the analysis.")
    st.stop()

try:
    input_data = pd.read_csv(uploaded)
    required = [
        "hour", "load_mw", "solar_mw", "wind_mw",
        "battery_capacity_mwh", "max_charge_mw", "max_discharge_mw",
        "initial_soc_percent", "min_soc_percent", "max_soc_percent",
        "charge_efficiency", "discharge_efficiency", "grid_import_limit_mw"
    ]
    missing = [c for c in required if c not in input_data.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        st.stop()

    results = calculate_results(input_data)
except Exception as e:
    st.error(f"Could not process the CSV: {e}")
    st.stop()

summary = summarize(results)

# ---------- Dashboard ----------
st.subheader("Grid Dashboard")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Peak Load", f"{summary['peak_load_mw']} MW")
c2.metric("Max Net Load", f"{summary['maximum_net_load_mw']} MW")
c3.metric("Minimum Battery SOC", f"{summary['minimum_battery_soc_percent']}%")
c4.metric("Renewable Curtailment", f"{summary['total_curtailment_mwh']} MWh")

st.subheader("24-hour operating profile")
chart_data = results.set_index("hour")[
    ["load_mw", "solar_mw", "wind_mw", "net_load_mw"]
]
st.line_chart(chart_data)

st.subheader("Battery and grid")
battery_chart = results.set_index("hour")[["battery_soc_percent"]]
st.line_chart(battery_chart)

grid_chart = results.set_index("hour")[["grid_import_mw", "curtailment_mw"]]
st.line_chart(grid_chart)

with st.expander("View calculated hourly results"):
    display_cols = [
        "hour", "load_mw", "solar_mw", "wind_mw", "renewable_mw",
        "net_load_mw", "battery_charge_mw", "battery_discharge_mw",
        "battery_soc_percent", "grid_import_mw", "curtailment_mw",
        "renewable_penetration_percent"
    ]
    st.dataframe(results[display_cols], use_container_width=True)

# ---------- AI Agent ----------
st.divider()
st.subheader("🤖 AI Grid Operations Agent")
st.write(
    "The agent can use four engineering analysis tools. "
    "It does not directly control the grid; it provides decision support."
)

question = st.text_input(
    "Ask the Grid Agent",
    placeholder="Why is the system experiencing an evening power deficit?"
)

if st.button("Investigate with AI Agent"):
    if not question.strip():
        question = "Investigate the most important operating problem in this 24-hour system and recommend actions."

    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        st.error("Add GROQ_API_KEY to Streamlit Cloud Secrets before using the AI agent.")
        st.stop()

    client = Groq(api_key=api_key)

    tool_results = {
        "grid_analysis": tool_grid_analysis(results),
        "battery_analysis": tool_battery_analysis(results),
        "problem_analysis": tool_problem_analysis(results),
        "recommendation": tool_recommendation(results),
    }

    # A simple, reliable agent-style loop:
    # The model receives the available tools and can request them.
    tools = [
        {
            "type": "function",
            "function": {
                "name": "grid_analysis",
                "description": "Analyze load, net load and renewable penetration.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "battery_analysis",
                "description": "Analyze battery SOC and grid import.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "problem_analysis",
                "description": "Identify renewable curtailment, power deficits and critical hours.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recommendation",
                "description": "Generate candidate operational actions from the engineering results.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a power-system operations decision-support assistant. "
                "Use the engineering tools when numerical analysis is needed. "
                "Do not invent measurements. Clearly separate calculated facts, "
                "engineering interpretation, and recommendations. "
                "Recommendations are for human review and are not automatic grid control."
            ),
        },
        {"role": "user", "content": question},
    ]

    with st.status("Agent is investigating the grid...", expanded=True) as status:
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=1200,
            )

            assistant_message = response.choices[0].message
            messages.append(assistant_message)

            if assistant_message.tool_calls:
                for call in assistant_message.tool_calls:
                    name = call.function.name
                    st.write(f"🔧 Agent called **{name}**")
                    result = tool_results.get(name, "Tool not available.")
                    st.caption(result)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result,
                        }
                    )

                final_response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=0.2,
                    max_tokens=1500,
                )
                answer = final_response.choices[0].message.content
            else:
                answer = assistant_message.content

            status.update(label="Investigation complete", state="complete")

            st.markdown("### AI Diagnosis and Recommendation")
            st.markdown(answer)

        except Exception as e:
            status.update(label="Agent error", state="error")
            st.error(f"Groq request failed: {e}")

st.divider()
st.caption(
    "Educational prototype: the calculations are simplified and should not be used "
    "for real-time or safety-critical grid operation."
)
